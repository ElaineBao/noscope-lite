#include <sys/mman.h>

#include <chrono>
#include <ctime>
#include <random>
#include <algorithm>
#include <iterator>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/core/platform/cuda.h"

#include "opencv2/opencv.hpp"

#include "tensorflow/noscope-lite/noscope_labeler.h"

namespace noscope {

NoscopeLabeler::NoscopeLabeler(tensorflow::Session *SmallCNN_Session,
              tensorflow::Session *LargeCNN_Session,
              noscope::filters::DifferenceFilter diff_filt,
              const std::string& avg_fname,
              const noscope::NoscopeData& data):
    kNbFrames_(data.kNbFrames_),
    all_data_(data),
    frame_status_(kNbFrames_, kUnprocessed), labels_(kNbFrames_),
    kDifferenceFilter_(diff_filt),
    diff_confidence_(kNbFrames_),
    small_cnn_confidence_(kNbFrames_),
    large_cnn_confidence_(kNbFrames_),
    avg_(NoscopeData::kSmallCNNResol_, CV_32FC3),
    large_session_(LargeCNN_Session),
    small_session_(SmallCNN_Session) {
  std::ifstream is(avg_fname);
  std::istream_iterator<float> start(is), end;
  std::vector<float> nums(start, end);

  if (nums.size() != NoscopeData::kSmallCNNFrameSize_) {
    throw std::runtime_error("nums not right size");
  }
  memcpy(avg_.data, &nums[0], NoscopeData::kSmallCNNFrameSize_ * sizeof(float));

}

void NoscopeLabeler::RunDifferenceFilter(const float lower_thresh,const float upper_thresh,const bool const_ref,const size_t kRef) {
  const std::vector<uint8_t>& kFrameData = all_data_.diff_data_;
  const int kFrameSize = NoscopeData::kDiffFrameSize_;
  #pragma omp parallel for num_threads(kNumThreads_) schedule(static)
  for (size_t i = kDiffDelay_; i < kNbFrames_; i++) {
    const uint8_t *kRefImg = const_ref ?
        &kFrameData[kRef * kFrameSize] :
        &kFrameData[(i - kDiffDelay_) * kFrameSize];  //use block=1, use specific frame; use block=0, use kDiffDelay_ frame.
    float tmp = kDifferenceFilter_.fp(&kFrameData[i * kFrameSize], kRefImg);
    diff_confidence_[i] = tmp;
    if (tmp < lower_thresh) {
      labels_[i] = false;
      frame_status_[i] = kDiffFiltered;
    } else {
      frame_status_[i] = kDiffUnfiltered;
    }
  }
  for (size_t i = kDiffDelay_; i < kNbFrames_; i++)
    if (frame_status_[i] == kDiffUnfiltered)
      small_cnn_frame_ind_.push_back(i);
}

void NoscopeLabeler::PopulateCNNFrames() {
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < kDiffDelay_; i++) small_cnn_frame_ind_.push_back(i);

  const std::vector<float>& kSmallCNNData = all_data_.small_cnn_data_;
  const int kFrameSize = NoscopeData::kSmallCNNFrameSize_;

  using namespace tensorflow;
  const size_t kNbCNNFrames = small_cnn_frame_ind_.size();
  const size_t kNbLoops = (kNbCNNFrames + kMaxSmallCNNBatch_ - 1) / kMaxSmallCNNBatch_;
  const float* avg = (float *) avg_.data;
  for (size_t i = 0; i < kNbLoops; i++) {
    const size_t kImagesToRun =
        std::min(kMaxSmallCNNBatch_, small_cnn_frame_ind_.size() - i * kMaxSmallCNNBatch_);
    Tensor input(DT_FLOAT,
                 TensorShape({kImagesToRun,
                             NoscopeData::kSmallCNNResol_.height,
                             NoscopeData::kSmallCNNResol_.width,
                             kNbChannels_}));
    auto input_mapped = input.tensor<float, 4>();
    float *tensor_start = &input_mapped(0, 0, 0, 0);
    #pragma omp parallel for
    for (size_t j = 0; j < kImagesToRun; j++) {
      const size_t kImgInd = i * kMaxSmallCNNBatch_ + j;
      float *output = tensor_start + j * kFrameSize;
      const float *input = &kSmallCNNData[small_cnn_frame_ind_[kImgInd] * kFrameSize];
      for (size_t k = 0; k < kFrameSize; k++)
        output[k] = input[k] / 255. - avg[k];
    }
    small_cnn_tensors_.push_back(input);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // std::cout << "PopulateCNNFrames time: " << diff.count() << " s" << std::endl;
}

void NoscopeLabeler::RunSmallCNN(const float lower_thresh, const float upper_thresh) {
  using namespace tensorflow;

  // Round up
  const size_t kNbSmallCNNFrames = small_cnn_frame_ind_.size();
  const size_t kNbLoops = (kNbSmallCNNFrames + kMaxSmallCNNBatch_ - 1) / kMaxSmallCNNBatch_;

  for (size_t i = 0; i < kNbLoops; i++) {
    const size_t kImagesToRun =
        std::min(kMaxSmallCNNBatch_, small_cnn_frame_ind_.size() - i * kMaxSmallCNNBatch_);
    auto input = small_cnn_tensors_[i];
    /*cudaHostRegister(&(input.tensor<float, 4>()(0, 0, 0, 0)),
                     kImagesToRun * kFrameSize * sizeof(float),
                     cudaHostRegisterPortable);*/

    std::vector<tensorflow::Tensor> outputs;
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input_img", input},
      // {"keras_learning_phase", learning_phase},
    };

    tensorflow::Status status = small_session_->Run(inputs, {"output_prob"}, {}, &outputs);
    // TF_CHECK_OK(status);
    // FIXME: should probably check the tensor output size here.

    {
      auto output_mapped = outputs[0].tensor<float, 2>();
      for (size_t j = 0; j < kImagesToRun; j++) {
        Status s;
        const int kInd = small_cnn_frame_ind_[i * kMaxSmallCNNBatch_ + j];
        small_cnn_confidence_[kInd] = output_mapped(j, 1);
        if (output_mapped(j, 1) < lower_thresh) {
          labels_[kInd] = false;
          s = kSmallCNNFiltered;
        } else if (output_mapped(j, 1) > upper_thresh) {
          labels_[kInd] = true;
          s = kSmallCNNFiltered;
        } else {
          s = kSmallCNNUnfiltered;
        }
        frame_status_[kInd] = s;
      }
    }
  }
}


void NoscopeLabeler::RunLargeCNN(const int class_id, const float large_cnn_thresh) {
  using namespace tensorflow;
  for (size_t i = 0; i < kNbFrames_; i++) {
      // Run YOLO on every unprocessed frame
      if (frame_status_[i] == kSmallCNNFiltered)
        continue;
      if (frame_status_[i] == kDiffFiltered)
        continue;
      large_cnn_frame_ind_.push_back(i);
  }

  // Round up
  const size_t kNbLargeCNNFrames = large_cnn_frame_ind_.size();
  const size_t kNbLoops = (kNbLargeCNNFrames + kMaxLargeCNNBatch_ - 1) / kMaxLargeCNNBatch_;
  const std::vector<uint8_t>& kLargeCNNData = all_data_.large_cnn_data_;
  const int kLargeCNNFrameSize = NoscopeData::kLargeCNNFrameSize_;

  for (size_t i_loop = 0; i_loop < kNbLoops; i_loop++) {
    const size_t kImagesToRun =
        std::min(kMaxLargeCNNBatch_, large_cnn_frame_ind_.size() - i_loop * kMaxLargeCNNBatch_);
    Tensor input_tensor(DT_UINT8,
                 TensorShape({kImagesToRun,
                             NoscopeData::kLargeCNNResol_.height,
                             NoscopeData::kLargeCNNResol_.width,
                             kNbChannels_}));
    auto input_mapped = input_tensor.tensor<uint8_t, 4>();
    uint8_t *tensor_start = &input_mapped(0, 0, 0, 0);

    #pragma omp parallel for
    for (size_t j_im = 0; j_im < kImagesToRun; j_im++) {
      const size_t kImgInd = i_loop * kMaxLargeCNNBatch_ + j_im;
      uint8_t *output = tensor_start + j_im * kLargeCNNFrameSize;
      const uint8_t *input = &kLargeCNNData[large_cnn_frame_ind_[kImgInd] * kLargeCNNFrameSize];
      for (size_t k = 0; k < kLargeCNNFrameSize; k++)
        output[k] = input[k];
    }

    std::vector<tensorflow::Tensor> outputs;
    tensorflow::Status run_status = large_session_->Run({{"image_tensor", input_tensor}},
                                    {"detection_boxes:0", "detection_scores:0",
                                    "detection_classes:0", "num_detections:0"}, {}, &outputs);

    {
      auto scores = outputs[1].tensor<float, 2>();
      auto classes = outputs[2].tensor<float, 2>();
      auto num_det = outputs[3].tensor<float, 1>();

      for (size_t j_im = 0; j_im < kImagesToRun; j_im++) {
        const int kInd = large_cnn_frame_ind_[i_loop * kMaxLargeCNNBatch_ + j_im];
        float class_score = 0;
        for(size_t i_det = 0; i_det < num_det(j_im);++i_det)
        {
            if (classes(j_im,i_det) == class_id && scores(j_im,i_det) > class_score)
              class_score = scores(j_im, i_det);
        }
        large_cnn_confidence_[kInd] = class_score;
        labels_[kInd] = class_score > large_cnn_thresh;
        frame_status_[kInd] = kLargeCNNLabeled;
      }

    }
  }
}

void NoscopeLabeler::DumpConfidences(const std::string& fname,
                                  const std::string& model_name,
                                  const size_t kSkip,
                                  const size_t kStartFrom,
                                  const bool kSkipSmallCNN,
                                  const float diff_thresh,
                                  const float small_cnn_thresh_lower,
                                  const float small_cnn_thresh_upper,
                                  const float large_cnn_thresh,
                                  const std::vector<double>& runtimes) {
  std::ofstream csv_file;
  csv_file.open(fname);

  std::stringstream rt;
  std::copy(runtimes.begin(), runtimes.end(), std::ostream_iterator<double>(rt, " "));

  csv_file << "# diff_thresh: "  << diff_thresh <<
      ", small_cnn_thresh_lower: " << small_cnn_thresh_lower <<
      ", small_cnn_thresh_upper: " << small_cnn_thresh_upper <<
      ", large_cnn_thresh: " << large_cnn_thresh <<
      ", skip: " << kSkip <<
      ", skip_small_cnn: " << kSkipSmallCNN <<
      ", runtime: " << rt.str() << "\n";
  csv_file << "# model: " << model_name <<
      ", diff_detection: " << kDifferenceFilter_.name << "\n";

  csv_file << "# frame,status,diff_confidence,small_cnn_confidence,large_cnn_confidence,label\n";
  for (size_t i = 0; i < kNbFrames_; i++) {
    csv_file << (kSkip*i + kStartFrom) << ",";
    csv_file << frame_status_[i] << ",";
    csv_file << diff_confidence_[i] << ",";
    csv_file << small_cnn_confidence_[i] << ",";
    csv_file << large_cnn_confidence_[i] << ",";
    csv_file << labels_[i] << "\n";

    // repeat the previous label for skipped frames
    for(size_t j = 0; j < kSkip-1; j++){
      csv_file << (kSkip*i + j + kStartFrom + 1) << ",";
      csv_file << kSkipped << ",";
      csv_file << 0 << ",";
      csv_file << 0 << ",";
      csv_file << 0 << ",";
      csv_file << labels_[i] << "\n";
    }
  }
}

} // namespace noscope
