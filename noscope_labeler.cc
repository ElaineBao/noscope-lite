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
    cnn_confidence_(kNbFrames_),
    yolo_confidence_(kNbFrames_),
    avg_(NoscopeData::kDistResol_, CV_32FC3),
    large_session_(LargeCNN_Session),
    small_session_(SmallCNN_Session) {
  std::ifstream is(avg_fname);
  std::istream_iterator<float> start(is), end;
  std::vector<float> nums(start, end);

  if (nums.size() != NoscopeData::kDistFrameSize_) {
    throw std::runtime_error("nums not right size");
  }
  memcpy(avg_.data, &nums[0], NoscopeData::kDistFrameSize_ * sizeof(float));

}

void NoscopeLabeler::RunDifferenceFilter(const float lower_thresh,const float upper_thresh,const bool const_ref,const size_t kRef) {
  const std::vector<uint8_t>& kFrameData = all_data_.diff_data_;
  const int kFrameSize = NoscopeData::kDiffFrameSize_;
  #pragma omp parallel for num_threads(kNumThreads_) schedule(static)
  for (size_t i = kDiffDelay_; i < kNbFrames_; i++) {
    const uint8_t *kRefImg = const_ref ?
        &kFrameData[kRef * kFrameSize] :
        &kFrameData[(i - kDiffDelay_) * kFrameSize];
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
      cnn_frame_ind_.push_back(i);
}

void NoscopeLabeler::PopulateCNNFrames() {
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < kDiffDelay_; i++) cnn_frame_ind_.push_back(i);

  const std::vector<float>& kDistData = all_data_.dist_data_;
  const int kFrameSize = NoscopeData::kDistFrameSize_;


  using namespace tensorflow;
  const size_t kNbCNNFrames = cnn_frame_ind_.size();
  const size_t kNbLoops = (kNbCNNFrames + kMaxCNNImages_ - 1) / kMaxCNNImages_;
  const float* avg = (float *) avg_.data;
  for (size_t i = 0; i < kNbLoops; i++) {
    const size_t kImagesToRun =
        std::min(kMaxCNNImages_, cnn_frame_ind_.size() - i * kMaxCNNImages_);
    Tensor input(DT_FLOAT,
                 TensorShape({kImagesToRun,
                             NoscopeData::kDistResol_.height,
                             NoscopeData::kDistResol_.width,
                             kNbChannels_}));
    auto input_mapped = input.tensor<float, 4>();
    float *tensor_start = &input_mapped(0, 0, 0, 0);
    #pragma omp parallel for
    for (size_t j = 0; j < kImagesToRun; j++) {
      const size_t kImgInd = i * kMaxCNNImages_ + j;
      float *output = tensor_start + j * kFrameSize;
      const float *input = &kDistData[cnn_frame_ind_[kImgInd] * kFrameSize];
      for (size_t k = 0; k < kFrameSize; k++)
        output[k] = input[k] / 255. - avg[k];
    }
    dist_tensors_.push_back(input);
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;
  // std::cout << "PopulateCNNFrames time: " << diff.count() << " s" << std::endl;
}

void NoscopeLabeler::RunSmallCNN(const float lower_thresh, const float upper_thresh) {
  using namespace tensorflow;

  // Round up
  const size_t kNbCNNFrames = cnn_frame_ind_.size();
  const size_t kNbLoops = (kNbCNNFrames + kMaxCNNImages_ - 1) / kMaxCNNImages_;

  for (size_t i = 0; i < kNbLoops; i++) {
    const size_t kImagesToRun =
        std::min(kMaxCNNImages_, cnn_frame_ind_.size() - i * kMaxCNNImages_);
    auto input = dist_tensors_[i];
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
        const int kInd = cnn_frame_ind_[i * kMaxCNNImages_ + j];
        cnn_confidence_[kInd] = output_mapped(j, 1);
        if (output_mapped(j, 1) < lower_thresh) {
          labels_[kInd] = false;
          s = kDistillFiltered;
        } else if (output_mapped(j, 1) > upper_thresh) {
          labels_[kInd] = true;
          s = kDistillFiltered;
        } else {
          s = kDistillUnfiltered;
        }
        frame_status_[kInd] = s;
      }
    }
  }
}


void NoscopeLabeler::RunLargeCNN(const int class_id, const float conf_thresh) {
  for (size_t i = 0; i < kNbFrames_; i++) {
    // Run LargeCNN on every unprocessed frame
    using namespace tensorflow;
    std::vector<Tensor> outputs;
    Tensor input_tensor(DT_UINT8,
                 TensorShape({1,
                             NoscopeData::kYOLOResol_.height,
                             NoscopeData::kYOLOResol_.width,
                             kNbChannels_}));
    auto input_mapped = input_tensor.tensor<uint8_t, 4>();
    uint8_t *tensor_start = &input_mapped(0, 0, 0, 0);
    const std::vector<uint8_t>& kYoloData = all_data_.yolo_data_;
    const uint8_t *input = &kYoloData[i * all_data_.kYOLOFrameSize_];
    uint8_t *normalized_input = tensor_start; //+ i * all_data_.kYOLOFrameSize_;

    for (size_t k = 0; k < all_data_.kYOLOFrameSize_; k++)
      normalized_input[k] = input[k];
    const Tensor& resized_tensor = input_tensor;
    std::cout<<"image shape:" << resized_tensor.shape().DebugString() << ",tensor type:"<< resized_tensor.dtype();

    std::vector<string> output_layer ={ "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0" };
    tensorflow::Status run_status = (*large_session_)->Run({{"image_tensor", resized_tensor}},
                                   output_layer, {}, &outputs);
    tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
    tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
    tensorflow::TTypes<float>::Flat num_detections = outputs[3].flat<float>();

    float class_score = 0;
    for(size_t j = 0; j < num_detections(0);++j)
    {
        if (classes(j) == class_id && scores(j) > class_score)
          class_score = scores(j);
    }
    yolo_confidence_[i] = class_score;
    labels_[i] = class_score > conf_thresh;
    frame_status_[i] = kYoloLabeled;
  }
}

void NoscopeLabeler::DumpConfidences(const std::string& fname,
                                  const std::string& model_name,
                                  const size_t kSkip,
                                  const bool kSkipSmallCNN,
                                  const float diff_thresh,
                                  const float distill_thresh_lower,
                                  const float distill_thresh_upper,
                                  const std::vector<double>& runtimes) {
  std::ofstream csv_file;
  csv_file.open(fname);

  std::stringstream rt;
  std::copy(runtimes.begin(), runtimes.end(), std::ostream_iterator<double>(rt, " "));

  csv_file << "# diff_thresh: "  << diff_thresh <<
      ", distill_thresh_lower: " << distill_thresh_lower <<
      ", distill_thresh_upper: " << distill_thresh_upper <<
      ", skip: " << kSkip <<
      ", skip_cnn: " << kSkipSmallCNN <<
      ", runtime: " << rt.str() << "\n";
  csv_file << "# model: " << model_name <<
      ", diff_detection: " << kDifferenceFilter_.name << "\n";

  csv_file << "# frame,status,diff_confidence,cnn_confidence,yolo_confidence,label\n";
  for (size_t i = 0; i < kNbFrames_; i++) {
    csv_file << (kSkip*i + 1) << ",";
    csv_file << frame_status_[i] << ",";
    csv_file << diff_confidence_[i] << ",";
    csv_file << cnn_confidence_[i] << ",";
    csv_file << yolo_confidence_[i] << ",";
    csv_file << labels_[i] << "\n";

    // repeat the previous label for skipped frames
    for(size_t j = 0; j < kSkip-1; j++){
      csv_file << (kSkip*i + j + 1) << ",";
      csv_file << kSkipped << ",";
      csv_file << 0 << ",";
      csv_file << 0 << ",";
      csv_file << 0 << ",";
      csv_file << labels_[i] << "\n";
    }
  }
}

} // namespace noscope
