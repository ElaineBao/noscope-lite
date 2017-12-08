#ifndef TENSORFLOW_VUSE_VUSELABELER_H_
#define TENSORFLOW_VUSE_VUSELABELER_H_

#include "opencv2/opencv.hpp"

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/noscope-lite/filters.h"
#include "tensorflow/noscope-lite/noscope_data.h"


namespace noscope {

class NoscopeLabeler {
  friend class FrameData; // access the internals

 public:
  // tensorflow doesn't support unique_ptr
  NoscopeLabeler(tensorflow::Session *SmallCNN_Session,
              tensorflow::Session *LargeCNN_Session,
              noscope::filters::DifferenceFilter diff_filt,
              const std::string& avg_fname,
              const noscope::NoscopeData& data);

  // This currently ignores the upper threshold

  void RunDifferenceFilter(const float lower_thresh, const float upper_thresh,
                           const bool const_ref, const size_t kRef);

  void PopulateCNNFrames();

  void RunSmallCNN(const float lower_thresh, const float upper_thresh);

  void RunLargeCNN(const int class_id, const float large_cnn);

  void DumpConfidences(const std::string& fname,
                       const std::string& model_name,
                       const size_t kSkip,
                       const size_t kStartFrom,
                       const bool kSkipSmallCNN,
                       const float diff_thresh,
                       const float small_cnn_thresh_lower,
                       const float small_cnn_thresh_upper,
                       const float large_cnn_thresh,
                       const std::vector<double>& runtime);

 private:
  constexpr static size_t kNumThreads_ = 32;
  constexpr static size_t kMaxSmallCNNBatch_ = 512;
  constexpr static size_t kMaxLargeCNNBatch_ = 128;
  constexpr static size_t kNbChannels_ = 3;
  constexpr static size_t kDiffDelay_ = 1;

  enum Status {
    kUnprocessed,
    kSkipped,
    kDiffFiltered,
    kDiffUnfiltered,
    kSmallCNNFiltered,
    kSmallCNNUnfiltered,
    kLargeCNNLabeled
  };

  const size_t kNbFrames_;

  const noscope::NoscopeData& all_data_;

  std::vector<Status> frame_status_;
  std::vector<bool> labels_;

  const noscope::filters::DifferenceFilter kDifferenceFilter_;
  std::vector<float> diff_confidence_;

  std::vector<int> small_cnn_frame_ind_;
  std::vector<float> small_cnn_confidence_;

  std::vector<int> large_cnn_frame_ind_;
  std::vector<float> large_cnn_confidence_;

  cv::Mat avg_;

  tensorflow::Session *large_session_;
  tensorflow::Session *small_session_;

  std::vector<tensorflow::Tensor> small_cnn_tensors_;
};

} // namespace noscope

#endif  // TENSORFLOW_VUSE_VUSELABELER_H_
