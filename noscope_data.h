#ifndef TENSORFLOW_VUSE_VUSEDATA_H_
#define TENSORFLOW_VUSE_VUSEDATA_H_

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

namespace noscope {

class NoscopeData {
 public:
  static const cv::Size kLargeCNNResol_;
  static const cv::Size kDiffResol_;
  static const cv::Size kSmallCNNResol_;

  constexpr static size_t kNbChannels_ = 3;
  constexpr static size_t kLargeCNNFrameSize_ = 299 * 299 * kNbChannels_;
  constexpr static size_t kDiffFrameSize_ = 100 * 100 * kNbChannels_;
  constexpr static size_t kSmallCNNFrameSize_ = 50 * 50 * kNbChannels_;

  const size_t kNbFrames_;
  const size_t kSkip_;

  std::vector<uint8_t> large_cnn_data_;
  std::vector<uint8_t> diff_data_;
  std::vector<float> small_cnn_data_;

  NoscopeData(const std::string& fname, const size_t kSkip, const size_t kNbFrames, const size_t kStart);
};

} // namespace noscope

#endif
