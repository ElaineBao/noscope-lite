#ifndef TENSORFLOW_VUSE_VUSEDATA_H_
#define TENSORFLOW_VUSE_VUSEDATA_H_

#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

namespace noscope-lite {

class NoscopeData {
 public:
  static const cv::Size kYOLOResol_;
  static const cv::Size kDiffResol_;
  static const cv::Size kDistResol_;

  constexpr static size_t kNbChannels_ = 3;
  constexpr static size_t kYOLOFrameSize_ = 299 * 299 * kNbChannels_;
  constexpr static size_t kDiffFrameSize_ = 100 * 100 * kNbChannels_;
  constexpr static size_t kDistFrameSize_ = 50 * 50 * kNbChannels_;

  const size_t kNbFrames_;
  const size_t kSkip_;

  std::vector<float> yolo_data_;
  std::vector<uint8_t> diff_data_;
  std::vector<float> dist_data_;

  NoscopeData(const std::string& fname, const size_t kSkip, const size_t kNbFrames, const size_t kStart);

  NoscopeData(const std::string& fname);

  void DumpAll(const std::string& fname);
};

} // namespace noscope-lite

#endif
