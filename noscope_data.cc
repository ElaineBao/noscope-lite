#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include "tensorflow/noscope-lite/noscope_data.h"

namespace noscope {

// FIXME: should really fix this
const cv::Size NoscopeData::kLargeCNNResol_(299, 299);
const cv::Size NoscopeData::kDiffResol_(100, 100);
const cv::Size NoscopeData::kSmallCNNResol_(50, 50);

NoscopeData::NoscopeData(const std::string& fname,
                   const size_t kSkip, const size_t kNbFrames, const size_t kStart) :
    kNbFrames_(kNbFrames / kSkip),
    kSkip_(kSkip),
    large_cnn_data_(kLargeCNNFrameSize_ * kNbFrames_),
    diff_data_(kDiffFrameSize_ * kNbFrames_),
    small_cnn_data_(kSmallCNNFrameSize_ * kNbFrames_){
  cv::VideoCapture cap(fname);
  if (kStart > 0)
    cap.set(cv::CAP_PROP_POS_FRAMES, kStart - 1);

  cv::Mat frame;
  cv::Mat large_cnn_frame(NoscopeData::kLargeCNNResol_, CV_8UC3);
  cv::Mat diff_frame(NoscopeData::kDiffResol_, CV_8UC3);
  cv::Mat dist_frame(NoscopeData::kSmallCNNResol_, CV_8UC3);
  cv::Mat dist_frame_f(NoscopeData::kSmallCNNResol_, CV_32FC3);
  for (size_t i = 0; i < kNbFrames; i++) {
    cap >> frame;
    if (i % kSkip_ == 0) {
      const size_t ind = i / kSkip_;
      cv::resize(frame, large_cnn_frame, NoscopeData::kLargeCNNResol_, 0, 0, cv::INTER_NEAREST);
      cv::resize(frame, diff_frame, NoscopeData::kDiffResol_, 0, 0, cv::INTER_NEAREST);
      cv::resize(frame, dist_frame, NoscopeData::kSmallCNNResol_, 0, 0, cv::INTER_NEAREST);
      dist_frame.convertTo(dist_frame_f, CV_32FC3);

      if (!large_cnn_frame.isContinuous()) {
        throw std::runtime_error("large_cnn frame is not continuous");
      }
      if (!diff_frame.isContinuous()) {
        throw std::runtime_error("diff frame is not continuous");
      }
      if (!dist_frame.isContinuous()) {
        throw std::runtime_error("dist frame is not conintuous");
      }
      if (!dist_frame_f.isContinuous()) {
        throw std::runtime_error("dist frame f is not continuous");
      }

      memcpy(&large_cnn_data_[ind * kLargeCNNFrameSize_], large_cnn_frame.data, kLargeCNNFrameSize_);
      memcpy(&diff_data_[ind * kDiffFrameSize_], diff_frame.data, kDiffFrameSize_);
      memcpy(&small_cnn_data_[ind * kSmallCNNFrameSize_], dist_frame_f.data, kSmallCNNFrameSize_ * sizeof(float));
    }
  }
}

} // namespace noscope
