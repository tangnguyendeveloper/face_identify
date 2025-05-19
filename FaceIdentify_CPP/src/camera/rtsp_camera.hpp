#ifndef RTSP_CAMERA_HPP
#define RTSP_CAMERA_HPP

#include "camera.hpp"

class RTSPCamera : public Camera {
public:
    explicit RTSPCamera(const CameraConfig& config);
    ~RTSPCamera() override;

    bool open() override;
    void release() override;
    bool read(cv::Mat& frame) override;
    bool isOpened() const override;

    bool setUrl(const std::string& url);
    std::string getUrl() const { return config_.rtsp_url(); }

    void setConfig(const CameraConfig& config) { config_ = config; }
    CameraConfig getConfig() const { return config_; }

    // to json string presentation
    std::string toJson() const;

private:
    cv::VideoCapture cap_;
    CameraConfig config_;
};

#endif // RTSP_CAMERA_HPP