#ifndef INTERNAL_CAMERA_HPP
#define INTERNAL_CAMERA_HPP

#include "camera.hpp"

#ifndef MAX_TESTED_CAMERAS
#define MAX_TESTED_CAMERAS 10
#endif

class InternalCamera : public Camera {
public:
    explicit InternalCamera(const CameraConfig& config);
    ~InternalCamera() override;

    bool open() override;
    void release() override;
    bool read(cv::Mat& frame) override;
    bool isOpened() const override;

    bool switchCamera(int index);

    int getDeviceIndex() const { return deviceIndex_; }
    void setDeviceIndex(int index) { deviceIndex_ = index; }

    void setConfig(const CameraConfig& config) { config_ = config; }
    CameraConfig getConfig() const { return config_; }

    // to json string presentation
    std::string toJson() const;

private:
    int deviceIndex_;
    cv::VideoCapture cap_;
    CameraConfig config_;

    void scanAndChooseCameraIndex();
};


#endif // INTERNAL_CAMERA_HPP