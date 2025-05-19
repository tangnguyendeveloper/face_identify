#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include "config/camera_config.hpp"

class Camera {
public:
    virtual ~Camera() = default;
    virtual bool open() = 0;
    virtual void release() = 0;
    virtual bool read(cv::Mat& frame) = 0;
    virtual bool isOpened() const = 0;
};

#endif // CAMERA_HPP