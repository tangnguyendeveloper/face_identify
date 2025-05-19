#include "internal_camera.hpp"


InternalCamera::InternalCamera(const CameraConfig& config) {
    if (config.source() == CameraConfig::SourceType::INTERNAL) {
        this->config_ = config;
        this->scanAndChooseCameraIndex();
    } else {
        throw std::invalid_argument("Invalid source type for InternalCamera");
    }
}


void InternalCamera::scanAndChooseCameraIndex() {
    
    for (int idx = 0; idx < MAX_TESTED_CAMERAS; ++idx) {
        this->deviceIndex_ = idx;
        if (this->open()) {
            return;
        }
        this->release();
    }
    throw std::runtime_error("No available internal camera found");
}


InternalCamera::~InternalCamera() {
    this->release();
}


bool InternalCamera::open() {
    if (this->cap_.isOpened()) {
        return true;
    }
    
    if (this->cap_.open(this->deviceIndex_)) {
        this->cap_.set(cv::CAP_PROP_FRAME_WIDTH, this->config_.frame_width());
        this->cap_.set(cv::CAP_PROP_FRAME_HEIGHT, this->config_.frame_height());
    }
    return this->isOpened();
}


void InternalCamera::release() {
    if (this->isOpened()) {
        this->cap_.release();
    }
}


bool InternalCamera::read(cv::Mat& frame) {
    if (!this->isOpened()) {
        return false;
    }
    return this->cap_.read(frame);
}


bool InternalCamera::isOpened() const {
    return this->cap_.isOpened();
}


bool InternalCamera::switchCamera(int index) {
    if (index < 0 || index >= MAX_TESTED_CAMERAS) {
        throw std::out_of_range("Camera index out of range");
    }

    if (this->deviceIndex_ == index) {
        return true;
    }

    int currentIndex = this->deviceIndex_;
    this->deviceIndex_ = index;
    this->release();

    if (this->open()) {
        return true;
    }
    this->deviceIndex_ = currentIndex;
    this->open();
    return false;
}


std::string InternalCamera::toJson() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"config\": " << this->config_.toJson() << ",";
    oss << "\"isOpened\": " << (this->isOpened() ? "true" : "false") << ",";
    oss << "\"deviceIndex\": " << this->deviceIndex_;
    oss << "}";
    return oss.str();
}