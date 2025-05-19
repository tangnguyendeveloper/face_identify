#include "rtsp_camera.hpp"


RTSPCamera::RTSPCamera(const CameraConfig& config) {
    if (config.source() == CameraConfig::SourceType::RTSP) {
        this->config_ = config;
        if (!this->open()) {
            throw std::runtime_error("Failed to open RTSP camera");
        }
    } else {
        throw std::invalid_argument("Invalid source type for RTSPCamera");
    }
}


RTSPCamera::~RTSPCamera() {
    this->release();
}


bool RTSPCamera::open() {
    if (this->cap_.isOpened()) {
        return true;
    }

    this->cap_.set(cv::CAP_PROP_OPEN_TIMEOUT_MSEC, this->config_.rtsp_timeout_ms());
    this->cap_.set(cv::CAP_PROP_BUFFERSIZE, this->config_.capture_buffer_size());

    if (this->cap_.open(this->config_.rtsp_url())) {
        this->cap_.set(cv::CAP_PROP_FRAME_WIDTH, this->config_.frame_width());
        this->cap_.set(cv::CAP_PROP_FRAME_HEIGHT, this->config_.frame_height());
    }
    return this->isOpened();
}


void RTSPCamera::release() {
    if (this->isOpened()) {
        this->cap_.release();
    }
}


bool RTSPCamera::read(cv::Mat& frame) {
    if (!this->isOpened()) {
        return false;
    }
    return this->cap_.read(frame);
}


bool RTSPCamera::isOpened() const {
    return this->cap_.isOpened();
}


bool RTSPCamera::setUrl(const std::string& url) {
    if (this->config_.rtsp_url() == url) {
        return true;
    }

    std::string currentURL = this->config_.rtsp_url();
    this->release();
    this->config_.set_rtsp_url(url);

    if (this->open()) {
        return true;
    }

    this->config_.set_rtsp_url(currentURL);
    this->open();

    return false;
}


std::string RTSPCamera::toJson() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"config\": " << this->config_.toJson() << ",";
    oss << "\"isOpened\": " << (this->isOpened() ? "true" : "false") << ",";
    oss << "\"url\": \"" << this->config_.rtsp_url() << "\"";
    oss << "}";
    return oss.str();
}