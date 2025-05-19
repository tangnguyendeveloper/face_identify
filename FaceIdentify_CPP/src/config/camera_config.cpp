#include "camera_config.hpp"


CameraConfig::CameraConfig() {
    this->source_ = SourceType::INTERNAL;
    this->rtsp_url_ = "";
    this->frame_width_ = 640;
    this->frame_height_ = 480;
    this->rtsp_timeout_ms_ = 5000;
    this->capture_buffer_size_ = 1;
}


CameraConfig::CameraConfig(SourceType source, const std::string& rtsp_url, int frame_width,
                            int frame_height, int rtsp_timeout_ms, int capture_buffer_size) {
    this->source_ = source;
    this->rtsp_url_ = rtsp_url;
    this->frame_width_ = frame_width;
    this->frame_height_ = frame_height;
    this->rtsp_timeout_ms_ = rtsp_timeout_ms;
    this->capture_buffer_size_ = capture_buffer_size;
}


CameraConfig::CameraConfig(const CameraConfig& config) {
    this->source_ = config.source_;
    this->rtsp_url_ = config.rtsp_url_;
    this->frame_width_ = config.frame_width_;
    this->frame_height_ = config.frame_height_;
    this->rtsp_timeout_ms_ = config.rtsp_timeout_ms_;
    this->capture_buffer_size_ = config.capture_buffer_size_;
}


CameraConfig& CameraConfig::operator=(const CameraConfig& config) {
    if (this != &config) {
        this->source_ = config.source_;
        this->rtsp_url_ = config.rtsp_url_;
        this->frame_width_ = config.frame_width_;
        this->frame_height_ = config.frame_height_;
        this->rtsp_timeout_ms_ = config.rtsp_timeout_ms_;
        this->capture_buffer_size_ = config.capture_buffer_size_;
    }
    return *this;
}


void CameraConfig::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open config file: " + filename);

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string key, eq, value;
        if (!(iss >> key >> eq >> value)) continue;
        if (eq != "=") continue;

        if (key == "source") {
            if (value == "INTERNAL") this->source_ = SourceType::INTERNAL;
            else if (value == "RTSP") this->source_ = SourceType::RTSP;
        } else if (key == "rtsp_url") {
            this->rtsp_url_ = value;
        } else if (key == "frame_width") {
            this->frame_width_ = std::stoi(value);
        } else if (key == "frame_height") {
            this->frame_height_ = std::stoi(value);
        } else if (key == "rtsp_timeout_ms") {
            this->rtsp_timeout_ms_ = std::stoi(value);
        } else if (key == "capture_buffer_size") {
            this->capture_buffer_size_ = std::stoi(value);
        }
    }

    if (this->source_ == SourceType::RTSP && this->rtsp_url_.empty()) {
        throw std::runtime_error("RTSP source is selected but no URL provided.");
    }
    in.close();
}


void CameraConfig::save(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open config file: " + filename);

    out << "source = " << (this->source_ == SourceType::INTERNAL ? "INTERNAL" : "RTSP") << "\n";
    out << "rtsp_url = " << this->rtsp_url_ << "\n";
    out << "frame_width = " << this->frame_width_ << "\n";
    out << "frame_height = " << this->frame_height_ << "\n";
    out << "rtsp_timeout_ms = " << this->rtsp_timeout_ms_ << "\n";
    out << "capture_buffer_size = " << this->capture_buffer_size_ << "\n";
    out.close();
}


std::string CameraConfig::toJson() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"source\": \"" << (this->source_ == SourceType::INTERNAL ? "INTERNAL" : "RTSP") << "\",\n";
    oss << "  \"rtsp_url\": \"" << this->rtsp_url_ << "\",\n";
    oss << "  \"frame_width\": " << this->frame_width_ << ",\n";
    oss << "  \"frame_height\": " << this->frame_height_ << ",\n";
    oss << "  \"rtsp_timeout_ms\": " << this->rtsp_timeout_ms_ << ",\n";
    oss << "  \"capture_buffer_size\": " << this->capture_buffer_size_ << "\n";
    oss << "}";
    return oss.str();
}