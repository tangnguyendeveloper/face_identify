#ifndef CAMERA_CONFIG_HPP
#define CAMERA_CONFIG_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>


class CameraConfig {
public:
    enum class SourceType { INTERNAL, RTSP };

    CameraConfig();
    CameraConfig(SourceType source, const std::string& rtsp_url, int frame_width,
                 int frame_height, int rtsp_timeout_ms,int capture_buffer_size);
                 
    CameraConfig(const CameraConfig& config);
    CameraConfig& operator=(const CameraConfig& config);

    
    inline SourceType source() const { return source_; }
    inline const std::string& rtsp_url() const { return rtsp_url_; }
    inline int frame_width() const { return frame_width_; }
    inline int frame_height() const { return frame_height_; }
    inline int rtsp_timeout_ms() const { return rtsp_timeout_ms_; }
    inline int capture_buffer_size() const { return capture_buffer_size_; }

    
    inline void set_source(SourceType s) { source_ = s; }
    inline void set_rtsp_url(const std::string& url) { rtsp_url_ = url; }
    inline void set_frame_width(int w) { frame_width_ = w; }
    inline void set_frame_height(int h) { frame_height_ = h; }
    inline void set_rtsp_timeout_ms(int t) { rtsp_timeout_ms_ = t; }
    inline void set_capture_buffer_size(int s) { capture_buffer_size_ = s; }

    // Read config from file
    void load(const std::string& filename);

    // Save config to file
    void save(const std::string& filename) const;

    // To json string representation
    std::string toJson() const;

private:
    SourceType source_;
    std::string rtsp_url_;
    int frame_width_;
    int frame_height_;
    int rtsp_timeout_ms_;
    int capture_buffer_size_;
};


#endif // CAMERA_CONFIG_HPP







