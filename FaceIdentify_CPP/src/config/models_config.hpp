#ifndef MTCNN_FACENET_MODELS_CONFIG_HPP
#define MTCNN_FACENET_MODELS_CONFIG_HPP

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

class ModelsConfig {
public:
    ModelsConfig();
    ModelsConfig(const std::string& pnet_path,
                 const std::string& rnet_path,
                 const std::string& onet_path,
                 const std::string& facenet_path);

    ModelsConfig(const ModelsConfig& config);
    ModelsConfig& operator=(const ModelsConfig& config);

    inline const std::string& pnet_path() const { return pnet_path_; }
    inline float pnet_threshold() const { return pnet_threshold_; }
    inline const std::string& rnet_path() const { return rnet_path_; }
    inline float rnet_threshold() const { return rnet_threshold_; }
    inline const std::string& onet_path() const { return onet_path_; }
    inline float onet_threshold() const { return onet_threshold_; }
    inline const std::string& facenet_path() const { return facenet_path_; }
    inline int facenet_input_shape() const { return facenet_input_shape_; }

    inline void set_pnet_path(const std::string& path) { pnet_path_ = path; }
    inline void set_pnet_threshold(float threshold) { pnet_threshold_ = threshold; }
    inline void set_rnet_path(const std::string& path) { rnet_path_ = path; }
    inline void set_rnet_threshold(float threshold) { rnet_threshold_ = threshold; }
    inline void set_onet_path(const std::string& path) { onet_path_ = path; }
    inline void set_onet_threshold(float threshold) { onet_threshold_ = threshold; }
    inline void set_facenet_path(const std::string& path) { facenet_path_ = path; }
    inline void set_facenet_input_shape(int shape) { facenet_input_shape_ = shape; }

    // Read config from file
    void load(const std::string& filename);

    // Save config to file
    void save(const std::string& filename) const;

    // To json string representation
    std::string toJson() const;

private:
    std::string pnet_path_;
    float pnet_threshold_ = 0.6f;
    std::string rnet_path_;
    float rnet_threshold_ = 0.7f;
    std::string onet_path_;
    float onet_threshold_ = 0.7f;
    std::string facenet_path_;
    int facenet_input_shape_ = 160;
};

#endif // MTCNN_FACENET_MODELS_CONFIG_HPP