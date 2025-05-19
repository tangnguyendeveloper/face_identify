#include "models_config.hpp"


ModelsConfig::ModelsConfig() {
    this->pnet_path_ = "";
    this->rnet_path_ = "";
    this->onet_path_ = "";
    this->facenet_path_ = "";
}


ModelsConfig::ModelsConfig(const std::string& pnet_path,
                           const std::string& rnet_path,
                           const std::string& onet_path,
                           const std::string& facenet_path) {

    this->pnet_path_ = pnet_path;
    this->rnet_path_ = rnet_path;
    this->onet_path_ = onet_path;
    this->facenet_path_ = facenet_path;
}


ModelsConfig::ModelsConfig(const ModelsConfig& config) {
    this->pnet_path_ = config.pnet_path_;
    this->rnet_path_ = config.rnet_path_;
    this->onet_path_ = config.onet_path_;
    this->facenet_path_ = config.facenet_path_;

    this->pnet_threshold_ = config.pnet_threshold_;
    this->rnet_threshold_ = config.rnet_threshold_;
    this->onet_threshold_ = config.onet_threshold_;
    this->facenet_input_shape_ = config.facenet_input_shape_;
}


ModelsConfig& ModelsConfig::operator=(const ModelsConfig& config) {
    if (this != &config) {
        this->pnet_path_ = config.pnet_path_;
        this->rnet_path_ = config.rnet_path_;
        this->onet_path_ = config.onet_path_;
        this->facenet_path_ = config.facenet_path_;

        this->pnet_threshold_ = config.pnet_threshold_;
        this->rnet_threshold_ = config.rnet_threshold_;
        this->onet_threshold_ = config.onet_threshold_;
        this->facenet_input_shape_ = config.facenet_input_shape_;
    }
    return *this;
}


void ModelsConfig::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open config file: " + filename);

    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string key, eq, value;
        if (!(iss >> key >> eq >> value)) continue;
        if (eq != "=") continue;

        if (key == "pnet_path") {
            this->pnet_path_ = value;
        } else if (key == "rnet_path") {
            this->rnet_path_ = value;
        } else if (key == "onet_path") {
            this->onet_path_ = value;
        } else if (key == "facenet_path") {
            this->facenet_path_ = value;
        } else if (key == "pnet_threshold") {
            this->pnet_threshold_ = std::stof(value);
        } else if (key == "rnet_threshold") {
            this->rnet_threshold_ = std::stof(value);
        } else if (key == "onet_threshold") {
            this->onet_threshold_ = std::stof(value);
        } else if (key == "facenet_input_shape") {
            this->facenet_input_shape_ = std::stoi(value);
        }
    }
    in.close();
}


void ModelsConfig::save(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot open config file: " + filename);

    out << "pnet_path = " << this->pnet_path_ << "\n";
    out << "rnet_path = " << this->rnet_path_ << "\n";
    out << "onet_path = " << this->onet_path_ << "\n";
    out << "facenet_path = " << this->facenet_path_ << "\n";
    
    out << "pnet_threshold = " << this->pnet_threshold_ << "\n";
    out << "rnet_threshold = " << this->rnet_threshold_ << "\n";
    out << "onet_threshold = " << this->onet_threshold_ << "\n";
    out << "facenet_input_shape = " << this->facenet_input_shape_ << "\n";

    out.close();
}


std::string ModelsConfig::toJson() const {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"pnet_path\": \"" << this->pnet_path_ << "\",\n";
    oss << "  \"pnet_threshold\": " << this->pnet_threshold_ << ",\n";
    oss << "  \"rnet_path\": \"" << this->rnet_path_ << "\",\n";
    oss << "  \"rnet_threshold\": " << this->rnet_threshold_ << ",\n";
    oss << "  \"onet_path\": \"" << this->onet_path_ << "\",\n";
    oss << "  \"onet_threshold\": " << this->onet_threshold_ << ",\n";
    oss << "  \"facenet_path\": \"" << this->facenet_path_ << "\"\n";
    oss << "  \"facenet_input_shape\": " << this->facenet_input_shape_ << "\n";
    oss << "}";
    return oss.str();
}