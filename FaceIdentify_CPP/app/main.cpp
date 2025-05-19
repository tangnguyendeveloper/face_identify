#include "config/camera_config.hpp"
#include "config/models_config.hpp"
#include "camera/internal_camera.hpp"
#include "camera/rtsp_camera.hpp"
#include "draw.hpp"
#include "mtcnn/detector.hpp"
#include "embedding/face_embedding.hpp"
#include "loging.hpp"



bool loadConfig(CameraConfig &camera_config, ModelsConfig &models_config, const std::string &filename, const Logging& logger);



void showFace(const cv::Mat &img, const std::vector<Face> &faces) {
    std::vector<rectPoints> data;
    for (size_t i = 0; i < faces.size(); ++i) {
        std::vector<cv::Point> pts;
        for (int p = 0; p < NUM_PTS; ++p) {
            pts.push_back(
                cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
        }
        auto rect = faces[i].bbox.getRect();
        auto d = std::make_pair(rect, pts);
        data.push_back(d);
    }

    auto resultImg = drawRectsAndPoints(img, data);
    cv::imshow("test-oc", resultImg);
    cv::waitKey(0);
}

int main(int argc, char *argv[]) {
    Logging logger("FaceIdentifyApp");

    if (argc < 2) {
        logger.log(Logging::LogStatus::ERROR, "Usage: " + std::string(argv[0]) + " <config file>");
        return 1;
    }

    // Load camera configuration
    CameraConfig camera_config;
    ModelsConfig models_config;
    if (!loadConfig(camera_config, models_config, argv[1], logger)) return 1;
    logger.log(Logging::LogStatus::INFO, "Camera and Models configuration loaded successfully.");
    logger.log(Logging::LogStatus::INFO, camera_config.toJson());
    logger.log(Logging::LogStatus::INFO, models_config.toJson());

    cv::Mat frame = cv::imread("/home/vht/FaceIdentify/WIN_20250519_06_09_52_Pro.jpg");

    try {
        //  Make MTCNN detector
        MTCNNDetector detector(models_config);
        std::vector<Face> faces = detector.detect(frame, 20, 0.709f);

        logger.log(Logging::LogStatus::INFO, "Detected " + std::to_string(faces.size()) + " faces.");

        // Make Face Embedding
        FaceEmbedding face_embedding(models_config);
        std::vector<std::vector<float>> embeddings = face_embedding.embeddings(frame, faces);

        // Print the embeddings
        for (size_t i = 0; i < embeddings.size(); ++i) {
            std::cout << "Face " << i + 1 << " Embedding: ";
            for (const auto& value : embeddings[i]) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }

        // Draw the detected faces and their embeddings
        showFace(frame, faces);

    } catch(const std::exception& e) {
        logger.log(Logging::LogStatus::ERROR, e.what());
        return 1;
    }

    return 0;
}

bool loadConfig(CameraConfig &camera_config, ModelsConfig &models_config, const std::string &filename, const Logging& logger) {
    try {
        camera_config.load(filename);
        models_config.load(filename);
    } catch (const std::exception &e) {
        logger.log(Logging::LogStatus::ERROR, e.what());
        return false;
    }
    return true;
}
