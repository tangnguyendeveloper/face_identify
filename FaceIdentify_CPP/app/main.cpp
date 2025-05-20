#include "config/camera_config.hpp"
#include "config/models_config.hpp"
#include "camera/internal_camera.hpp"
#include "camera/rtsp_camera.hpp"
#include "draw.hpp"
#include "mtcnn/detector.hpp"
#include "embedding/face_embedding.hpp"
#include "embedding/embedding_db.hpp"
#include "loging.hpp"
#include "People.hpp"




bool loadConfig(CameraConfig &camera_config, ModelsConfig &models_config, const std::string &filename, const Logging& logger);




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

    cv::Mat frame1 = cv::imread("/home/vht/FaceIdentify/WIN_20250519_06_09_52_Pro.jpg");
    cv::Mat frame2 = cv::imread("/home/vht/FaceIdentify/WIN_20250519_06_10_00_Pro.jpg");

    // create embedding database
    EmbeddingDB<People> embedding_db(FaceEmbedding::EMBEDDING_SIZE);

    try {
        //  Make MTCNN detector
        MTCNNDetector detector(models_config);
        std::vector<Face> faces1 = detector.detect(frame1, 20, 0.709f);
        std::vector<Face> faces2 = detector.detect(frame2, 20, 0.709f);
        logger.log(Logging::LogStatus::INFO, "Detected " + std::to_string(faces1.size()) + " faces in frame 1.");
        logger.log(Logging::LogStatus::INFO, "Detected " + std::to_string(faces2.size()) + " faces in frame 2.");

        // Make Face Embedding
        FaceEmbedding face_embedding(models_config);
        std::vector<std::vector<float>> embeddings1 = face_embedding.embeddings(frame1, faces1);
        std::vector<std::vector<float>> embeddings2 = face_embedding.embeddings(frame2, faces2);
        
        People person1(1, "Person 1", 25);
        People person2(2, "Person 2", 30);
        embedding_db.insert(convert_embeddings_to_matrix(embeddings1, FaceEmbedding::EMBEDDING_SIZE), person1);
        embedding_db.insert(convert_embeddings_to_matrix(embeddings2, FaceEmbedding::EMBEDDING_SIZE), person2);
        logger.log(Logging::LogStatus::INFO, "Inserted embeddings into the database.");

        // Query nearest embedding
        auto query_result = embedding_db.query_nearest(embeddings1[0]);
        if (query_result.first != -1) {
            logger.log(Logging::LogStatus::INFO, "Nearest embedding found with ID: " + std::to_string(query_result.first) + ", Distance: " + std::to_string(query_result.second));
            auto person = embedding_db.infos()[query_result.first];
            logger.log(Logging::LogStatus::INFO, "Person Info: " + person.toJsonString());
            double distance = compute_cosine_distance(embeddings1[0], embeddings2[0]);
            logger.log(Logging::LogStatus::INFO, "Cosine distance between face 1 and face 2: " + std::to_string(distance));
        } else {
            logger.log(Logging::LogStatus::INFO, "No nearest embedding found.");
        }

        // Save the database to a file
        if (embedding_db.store("embedding_db.db")) {
            logger.log(Logging::LogStatus::INFO, "Database saved successfully.");
        } else {
            logger.log(Logging::LogStatus::ERROR, "Failed to save the database.");
        }

        // clear the database
        while (embedding_db.size() > 0) {
            auto person = embedding_db.info(0);
            embedding_db.erase(person);
            logger.log(Logging::LogStatus::INFO, "Erased person with ID: " + person.toJsonString());
        }
        embedding_db.clear();
        logger.log(Logging::LogStatus::INFO, "Database cleared.");

        // Load the database from a file
        EmbeddingDB<People> loaded_db(FaceEmbedding::EMBEDDING_SIZE);
        if (loaded_db.load("embedding_db.db")) {
            logger.log(Logging::LogStatus::INFO, "Database loaded successfully.");
        } else {
            logger.log(Logging::LogStatus::ERROR, "Failed to load the database.");
        }
        logger.log(Logging::LogStatus::INFO, "Loaded database size: " + std::to_string(loaded_db.size()));
        for (size_t i = 0; i < loaded_db.size(); ++i) {
            auto person = loaded_db.info(i);
            logger.log(Logging::LogStatus::INFO, "Person Info: " + person.toJsonString());
        }
        
        // Draw the detected faces and their embeddings
        cv::Mat draw_frame1 = getDrawFacesImage(frame1, faces1);
        cv::Mat draw_frame2 = getDrawFacesImage(frame2, faces2);
        cv::imshow("Detected Faces 1", draw_frame1);
        cv::imshow("Detected Faces 2", draw_frame2);
        cv::waitKey(0);
        

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
