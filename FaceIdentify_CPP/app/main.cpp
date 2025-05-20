#include "config/camera_config.hpp"
#include "config/models_config.hpp"
#include "camera/internal_camera.hpp"
#include "camera/rtsp_camera.hpp"
#include "draw.hpp"
#include "mtcnn/detector.hpp"
#include "embedding/face_embedding.hpp"
#include "embedding/embedding_db.hpp"
#include "People.hpp"
#include "stream.hpp"




static std::queue<cv::Mat> frame_queue;
static std::queue<std::pair<cv::Mat, std::vector<std::vector<float>>>> processed_frame_queue;
static std::mutex processed_frame_mutex;


bool parseCommandLineArgs(int argc, char *argv[], std::string &app_name, std::string &config_file, std::string &database_file);
bool loadConfig(CameraConfig &camera_config, ModelsConfig &models_config, const std::string &filename, const Logging& logger);
bool createCamera(const CameraConfig &camera_config, std::unique_ptr<Camera> &camera, const Logging& logger);
bool loadDatabase(const std::string &filename, EmbeddingDB<People> &embedding_db, const Logging& logger);

void cameraCaptureThread(const CameraConfig& config, cv::Mat &frame, std::atomic<bool> &is_capture,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready);

void processFrameThread(const ModelsConfig& models_config, cv::Mat &frame, std::atomic<bool>& is_process, const std::string &filename,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready);





int main(int argc, char *argv[]) {
    std::string config_file;
    std::string database_file = "embedding_db.db";
    std::string app_name = argv[0];

    // Parse command line arguments
    if (!parseCommandLineArgs(argc, argv, app_name, config_file, database_file)) {
        return 1;
    }

    // Create a logger instance
    Logging logger(app_name);

    // Load camera configuration
    CameraConfig camera_config;
    ModelsConfig models_config;
    if (!loadConfig(camera_config, models_config, config_file, logger)) return 1;
    logger.log(Logging::LogStatus::INFO, "Camera and Models configuration loaded successfully.");
    logger.log(Logging::LogStatus::INFO, camera_config.toJson());
    logger.log(Logging::LogStatus::INFO, models_config.toJson());


    std::atomic<bool> running(true);


    try{
        // Start camera capture thread
        cv::Mat frame;
        std::mutex frame_mutex;
        std::atomic<bool> is_capture(false);
        std::atomic<bool> frame_ready(false);
        std::condition_variable capture_cv;

        std::thread capture_thread(
            cameraCaptureThread, std::ref(camera_config), std::ref(frame), std::ref(is_capture),
            std::ref(running), std::ref(capture_cv), std::ref(logger), std::ref(frame_mutex), std::ref(frame_ready)
        );
        
         // Start process thread
        std::atomic<bool> is_process(false);
        
        std::thread process_thread(
            processFrameThread, std::ref(models_config), std::ref(frame), std::ref(is_process), std::ref(database_file),
            std::ref(running), std::ref(capture_cv), std::ref(logger), std::ref(frame_mutex), std::ref(frame_ready)
        );

        // Start stream server
        std::thread stream_thread(
            streamWebsocketThread, std::ref(running), std::ref(frame_mutex), std::ref(frame_queue),
            std::ref(processed_frame_mutex), std::ref(processed_frame_queue), std::ref(logger)
        );

        capture_thread.join();
        running = false;
        logger.log(Logging::LogStatus::INFO, "Camera capture thread joined successfully.");
        process_thread.join();
        running = false;
        logger.log(Logging::LogStatus::INFO, "Frame processing thread joined successfully.");
        stream_thread.join();
        logger.log(Logging::LogStatus::INFO, "WebSocket stream thread joined successfully.");
        
    }
    catch(const std::exception& e){
        logger.log(Logging::LogStatus::ERROR, e.what());
        running = false;
        return 1;
    }


    

    /*
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
    */
    running = false;
    return 0;
}


bool parseCommandLineArgs(int argc, char *argv[], std::string &app_name, std::string &config_file, std::string &database_file) {
    // Remove the path from the app name
    size_t pos = app_name.find_last_of("/\\");
    if (pos != std::string::npos) {
        app_name = app_name.substr(pos + 1);
    }
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << app_name << " - Face Detection and Embedding Application\n\n";
            std::cout << "Usage: " << app_name << " --config <config_file> [--database <database_file>]\n";
            std::cout << "  --config <config_file>    Path to the configuration file (required)\n";
            std::cout << "  --database <db_file>      Path to the embedding database file (optional)\n";
            std::cout << "  --help                    Show this help message\n";
            return false;
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--database" && i + 1 < argc) {
            database_file = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information.\n";
            return false;
        }
    }
    if (config_file.empty()) {
        std::cerr << "Error: --config <config_file> is required.\n";
        return false;
    }
    return true;
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


bool createCamera(const CameraConfig &camera_config, std::unique_ptr<Camera> &camera, const Logging& logger) {
    try {
        if (camera_config.source() == CameraConfig::SourceType::INTERNAL) {
            camera = std::make_unique<InternalCamera>(camera_config);
        } else if (camera_config.source() == CameraConfig::SourceType::RTSP) {
            camera = std::make_unique<RTSPCamera>(camera_config);
        } else {
            logger.log(Logging::LogStatus::WARNING, "Unknown camera source type.");
            return false;
        }
    } catch (const std::exception &e) {
        logger.log(Logging::LogStatus::ERROR, e.what());
        return false;
    }
    return camera != nullptr ? camera->isOpened() : false;
}


bool loadDatabase(const std::string &filename, EmbeddingDB<People> &embedding_db, const Logging& logger) {
    try {
        if (!embedding_db.load(filename)) {
            logger.log(Logging::LogStatus::WARNING, "Failed to load the database " + filename);
            return false;
        }
    } catch (const std::exception &e) {
        logger.log(Logging::LogStatus::ERROR, e.what());
        return false;
    }
    return true;
}


void cameraCaptureThread(const CameraConfig& config, cv::Mat &frame, std::atomic<bool> &is_capture,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready) {    

    logger.log(Logging::LogStatus::INFO, "Camera capture thread started.");
    std::unique_ptr<Camera> camera = nullptr;
    
    while (running) {
        if (!is_capture) {
            if (camera) {
                camera->release();
                camera.reset();
                logger.log(Logging::LogStatus::INFO, "Camera released.");
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        // Create camera instance
        if (!camera) {
            if (!createCamera(config, camera, logger)) {
                logger.log(Logging::LogStatus::ERROR, "Failed to open camera.");
                return;
            }
            logger.log(Logging::LogStatus::INFO, "Camera opened successfully.");
        }

        // Capture frame
        cv::Mat temp_frame;
        if (camera->read(temp_frame)) {
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame_queue.push(temp_frame.clone());
                frame = temp_frame.clone();
                frame_ready = true;
            }
            capture_cv.notify_one();

        } else {
            logger.log(Logging::LogStatus::WARNING, "Failed to read frame from camera.");
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
                frame_ready = false;
            }
        }
    }
}


void processFrameThread(const ModelsConfig& models_config, cv::Mat &frame, std::atomic<bool>& is_process, const std::string &filename,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready) {
                            
    logger.log(Logging::LogStatus::INFO, "Frame processing thread started.");
    
    try {
        // Create detector and face embedding instances
        MTCNNDetector detector(models_config);
        FaceEmbedding face_embedding(models_config);
        logger.log(Logging::LogStatus::INFO, "Detector and Face Embedding instances created successfully.");

        // Create or load embedding database
        EmbeddingDB<People> embedding_db(FaceEmbedding::EMBEDDING_SIZE);
        if (!filename.empty()) {
            if (!loadDatabase(filename, embedding_db, logger)) {
                logger.log(Logging::LogStatus::INFO, "Using empty database.");
            }
        }

        while (running) {
            if (!is_process) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            
            cv::Mat temp_frame;
            
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                // Wait for a frame to be ready
                capture_cv.wait(lock, [&frame_ready] { return frame_ready.load(); });
                temp_frame = frame.clone();
                frame_ready = false;
            }

            // Detect faces
            try{
                std::vector<Face> faces = detector.detect(temp_frame, 20, 0.709f);
                if (faces.empty()) continue;
                std::vector<std::vector<float>> embeddings = face_embedding.embeddings(temp_frame, faces);
                if (embeddings.empty()) continue;

                cv::Mat draw_frame = getDrawFacesImage(temp_frame, faces);
                // put the frame into the queue
                {
                    std::lock_guard<std::mutex> lock(processed_frame_mutex);
                    processed_frame_queue.push({draw_frame.clone(), embeddings});
                }
            }
            catch(const std::exception& e){
                logger.log(Logging::LogStatus::WARNING, e.what());
            }

        }
    } catch (const std::exception &e) {
        logger.log(Logging::LogStatus::ERROR, e.what());
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            running = false;
        }
    }
    
}



