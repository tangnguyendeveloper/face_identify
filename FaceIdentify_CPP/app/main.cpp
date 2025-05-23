#include "config/camera_config.hpp"
#include "config/models_config.hpp"
#include "camera/internal_camera.hpp"
#include "camera/rtsp_camera.hpp"
#include "mtcnn/detector.hpp"
#include "embedding/face_embedding.hpp"
#include "embedding/embedding_db.hpp"
#include "draw.hpp"
#include "stream.hpp"



// for streaming
static std::queue<cv::Mat> frame_queue;
static std::mutex frame_queue_mutex;
static std::queue<std::pair<cv::Mat, std::vector<std::vector<float>>>> processed_frame_queue;
static std::mutex processed_frame_queue_mutex;




bool parseCommandLineArgs(int argc, char *argv[], std::string &app_name, std::string &config_file,
                          std::string &database_file, std::string &websocket_host, unsigned short &websocket_port);

bool loadConfig(CameraConfig &camera_config, ModelsConfig &models_config, const std::string &filename, const Logging& logger);
bool createCamera(const CameraConfig &camera_config, std::unique_ptr<Camera> &camera, const Logging& logger);
bool loadDatabase(const std::string &filename, EmbeddingDB<People> &embedding_db, const Logging& logger);

void cameraCaptureThread(const CameraConfig& config, cv::Mat &frame, std::atomic<bool> &is_capture,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready);

void processFrameThread(const ModelsConfig& models_config, cv::Mat &frame, std::atomic<bool>& is_process,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready);



static std::atomic<bool>* running_ptr = nullptr;

void handle_signal(int) {
    if (running_ptr) *running_ptr = false;
}

void set_signal_running_ptr(std::atomic<bool>* ptr) {
    running_ptr = ptr;
}


int main(int argc, char *argv[]) {
    std::string config_file;
    std::string database_file = "embedding_db.db";
    std::string app_name = argv[0];

    std::string websocket_host = "localhost";
    unsigned short websocket_port = 9002;

    // Parse command line arguments
    if (!parseCommandLineArgs(argc, argv, app_name, config_file, database_file, websocket_host, websocket_port)) {
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

    // Create or load embedding database
    EmbeddingDB<People> embedding_db(FaceEmbedding::EMBEDDING_SIZE);
    if (!database_file.empty()) {
        if (!loadDatabase(database_file, embedding_db, logger)) {
            logger.log(Logging::LogStatus::INFO, "Using empty database.");
        }
    }


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
        std::atomic<bool> is_add_embedding(false);
        
        std::thread process_thread(
            processFrameThread, std::ref(models_config), std::ref(frame), std::ref(is_process),
            std::ref(running), std::ref(capture_cv), std::ref(logger), std::ref(frame_mutex), std::ref(frame_ready)
        );

        // Start stream server
        std::queue<People> identify_queue;
        std::mutex identify_mutex;
        std::unique_ptr<People> new_person_ptr = nullptr;
        std::mutex new_person_mutex;

        std::thread stream_thread(
            websocketServerThread, std::ref(running), std::ref(is_capture), std::ref(is_process), 
            std::ref(frame_queue_mutex), std::ref(frame_queue), std::ref(processed_frame_queue_mutex), std::ref(processed_frame_queue), 
            std::ref(identify_mutex), std::ref(identify_queue), std::ref(new_person_ptr), std::ref(new_person_mutex),
            std::ref(logger), std::ref(websocket_host), websocket_port
        );

        // Set up signal handling
        set_signal_running_ptr(&running);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        
        while (running) {
            try {
                if (is_capture && is_process && !processed_frame_queue.empty() && embedding_db.size() > 0 && new_person_ptr == nullptr) {
                    std::vector<std::vector<float>> embeddings;
                    {
                        std::lock_guard<std::mutex> lock(processed_frame_queue_mutex);
                        if (!processed_frame_queue.empty()) {
                            embeddings = processed_frame_queue.front().second; 
                            processed_frame_queue.pop();
                        }
                    }
                    
                    if (!embeddings.empty()) {
                        bool valid_embeddings = true;
                        for (const auto& embedding : embeddings) {
                            if (embedding.size() != FaceEmbedding::EMBEDDING_SIZE) {
                                valid_embeddings = false;
                                break;
                            }
                        }
                        
                        if (valid_embeddings) {
                            double min_distance = 2.0;
                            People nearest_person;
                            for (const auto& embedding : embeddings) {
                                auto query_result = embedding_db.query_nearest(embedding);
                                if (query_result.second < min_distance) {
                                    min_distance = query_result.second;
                                    nearest_person = embedding_db.info(query_result.first);
                                }
                            }

                            if (min_distance < 0.4) {
                                std::lock_guard<std::mutex> lock(identify_mutex);
                                identify_queue.push(People(nearest_person));
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                logger.log(Logging::LogStatus::ERROR, "Error in main loop: " + std::string(e.what()));
            }
            
            if (is_capture && is_process && new_person_ptr != nullptr) {
                std::lock_guard<std::mutex> lock(new_person_mutex);
                
                std::vector<std::vector<float>> capture_embeddings;

                // wait for processed frames by sleeping
                std::this_thread::sleep_for(std::chrono::seconds(1));

                unsigned int frame_count = 0; 
                
                while (frame_count < 300){
                    if (processed_frame_queue.empty()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(33));
                        continue;
                    }
                    std::vector<std::vector<float>> embeddings;
                    {
                        std::lock_guard<std::mutex> lock(processed_frame_queue_mutex);
                        if (!processed_frame_queue.empty()) {  // Double check
                            embeddings = processed_frame_queue.front().second;
                            processed_frame_queue.pop();
                        }
                    }
                    if (embeddings.empty() || embeddings.size() > 1) continue;
                    capture_embeddings.push_back(embeddings[0]);
                    frame_count++;
                }
                
                if (!capture_embeddings.empty()) { 
                    // generate new ID
                    int new_id = embedding_db.size() + 1;
                    new_person_ptr->setId(new_id);

                    // convert embeddings to Eigen matrix
                    Eigen::MatrixXf capture_embeddings_matrix = convert_embeddings_to_matrix(capture_embeddings, FaceEmbedding::EMBEDDING_SIZE);
                    // compute mean embeddings
                    Eigen::VectorXf mean_embedding = capture_embeddings_matrix.colwise().mean();
                    // normalize mean embedding 
                    mean_embedding.normalize();
                    // insert into database
                    if (embedding_db.insert(mean_embedding.transpose(), *new_person_ptr)) {
                        logger.log(Logging::LogStatus::INFO, "Inserted new person into the database: " + new_person_ptr->toJsonString());
                    } else {
                        logger.log(Logging::LogStatus::ERROR, "Failed to insert new person into the database.");
                    }
                }
                new_person_ptr.reset();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        capture_thread.join();
        logger.log(Logging::LogStatus::INFO, "Camera capture thread joined successfully.");
        process_thread.join();
        logger.log(Logging::LogStatus::INFO, "Frame processing thread joined successfully.");
        stream_thread.join();
        logger.log(Logging::LogStatus::INFO, "WebSocket stream thread joined successfully.");
        
    }
    catch(const std::exception& e){
        logger.log(Logging::LogStatus::ERROR, e.what());
        running = false;
        return 1;
    }

    // Save database before exit
    if (!database_file.empty() && embedding_db.size() > 0) {
        try {
            if (embedding_db.store(database_file)) {
                logger.log(Logging::LogStatus::INFO, "Database saved successfully.");
            } else {
                logger.log(Logging::LogStatus::WARNING, "Failed to save database.");
            }
        } catch (const std::exception& e) {
            logger.log(Logging::LogStatus::ERROR, "Error saving database: " + std::string(e.what()));
        }
    }

    running = false;
    return 0;
}


bool parseCommandLineArgs(int argc, char *argv[], std::string &app_name, std::string &config_file,
                          std::string &database_file, std::string &websocket_host, unsigned short &websocket_port) {

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
            std::cout << "  --config, -c <config_file>    Path to the configuration file (required)\n";
            std::cout << "  --database, -d <db_file>      Path to the embedding database file (optional)\n";
            std::cout << "  --websocket_host, -w <host>   WebSocket server host (default: localhost)\n";
            std::cout << "  --websocket_port, -p <port>   WebSocket server port (default: 9002)\n";
            std::cout << "  --help, -h                    Show this help message\n";
            return false;

        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "--database" || arg == "-d") && i + 1 < argc) {
            database_file = argv[++i];
        } else if ((arg == "--websocket_host" || arg == "-w") && i + 1 < argc) {
            websocket_host = argv[++i];
        } else if ((arg == "--websocket_port" || arg == "-p") && i + 1 < argc) {
            try {
                websocket_port = static_cast<unsigned short>(std::stoi(argv[++i]));
                if (websocket_port == 0) {
                    std::cerr << "Invalid port number: " << argv[i] << "\n";
                    return false;
                }
            } catch (const std::invalid_argument &e) {
                std::cerr << "Invalid port number: " << argv[i] << "\n";
                return false;
            }
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
    const size_t MAX_QUEUE_SIZE = 10; 

    while (running) {
        if (!is_capture) {
            if (camera) {
                camera->release();
                camera.reset();
                logger.log(Logging::LogStatus::INFO, "Camera released.");
            }

            // free the frame queue
            {
                std::lock_guard<std::mutex> lock(frame_queue_mutex);
                while (!frame_queue.empty()) {
                    frame_queue.pop();
                }
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        // Create camera instance
        if (!camera) {
            if (!createCamera(config, camera, logger)) {
                logger.log(Logging::LogStatus::ERROR, "Failed to open camera.");
                is_capture = false;
                continue;
            }
            logger.log(Logging::LogStatus::INFO, "Camera opened successfully.");
        }

        // Capture frame
        cv::Mat temp_frame;
        if (camera->read(temp_frame)) {
            {
                std::lock_guard<std::mutex> lock(frame_queue_mutex);
                if (frame_queue.size() >= MAX_QUEUE_SIZE) {
                    frame_queue.pop();
                }
                frame_queue.push(temp_frame.clone());
            }
            {
                std::lock_guard<std::mutex> lock(frame_mutex);
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
            is_capture = false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
    }
}


void processFrameThread(const ModelsConfig& models_config, cv::Mat &frame, std::atomic<bool>& is_process,
                        std::atomic<bool> &running, std::condition_variable &capture_cv, const Logging& logger,
                        std::mutex& frame_mutex, std::atomic<bool>& frame_ready) {
                            
    logger.log(Logging::LogStatus::INFO, "Frame processing thread started.");
    
    try {
        MTCNNDetector detector(models_config);
        FaceEmbedding face_embedding(models_config);
        logger.log(Logging::LogStatus::INFO, "Detector and Face Embedding instances created successfully.");

        const size_t MAX_QUEUE_SIZE = 10; 

        while (running) {
            if (!is_process) {
                {
                    std::lock_guard<std::mutex> lock(processed_frame_queue_mutex);
                    while (!processed_frame_queue.empty()) {
                        processed_frame_queue.pop();
                    }
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
            
            cv::Mat temp_frame;
            
            {
                std::unique_lock<std::mutex> lock(frame_mutex);
                capture_cv.wait(lock, [&frame_ready, &running] { return frame_ready.load() || !running.load(); });
                if (!running) break;
                
                if (frame.empty()) {
                    frame_ready = false;
                    continue;
                }
                temp_frame = frame.clone();
                frame_ready = false;
            }

            if (temp_frame.empty()) continue;

            try {
                std::vector<Face> faces = detector.detect(temp_frame, 20, 0.709f);
                if (faces.empty()) continue;
                
                bool valid_faces = true;
                for (const auto& face : faces) {
                    cv::Rect rect = face.bbox.getRect();
                    if (rect.x < 0 || rect.y < 0 || 
                        rect.x + rect.width > temp_frame.cols || 
                        rect.y + rect.height > temp_frame.rows ||
                        rect.width <= 0 || rect.height <= 0) {
                        valid_faces = false;
                        break;
                    }
                }
                
                if (!valid_faces) continue;
                
                std::vector<std::vector<float>> embeddings = face_embedding.embeddings(temp_frame, faces);
                if (embeddings.empty()) continue;

                cv::Mat draw_frame = getDrawFacesImage(temp_frame, faces);
                
                {
                    std::lock_guard<std::mutex> lock(processed_frame_queue_mutex);
                    if (processed_frame_queue.size() >= MAX_QUEUE_SIZE) {
                        processed_frame_queue.pop();
                    }
                    processed_frame_queue.push({draw_frame.clone(), embeddings});
                }
            }
            catch(const std::exception& e){
                logger.log(Logging::LogStatus::WARNING, "Error in face processing: " + std::string(e.what()));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    } catch (const std::exception &e) {
        logger.log(Logging::LogStatus::ERROR, "Fatal error in processFrameThread: " + std::string(e.what()));
        running = false;
    }
    
}



