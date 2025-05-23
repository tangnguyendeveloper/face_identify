#ifndef IDENTIFY_STREAM_HPP
#define IDENTIFY_STREAM_HPP


#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/json.hpp>
#include <boost/beast/websocket.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>


#include "loging.hpp"
#include "People.hpp"


namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;
namespace json = boost::json;


// Helper: Encode JPEG buffer to base64
std::string base64_encode(const std::vector<uchar>& buf) {
    static const char* base64_chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string ret;
    int i = 0;
    unsigned char char_array_3[3], char_array_4[4];
    size_t pos = 0;
    size_t len = buf.size();
    while (len--) {
        char_array_3[i++] = buf[pos++];
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;
            for(i = 0; i < 4; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }
    if (i) {
        for(int j = i; j < 3; j++)
            char_array_3[j] = '\0';
        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;
        for (int j = 0; j < i + 1; j++)
            ret += base64_chars[char_array_4[j]];
        while((i++ < 3))
            ret += '=';
    }
    return ret;
}


/*
    * WebSocket server thread function
    * This function handles incoming WebSocket connections and messages.
    * It sends raw frames, processed frames, and identification results to the client.
    *
    * @param running: Atomic boolean to control the running state of the server.
    * @param is_capture: Atomic boolean to control the capture state.
    * @param is_process: Atomic boolean to control the processing state.
    * @param frame_queue_mutex: Mutex for synchronizing access to the frame queue.
    * @param frame_queue: Queue for storing raw frames.
    * @param processed_frame_queue_mutex: Mutex for synchronizing access to the processed frame queue.
    * @param processed_frame_queue: Queue for storing processed frames and their embeddings.
    * @param identify_mutex: Mutex for synchronizing access to the identification queue.
    * @param identify_queue: Queue for storing identification results.
    * @param new_person_ptr: Pointer to a new person object for identification.
    * @param new_person_mutex: Mutex for synchronizing access to the new person pointer.
    * @param logger: Logger instance for logging messages.
    * @param host: Host address for the WebSocket server 
    * @param port: Port number for the WebSocket server
*/
void websocketServerThread(
    std::atomic<bool>& running,
    std::atomic<bool>& is_capture,
    std::atomic<bool>& is_process,
    std::mutex& frame_queue_mutex,
    std::queue<cv::Mat>& frame_queue,
    std::mutex& processed_frame_queue_mutex,
    std::queue<std::pair<cv::Mat, std::vector<std::vector<float>>>>& processed_frame_queue,
    std::mutex& identify_mutex,
    std::queue<People>& identify_queue,
    std::unique_ptr<People>& new_person_ptr,
    std::mutex& new_person_mutex,
    const Logging& logger,
    const std::string& host = "127.0.0.1",
    unsigned short port = 9002
) {
    try {
        std::mutex capture, process;
        net::io_context ioc{1};
        
        std::atomic<bool> client_connected{false};
        std::mutex client_mutex;
        std::condition_variable client_cv;
        
        tcp::resolver resolver(ioc);
        auto resolved = resolver.resolve(host, std::to_string(port));
        tcp::acceptor acceptor{ioc, *resolved.begin()};
        logger.log(Logging::LogStatus::INFO, "WebSocket server started at ws://" + host + ":" + std::to_string(port));

        while (running) {
            
            if (client_connected) {
                
                std::unique_lock<std::mutex> lock(client_mutex);
                client_cv.wait_for(lock, std::chrono::seconds(1), [&client_connected] { 
                    return !client_connected; 
                });
                continue;
            }

            tcp::socket socket{ioc};
            boost::system::error_code ec;

            acceptor.accept(socket, ec);
            if (ec) {
                logger.log(Logging::LogStatus::ERROR, "Accept error: " + ec.message());
                continue;
            }
            logger.log(Logging::LogStatus::INFO, "New WebSocket client connected.");

            std::thread([
                            &running, &is_capture, &is_process, &frame_queue_mutex, &frame_queue, &processed_frame_queue_mutex, 
                            &processed_frame_queue, &identify_mutex, &identify_queue, &new_person_ptr, &new_person_mutex, &logger,
                            &capture, &process, &client_connected, &client_cv, &client_mutex
                        ](tcp::socket sock) mutable {
                try {
                    websocket::stream<tcp::socket> ws{std::move(sock)};
                    ws.accept();

                    while (running) {
                        
                        // Read incoming messages

                        std::thread([&ws, &running, &logger, &is_capture, &is_process, &capture, &process, 
                                    &new_person_mutex, &new_person_ptr, &client_connected, &client_cv, &client_mutex]() {
                            beast::flat_buffer buffer;
                            boost::system::error_code ec;
                            while (running) {
                                ws.read(buffer, ec);
                                
                                if (ec) {
                                    logger.log(Logging::LogStatus::INFO, "Client disconnected: " + ec.message());
                                    {
                                        std::lock_guard<std::mutex> lock(client_mutex);
                                        client_connected = false;
                                    }
                                    client_cv.notify_all();
                                    break;
                                }
                                
                                if (buffer.size() > 0) {
                                    std::string msg = beast::buffers_to_string(buffer.data());
                                    buffer.consume(buffer.size());

                                    boost::system::error_code jec;
                                    json::value jv = json::parse(msg, jec);

                                    if (jec) {
                                        logger.log(Logging::LogStatus::WARNING, "Invalid JSON received: " + msg);
                                    } else {
                                        json::object obj = jv.as_object();

                                        // Handle recording
                                        if (obj.if_contains("type") && obj["type"].as_string() == "recording") {
                                            std::lock_guard<std::mutex> lock(capture);
                                            is_capture = obj.if_contains("value") && obj["value"].as_bool();
                                            logger.log(Logging::LogStatus::INFO, is_capture ? "Client started recording." : "Client stopped recording.");
                                        }

                                        // Handle processing
                                        if (obj.if_contains("type") && obj["type"].as_string() == "process") {
                                            std::lock_guard<std::mutex> lock(process);
                                            is_process = obj.if_contains("value") && obj["value"].as_bool();
                                            logger.log(Logging::LogStatus::INFO, is_process ? "Client started processing." : "Client stopped processing.");
                                        }

                                        // Handle add_identify
                                        if (obj.if_contains("type") && obj["type"].as_string() == "add_identify" && obj.if_contains("info")) {
                                            auto info = obj["info"].as_object();
                                            std::string name = info.if_contains("name") ? std::string(info["name"].as_string().c_str()) : "";
                                            int old = info.if_contains("old") ? static_cast<int>(info["old"].as_int64()) : 0;
                                            if (!name.empty() && old > 0) {
                                                {
                                                    std::lock_guard<std::mutex> lock(new_person_mutex);
                                                    new_person_ptr = std::make_unique<People>(People(0, name, old));
                                                }
                                                logger.log(Logging::LogStatus::INFO, "Received add_identify: " + name + ", tuổi: " + std::to_string(old));
                                            }
                                        }

                                        if (obj.if_contains("type") && obj["type"].as_string() == "shutdown" && obj.if_contains("value") && obj["value"].as_bool()) {
                                            logger.log(Logging::LogStatus::WARNING, "System shutdown requested by client");
                                            // Stop all threads by setting running to false
                                            running = false;
                                        }
                                    }
                                } 
                                
                            }
                        }).detach();


                        
                        beast::flat_buffer buffer;

                        // Send raw frame if available
                        cv::Mat frame;
                        {
                            std::lock_guard<std::mutex> lock(frame_queue_mutex);
                            if (!frame_queue.empty()) {
                                frame = frame_queue.front().clone();
                                frame_queue.pop();
                            }
                        }
                        std::vector<uchar> buf;
                        std::ostringstream oss;
                        if (!frame.empty()) {
                            cv::imencode(".jpg", frame, buf);
                            std::string img_b64 = base64_encode(buf);
                            oss << "{\"type\":\"raw\",\"image\":\"" << img_b64 << "\"}";
                            ws.text(true);
                            ws.write(net::buffer(oss.str()));
                        }

                        // Send processed frame if available
                        if (new_person_ptr == nullptr) {
                            std::lock_guard<std::mutex> lock(processed_frame_queue_mutex);
                            if (!processed_frame_queue.empty()) {
                                frame = processed_frame_queue.front().first.clone();
                                processed_frame_queue.pop();
                            }
                        }
                        buf.clear();
                        oss.str("");
                        oss.clear();
                        if (!frame.empty()) {
                            cv::imencode(".jpg", frame, buf);
                            std::string img_b64 = base64_encode(buf);
                            oss << "{\"type\":\"processed\",\"image\":\"" << img_b64 << "\"}";
                            ws.text(true);
                            ws.write(net::buffer(oss.str()));
                        }

                        // send identify queue if available
                        People person;
                        {
                            std::lock_guard<std::mutex> lock(identify_mutex);
                            if (!identify_queue.empty()) {
                                person = identify_queue.front();
                                identify_queue.pop();
                            }
                        }
                        oss.str("");
                        oss.clear();
                        if (!person.getName().empty()) {
                            oss << "{\"type\":\"identify\",\"info\":" << person.toJsonString() << "}";
                            ws.text(true);
                            ws.write(net::buffer(oss.str()));
                        }

                        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
                    }
                } catch (std::exception const& e) {
                    logger.log(Logging::LogStatus::WARNING, "WebSocket session error: " + std::string(e.what()));
                    {
                        std::lock_guard<std::mutex> lock(client_mutex);
                        client_connected = false;
                    }
                    client_cv.notify_all();
                    
                }
            }, std::move(socket)).detach();
        }
    } catch (std::exception const& e) {
        logger.log(Logging::LogStatus::ERROR, "WebSocket server error: " + std::string(e.what()));
    }
}



#endif // IDENTIFY_STREAM_HPP