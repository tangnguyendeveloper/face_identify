#ifndef _stream_hpp
#define _stream_hpp


#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <sstream>
#include <iomanip>

#include "loging.hpp"

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;


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


// Stream frames from frame_queue and processed_frame_queue to WebSocket clients
void streamWebsocketThread(
    std::atomic<bool>& running,
    std::mutex& frame_mutex,
    std::queue<cv::Mat>& frame_queue,
    std::mutex& processed_frame_mutex,
    std::queue<std::pair<cv::Mat, std::vector<std::vector<float>>>>& processed_frame_queue,
    const Logging& logger,
    const std::string& host = "127.0.0.1",
    unsigned short port = 9002
) {
    try {
        net::io_context ioc{1};

        tcp::endpoint endpoint(net::ip::make_address(host), port);
        tcp::acceptor acceptor{ioc, endpoint};
        logger.log(Logging::LogStatus::INFO, "WebSocket server started at ws://" + host + ":" + std::to_string(port));

        while (running) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);

            std::thread([&running, &frame_mutex, &frame_queue, &processed_frame_mutex, &processed_frame_queue, &logger](tcp::socket sock) mutable {
                try {
                    websocket::stream<tcp::socket> ws{std::move(sock)};
                    ws.accept();

                    while (running) {
                        // Send raw frame if available
                        {
                            std::lock_guard<std::mutex> lock(frame_mutex);
                            if (!frame_queue.empty()) {
                                cv::Mat frame = frame_queue.front();
                                frame_queue.pop();
                                std::vector<uchar> buf;
                                cv::imencode(".jpg", frame, buf);
                                std::string img_b64 = base64_encode(buf);
                                std::ostringstream oss;
                                oss << "{\"type\":\"raw\",\"image\":\"" << img_b64 << "\"}";
                                ws.text(true);
                                ws.write(net::buffer(oss.str()));
                            }
                        }

                        // Send processed frame if available
                        {
                            std::lock_guard<std::mutex> lock(processed_frame_mutex);
                            if (!processed_frame_queue.empty()) {
                                cv::Mat processed_frame = processed_frame_queue.front().first;
                                processed_frame_queue.pop();
                                std::vector<uchar> buf;
                                cv::imencode(".jpg", processed_frame, buf);
                                std::string img_b64 = base64_encode(buf);
                                std::ostringstream oss;
                                oss << "{\"type\":\"processed\",\"image\":\"" << img_b64 << "\"}";
                                ws.text(true);
                                ws.write(net::buffer(oss.str()));
                            }
                        }

                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                } catch (std::exception const& e) {
                    logger.log(Logging::LogStatus::WARNING, "WebSocket session error: " + std::string(e.what()));
                }
            }, std::move(socket)).detach();
        }
    } catch (std::exception const& e) {
        logger.log(Logging::LogStatus::ERROR, "WebSocket server error: " + std::string(e.what()));
    }
}



#endif // _stream_hpp