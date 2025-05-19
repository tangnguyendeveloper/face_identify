#ifndef LOGGING_HPP
#define LOGGING_HPP

#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sstream>

class Logging {
public:
    enum class LogStatus { INFO, WARNING, ERROR };

    explicit Logging(const std::string& appName) : appName_(appName) {}

    virtual void log(LogStatus status, const std::string& message) const {
        std::cout << "[" << getCurrentDateTime() << "][" << appName_ << "]["
                  << logStatusToString(status) << "] " << message << std::endl;
    }

protected:
    std::string appName_;

    static std::string getCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_now;
    #ifdef _WIN32
        localtime_s(&tm_now, &now_time);
    #else
        localtime_r(&now_time, &tm_now);
    #endif
        std::ostringstream oss;
        oss << std::put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

    static std::string logStatusToString(LogStatus status) {
        switch (status) {
            case LogStatus::INFO: return "INFO";
            case LogStatus::WARNING: return "WARNING";
            case LogStatus::ERROR: return "ERROR";
            default: return "UNKNOWN";
        }
    }
};


class FileLogging : public Logging {
public:
    FileLogging(const std::string& appName, const std::string& filename)
        : Logging(appName), filename_(filename) {}

    void log(LogStatus status, const std::string& message) const override {
        std::ofstream fileStream(filename_, std::ios::app);
        if (fileStream.is_open()) {
            fileStream << "[" << getCurrentDateTime() << "][" << appName_ << "]["
                       << logStatusToString(status) << "] " << message << std::endl;
        }
    }

private:
    std::string filename_;
};

#endif // LOGGING_HPP