#ifndef TFLITE_FACE_EMBEDDING_HPP_
#define TFLITE_FACE_EMBEDDING_HPP_

#include <string>
#include <memory>
#include <stdexcept>
#include <vector>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>

#include "config/models_config.hpp"
#include "mtcnn/face.h"
#include "mtcnn/helpers.h"


class FaceEmbedding {

    private:
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
    int _input_shape;
    int space_size = 0;


    FaceEmbedding(const FaceEmbedding&) = delete;
    FaceEmbedding& operator=(const FaceEmbedding&) = delete;

    public:
    FaceEmbedding(const ModelsConfig& config);
    ~FaceEmbedding() = default;

    const std::vector<std::vector<float>> embeddings(const cv::Mat& img, const std::vector<Face>& faces);

    inline const TfLiteTensor* embedding(const cv::Mat& img, const Face& face) {
        return this->embedding(img, face.bbox.getRect());
    }

    inline const TfLiteTensor* embedding(const cv::Mat& img, const BBox& bbox) {
        return this->embedding(img, bbox.getRect());
    }

    const TfLiteTensor* embedding(const cv::Mat& img, const cv::Rect& rect);

    
    static constexpr int EMBEDDING_SIZE = 512;
    static constexpr float SCALE_FACTOR = 1.0 / 127.5;
};


#endif // TFLITE_FACE_EMBEDDING_HPP_