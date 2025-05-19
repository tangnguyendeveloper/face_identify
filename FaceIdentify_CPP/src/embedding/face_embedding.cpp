#include "face_embedding.hpp"

FaceEmbedding::FaceEmbedding(const ModelsConfig &config) {
    this->_input_shape = config.facenet_input_shape();
    this->space_size = this->_input_shape * this->_input_shape * 3;

    this->_model = tflite::FlatBufferModel::BuildFromFile(config.facenet_path().c_str());
    if (!this->_model) {
        throw std::invalid_argument("Failed to load FaceEmbedding tflite model");
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*this->_model, resolver);
    builder(&this->_interpreter);

    if (!this->_interpreter) {
        throw std::runtime_error("Failed to create FaceEmbedding tflite interpreter");
    }
}


const TfLiteTensor* FaceEmbedding::embedding(const cv::Mat &img, const cv::Rect &rect) {
    cv::Mat face = cropImage(img, rect);

    if (face.empty()) return nullptr;
    if (face.channels() == 3) {
        cv::cvtColor(face, face, cv::COLOR_BGR2RGB);
    } else if (face.channels() == 4) {
        cv::cvtColor(face, face, cv::COLOR_BGRA2RGB);
    }

    cv::resize(face, face, cv::Size(this->_input_shape, this->_input_shape), 0, 0, cv::INTER_LINEAR);

    face.convertTo(face, CV_32FC3, SCALE_FACTOR, -1.0); // Normalize to [-1, 1]

    // Resize input tensor if needed
    TfLiteTensor *input = this->_interpreter->tensor(this->_interpreter->inputs()[0]);
    if (input->dims->data[0] != 1 ||
        input->dims->data[1] != this->_input_shape ||
        input->dims->data[2] != this->_input_shape) {
        if (this->_interpreter->ResizeInputTensorStrict(
                this->_interpreter->inputs()[0], {1, this->_input_shape, this->_input_shape, 3}) != kTfLiteOk) {
            throw std::runtime_error("FaceEmbedding failed to resize input tensor");
        }
        if (this->_interpreter->AllocateTensors() != kTfLiteOk) {
            throw std::runtime_error("FaceEmbedding failed to allocate tensors after resize");
        }
        input = this->_interpreter->tensor(this->_interpreter->inputs()[0]);
    }

    // input shape: [1, H, W, 3]
    float *input_data = this->_interpreter->typed_input_tensor<float>(0);
    //copy data to input tensor
    std::memcpy(input_data, face.ptr<float>(), sizeof(float) * space_size);

    if (this->_interpreter->Invoke() != kTfLiteOk) {
        throw std::runtime_error("FaceEmbedding failed to invoke interpreter");
    }

    // Get output tensor
    TfLiteTensor *output = this->_interpreter->tensor(this->_interpreter->outputs()[0]);
    if (output->dims->data[0] != 1 || output->dims->data[1] != EMBEDDING_SIZE) {
        throw std::runtime_error("FaceEmbedding output shape mismatch");
    }
    return output;
}


const std::vector<std::vector<float>> FaceEmbedding::embeddings(const cv::Mat &img, const std::vector<Face> &faces) {
    if (faces.empty()) return {};

    std::vector<std::vector<float>> empty_embeddings(faces.size(), std::vector<float>(EMBEDDING_SIZE, 0.0f));

    constexpr int BATCH_SIZE = 32;
    for (int start = 0; start < faces.size(); start += BATCH_SIZE) {
        int end = std::min(start + BATCH_SIZE, static_cast<int>(faces.size()));
        int batch_size = end - start;

        // Resize input tensor if needed
        TfLiteTensor *input = this->_interpreter->tensor(this->_interpreter->inputs()[0]);
        if (input->dims->data[0] != batch_size ||
            input->dims->data[1] != this->_input_shape ||
            input->dims->data[2] != this->_input_shape) {
            if (this->_interpreter->ResizeInputTensorStrict(
                    this->_interpreter->inputs()[0], {batch_size, this->_input_shape, this->_input_shape, 3}) != kTfLiteOk) {
                throw std::runtime_error("FaceEmbedding failed to resize input tensor");
            }
            if (this->_interpreter->AllocateTensors() != kTfLiteOk) {
                throw std::runtime_error("FaceEmbedding failed to allocate tensors after resize");
            }
            input = this->_interpreter->tensor(this->_interpreter->inputs()[0]);
        }

        // input shape: [BATCH_SIZE, H, W, 3]
        float *input_data = this->_interpreter->typed_input_tensor<float>(0);
        for (int i = 0; i < batch_size; ++i) {
            const Face &face = faces[start + i];
            cv::Mat face_img = cropImage(img, face.bbox.getRect());
            if (face_img.empty()) {
                std::memset(input_data + i * space_size, 0, sizeof(float) * space_size);
                continue;
            }
            if (face_img.channels() == 3) {
                cv::cvtColor(face_img, face_img, cv::COLOR_BGR2RGB);
            } else if (face_img.channels() == 4) {
                cv::cvtColor(face_img, face_img, cv::COLOR_BGRA2RGB);
            }
            cv::resize(face_img, face_img, cv::Size(this->_input_shape, this->_input_shape), 0, 0, cv::INTER_LINEAR);
            face_img.convertTo(face_img, CV_32FC3, SCALE_FACTOR, -1.0); // Normalize to [-1, 1]
            std::memcpy(input_data + i * space_size, face_img.ptr<float>(), sizeof(float) * space_size);
        }

        if (this->_interpreter->Invoke() != kTfLiteOk) {
            throw std::runtime_error("FaceEmbedding failed to invoke interpreter");
        }

        // Get output tensor
        TfLiteTensor *output = this->_interpreter->tensor(this->_interpreter->outputs()[0]);
        if (output->dims->data[0] != batch_size || output->dims->data[1] != EMBEDDING_SIZE) {
            throw std::runtime_error("FaceEmbedding output shape mismatch");
        }

        // Copy output data to embeddings
        for (int i = 0; i < batch_size; ++i) {
            std::vector<float> embedding(EMBEDDING_SIZE);
            std::memcpy(embedding.data(), output->data.f + i * EMBEDDING_SIZE, sizeof(float) * EMBEDDING_SIZE);
            empty_embeddings[start + i] = embedding;
        }

    }

    return empty_embeddings;
}