#ifndef __facenet_tflite_utils_hpp__
#define __facenet_tflite_utils_hpp__

#include <vector>
#include <Eigen/Dense>


inline double compute_cosine_distance(const std::vector<float>& embedding1, const std::vector<float>& embedding2) {
    if (embedding1.size() != embedding2.size() || embedding1.empty()) {
        throw std::invalid_argument("Embeddings must be of the same non-zero size.");
    }
    Eigen::Map<const Eigen::VectorXf> vec1(embedding1.data(), embedding1.size());
    Eigen::Map<const Eigen::VectorXf> vec2(embedding2.data(), embedding2.size());
    float dot_product = vec1.dot(vec2);
    float norm1 = vec1.norm() + 1e-6f; // Adding a small value to avoid division by zero
    float norm2 = vec2.norm() + 1e-6f; // Adding a small value to avoid division by zero
   
    float cosine_similarity = dot_product / (norm1 * norm2);
    return 1.0 - static_cast<double>(cosine_similarity);
}


inline Eigen::MatrixXf convert_embeddings_to_matrix(const std::vector<std::vector<float>>& embeddings, size_t embedding_dim) {
    if (embeddings.empty()) {
        throw std::invalid_argument("Embeddings list is empty.");
    }
    Eigen::MatrixXf mat(embeddings.size(), embedding_dim);
    for (size_t i = 0; i < embeddings.size(); ++i) {
        if (embeddings[i].size() != embedding_dim) {
            throw std::invalid_argument("All embeddings must have the same dimension as embedding_dim.");
        }
        for (size_t j = 0; j < embedding_dim; ++j) {
            mat(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = embeddings[i][j];
        }
    }
    return mat;
}


inline Eigen::Index find_nearest_embedding_index(const Eigen::MatrixXf& value_embeddings, const std::vector<float>& key_embedding) {
    if (key_embedding.empty()) {
        throw std::invalid_argument("Key_embedding is empty.");
    }
    const size_t embedding_dim = key_embedding.size();
    if (value_embeddings.cols() != static_cast<Eigen::Index>(embedding_dim)) {
        throw std::invalid_argument("Embedding dimension mismatch between value_embeddings and key_embedding.");
    }

    Eigen::VectorXf key_vec = Eigen::Map<const Eigen::VectorXf>(key_embedding.data(), embedding_dim);

    // Compute dot products
    Eigen::VectorXf dot_products = value_embeddings * key_vec;

    // Compute norms
    Eigen::VectorXf embeddings_norms = value_embeddings.rowwise().norm();
    float key_norm = key_vec.norm() + 1e-6f;

    // Compute cosine similarities
    Eigen::VectorXf cosine_similarities = dot_products.array() / (embeddings_norms.array() * key_norm);

    // Compute cosine distances
    Eigen::VectorXf cosine_distances = 1.0f - cosine_similarities.array();

    // Find the index of the minimum distance
    Eigen::Index min_index;
    cosine_distances.minCoeff(&min_index);

    return min_index;
}


inline Eigen::Index find_nearest_embedding_index(const std::vector<std::vector<float>>& value_embeddings, const std::vector<float>& key_embedding) {
    if (key_embedding.empty()) {
        throw std::invalid_argument("Key_embedding is empty.");
    }

    const size_t embedding_dim = key_embedding.size();
    Eigen::MatrixXf mat = convert_embeddings_to_matrix(value_embeddings, embedding_dim);

    return find_nearest_embedding_index(mat, key_embedding);
}


inline std::vector<float> get_embedding_from_index(const std::vector<std::vector<float>>& value_embeddings, Eigen::Index index) {
    if (index < 0 || static_cast<size_t>(index) >= value_embeddings.size()) {
        throw std::out_of_range("Index is out of range.");
    }
    return value_embeddings[index];
}


inline std::vector<float> get_embedding_from_index(const std::vector<std::vector<float>>& value_embeddings, size_t index) {
    if (index >= value_embeddings.size()) {
        throw std::out_of_range("Index is out of range.");
    }
    return value_embeddings[index];
}

#endif