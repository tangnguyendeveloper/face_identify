
template<typename InfoType>
EmbeddingDB<InfoType>::EmbeddingDB(size_t embedding_dim)
    : embedding_dim_(embedding_dim) {
    embeddings_.resize(0, embedding_dim_);
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::insert(const EmbeddingType& embedding, const InfoType& info) {
    if (embedding.cols() != static_cast<Eigen::Index>(embedding_dim_)) {
        return false;
    }
    if (embedding.rows() != 1) {
        return false;
    }
    // Append embedding
    Eigen::Index old_rows = embeddings_.rows();
    embeddings_.conservativeResize(old_rows + 1, embedding_dim_);
    embeddings_.row(old_rows) = embedding;
    infos_.push_back(info);
    return true;
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::insert(const EmbeddingType& embeddings, const std::vector<InfoType>& infos) {
    if (embeddings.cols() != static_cast<Eigen::Index>(embedding_dim_)) {
        return false;
    }
    if (static_cast<size_t>(embeddings.rows()) != infos.size()) {
        return false;
    }
    Eigen::Index old_rows = embeddings_.rows();
    Eigen::Index new_rows = old_rows + embeddings.rows();
    embeddings_.conservativeResize(new_rows, embedding_dim_);
    embeddings_.block(old_rows, 0, embeddings.rows(), embedding_dim_) = embeddings;
    infos_.insert(infos_.end(), infos.begin(), infos.end());
    return true;
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::insert(const std::vector<float>& embedding, const InfoType& info) {
    if (embedding_dim_ == 0) {
        embedding_dim_ = embedding.size();
        embeddings_.resize(0, embedding_dim_);
    }
    if (embedding.size() != embedding_dim_) {
        return false;
    }
    Eigen::RowVectorXf emb = Eigen::Map<const Eigen::RowVectorXf>(embedding.data(), embedding_dim_);
    return insert(emb, info);
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::insert(const std::vector<std::vector<float>>& embeddings, const std::vector<InfoType>& infos) {
    if (embeddings.empty()) return false;
    if (embeddings.size() != infos.size()) {
        return false;
    }
    size_t dim = embeddings[0].size();
    if (embedding_dim_ == 0) {
        embedding_dim_ = dim;
        embeddings_.resize(0, embedding_dim_);
    }
    if (dim != embedding_dim_) {
        return false;
    }
    Eigen::MatrixXf mat = convert_embeddings_to_matrix(embeddings, embedding_dim_);
    return insert(mat, infos);
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::erase(size_t idx) {
    if (idx >= infos_.size()) return false;
    // Remove row from embeddings_
    if (embeddings_.rows() > 1) {
        Eigen::MatrixXf new_embeddings(embeddings_.rows() - 1, embedding_dim_);
        if (idx > 0)
            new_embeddings.topRows(idx) = embeddings_.topRows(idx);
        if (idx + 1 < static_cast<size_t>(embeddings_.rows()))
            new_embeddings.bottomRows(embeddings_.rows() - idx - 1) = embeddings_.bottomRows(embeddings_.rows() - idx - 1);
        embeddings_ = new_embeddings;
    } else {
        embeddings_.resize(0, embedding_dim_);
    }
    infos_.erase(infos_.begin() + idx);
    return true;
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::erase(const InfoType& info) {
    auto it = std::find(infos_.begin(), infos_.end(), info);
    if (it == infos_.end()) return false;
    size_t idx = std::distance(infos_.begin(), it);
    return erase(idx);
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::erase(const std::vector<InfoType>& infos) {
    bool result = true;
    for (const auto& info : infos) {
        result &= erase(info);
    }
    return result;
}


template<typename InfoType>
bool EmbeddingDB<InfoType>::erase(const std::vector<size_t>& idxs) {
    // Remove in reverse order to keep indices valid
    std::vector<size_t> sorted_idxs = idxs;
    std::sort(sorted_idxs.rbegin(), sorted_idxs.rend());
    bool result = true;
    for (size_t idx : sorted_idxs) {
        result &= erase(idx);
    }
    return result;
}


template<typename InfoType>
std::pair<size_t, double> EmbeddingDB<InfoType>::query_nearest(const EmbeddingType& embedding) const {
    if (embeddings_.rows() == 0) throw std::runtime_error("No embeddings in database.");
    if (embedding.cols() != static_cast<Eigen::Index>(embedding_dim_)) {
        throw std::invalid_argument("Embedding dimension mismatch.");
    }
    Eigen::VectorXf query_vec = embedding.row(0);
    Eigen::VectorXf dot_products = embeddings_ * query_vec;
    Eigen::VectorXf db_norms = embeddings_.rowwise().norm();
    float query_norm = query_vec.norm() + 1e-6f;
    Eigen::VectorXf cosine_similarities = dot_products.array() / (db_norms.array() * query_norm);
    Eigen::VectorXf cosine_distances = 1.0f - cosine_similarities.array();
    Eigen::Index min_idx;
    float min_dist = cosine_distances.minCoeff(&min_idx);
    return {static_cast<size_t>(min_idx), static_cast<double>(min_dist)};
}

template<typename InfoType>
std::pair<size_t, double> EmbeddingDB<InfoType>::query_nearest(const std::vector<float>& embedding) const {
    if (embedding.size() != embedding_dim_) {
        throw std::invalid_argument("Embedding dimension mismatch.");
    }
    Eigen::RowVectorXf emb = Eigen::Map<const Eigen::RowVectorXf>(embedding.data(), embedding_dim_);
    
    return query_nearest(emb);
}

template<typename InfoType>
bool EmbeddingDB<InfoType>::store(const std::string& filename) const {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;
    // Write embedding_dim_ and number of embeddings
    size_t n = size();
    ofs.write(reinterpret_cast<const char*>(&embedding_dim_), sizeof(embedding_dim_));
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
    // Write embeddings
    if (n > 0) {
        ofs.write(reinterpret_cast<const char*>(embeddings_.data()), sizeof(float) * n * embedding_dim_);
    }
    // Write infos
    for (const auto& info : infos_) {
        ofs << info << '\n';
    }
    return ofs.good();
}

template<typename InfoType>
bool EmbeddingDB<InfoType>::load(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) return false;
    clear();
    size_t n = 0;
    ifs.read(reinterpret_cast<char*>(&embedding_dim_), sizeof(embedding_dim_));
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    embeddings_.resize(n, embedding_dim_);
    if (n > 0) {
        ifs.read(reinterpret_cast<char*>(embeddings_.data()), sizeof(float) * n * embedding_dim_);
    }
    infos_.resize(n);
    std::string line;
    // Read infos
    for (size_t i = 0; i < n; ++i) {
        if (!std::getline(ifs, line)) return false;
        std::istringstream iss(line);
        iss >> infos_[i];
    }
    return ifs.good();
}

// Explicit template instantiation for common types (optional, can be omitted if using only in headers)
// template class EmbeddingDB<std::string>;