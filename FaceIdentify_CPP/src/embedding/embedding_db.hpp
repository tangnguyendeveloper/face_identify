#ifndef __embedding_db_hpp__
#define __embedding_db_hpp__

#include "utils.hpp"
#include <unordered_map>
#include <type_traits>
#include <fstream>


template<typename InfoType>
class EmbeddingDB {
public:
    using EmbeddingType = Eigen::MatrixXf;
    using InfoTypeT = InfoType;

    // Constructor
    EmbeddingDB() = default;
    explicit EmbeddingDB(size_t embedding_dim);
    EmbeddingDB(const EmbeddingDB&) = delete; // Disable copy constructor
    EmbeddingDB& operator=(const EmbeddingDB&) = delete; // Disable copy assignment

    //destructor
    ~EmbeddingDB() {this->clear();}

    // Add an embedding with associated info
    bool insert(const EmbeddingType& embedding, const InfoType& info);
    bool insert (const EmbeddingType& embeddings, const std::vector<InfoType>& infos);
    bool insert(const std::vector<float>& embedding, const InfoType& info);
    bool insert(const std::vector<std::vector<float>>& embeddings, const std::vector<InfoType>& infos);

    // delete an embedding by index
    bool erase(size_t idx);
    bool erase(const InfoType& info);
    bool erase(const std::vector<InfoType>& infos);
    bool erase(const std::vector<size_t>& idxs);

    // query information of nearest embedding
    std::pair<size_t, double> query_nearest(const EmbeddingType& embedding) const;
    std::pair<size_t, double> query_nearest(const std::vector<float>& embedding) const;

    // store to file
    bool store(const std::string& filename) const;

    // load from file
    bool load(const std::string& filename);

    // Get all embeddings
    const EmbeddingType& embeddings() const {
        return embeddings_;
    }

    // Get all infos
    const std::vector<InfoType>& infos() const {
        return infos_;
    }

    // Get embedding dimension
    size_t embedding_dim() const {
        return embedding_dim_;
    }

    // Clear database
    void clear() {
        embeddings_.resize(0, 0);
        infos_.clear();
    }

    // Get size (number of embeddings)
    size_t size() const {
        return infos_.size();
    }

    // Access embedding and info by index
    EmbeddingType embedding(size_t idx) const {
        if (embeddings_.rows() == 0) throw std::out_of_range("No embeddings stored");
        return embeddings_.row(idx);
    }

    const InfoType& info(size_t idx) const {
        return infos_.at(idx);
    }

private:
    // embeddings_ is a matrix of shape N x embeddings_dim, where N is the number of embeddings stored
    EmbeddingType embeddings_; // Each row is an embedding
    std::vector<InfoType> infos_;
    size_t embedding_dim_ = 0; // Dimension of each embedding
};

#include "embedding_db_impl.hpp" // Include the implementation file

#endif // __embedding_db_hpp__