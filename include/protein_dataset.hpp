#pragma once

#include "../include/dataset.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>

class ProteinEmbedding_Dataset : public Dataset {
public:
    void load(const std::string &path) override {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            throw std::runtime_error("Cannot open file: " + path);

        // Get file size to estimate vector count
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        f.seekg(0, std::ios::beg);

        // Read first dimension to calculate expected vector count
        int dim;
        f.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (dim <= 0 || dim > 10000)
            throw std::runtime_error("Invalid dimension in .fvecs file: " + std::to_string(dim));

        dimension = dim;
        size_t vector_size = sizeof(int) + sizeof(float) * dim;
        size_t n = file_size / vector_size;

        std::cout << "Loading " << n << " vectors of dimension " << dim << " from " << path << std::endl;

        // Reset to beginning and read all vectors
        f.seekg(0, std::ios::beg);
        vectors.resize(n);

        for (size_t i = 0; i < n; ++i) {
            int d;
            f.read(reinterpret_cast<char*>(&d), sizeof(int));
            if (d != dim)
                throw std::runtime_error("Inconsistent dimension at vector " + std::to_string(i));

            vectors[i].id = static_cast<int>(i);
            vectors[i].values.resize(dim);
            f.read(reinterpret_cast<char*>(vectors[i].values.data()), dim * sizeof(float));
        }

        count = static_cast<int>(n);
        f.close();

        std::cout << "Successfully loaded " << count << " protein embeddings" << std::endl;
    }

    void loadIdMapping(const std::string &path) {
        std::ifstream f(path);
        if (!f)
            throw std::runtime_error("Cannot open ID mapping file: " + path);

        proteinIds.clear();
        std::string line;
        while (std::getline(f, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start != std::string::npos)
                proteinIds.push_back(line.substr(start, end - start + 1));
            else
                proteinIds.push_back("");
        }
        f.close();

        if (proteinIds.size() != vectors.size()) {
            std::cerr << "Warning: ID count (" << proteinIds.size() 
                      << ") doesn't match vector count (" << vectors.size() << ")" << std::endl;
        }

        std::cout << "Loaded " << proteinIds.size() << " protein IDs" << std::endl;
    }

    std::string getProteinId(int index) const {
        if (index >= 0 && index < static_cast<int>(proteinIds.size()))
            return proteinIds[index];
        return std::to_string(index);
    }

    const std::vector<std::string>& getProteinIds() const {
        return proteinIds;
    }

    bool hasIdMapping() const {
        return !proteinIds.empty();
    }

private:
    std::vector<std::string> proteinIds;
};