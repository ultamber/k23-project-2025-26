#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

struct VectorData
{
    int id;
    std::vector<float> values;
};

class Dataset
{
public:
    std::vector<VectorData> vectors;
    int dimension = 0;
    int count = 0;

    // Helper: read 32-bit big-endian integer
    inline uint32_t readBigEndian(std::ifstream &f)
    {
        unsigned char bytes[4];
        f.read((char *)bytes, 4);
        return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
               (uint32_t(bytes[2]) << 8) | (uint32_t(bytes[3]));
    };

    virtual void load(const std::string &path) = 0;
    virtual ~Dataset() = default;
};


class SIFT_Dataset : public Dataset{
public:
    void load(const std::string &path) override;
};

class MNIST_Dataset: public Dataset{
public:
    void load(const std::string &path) override;
};

