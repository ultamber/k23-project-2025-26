#pragma once
#include <vector>
#include <cstdint>
#include "ivfflat.hpp"

class IVFPQ : public IVFFlat
{
public:
    explicit IVFPQ(const Arguments &a, const int &b, const std::vector<VectorData> &c) : IVFFlat(a, b, c) {}
    void buildIndex() override;
    void search(const std::vector<VectorData> &queries, std::ofstream &out) override;

private:
    int M_ = 0;               // subquantizers
    int Ks_ = 256;            // 2^nbits
    std::vector<int> subdim_; // size M_

    std::vector<std::vector<std::vector<float>>> codebooks_; // [M][Ks][subdim[m]]
    std::vector<std::vector<uint8_t>> codes_;                // [N][M]

    // Helper functions
    void makeResiduals(const std::vector<int> &assign, std::vector<std::vector<float>> &residuals);
    void trainPQ(const std::vector<std::vector<float>> &residuals);
    void encodeAll(const std::vector<std::vector<float>> &residuals);
};
