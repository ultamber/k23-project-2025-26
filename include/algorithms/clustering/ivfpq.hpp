#pragma once
#include <vector>
#include <cstdint>
#include "ivfflat.hpp"

class IVFPQ : public IVFFlat
{
public:
    explicit IVFPQ(const Arguments &a, const int &b, const vector<VectorData> &c) : IVFFlat(a, b, c) {}
    void buildIndex() override;
    void search(const vector<VectorData> &queries, ofstream &out) override;

private:
    int Ks_ = 1 << Args.nbits;            // 2^nbits
    vector<int> subdim_; // size M_

    vector<vector<vector<float>>> codebooks_; // [M][Ks][subdim[m]]
    vector<vector<uint8_t>> codes_;                // [N][M]

    // Helper functions
    void makeResiduals(const vector<int> &assign, vector<vector<float>> &residuals);
    void trainPQ(const vector<vector<float>> &residuals);
    void encodeAll(const vector<vector<float>> &residuals);
};
