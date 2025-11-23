#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include "../search_method.hpp"

class LSH : public SearchMethod
{
public:
    explicit LSH(const Arguments &a, const int &b, const vector<VectorData> &c): SearchMethod(a, b, c){}
    void buildIndex() override;
    void search(const vector<VectorData> &queries, ofstream &out) override;

private:
    const uint64_t MOD_M = 4294967291ULL; // 2^32 - 5, a large prime
    // --- LSH parameters ---
    size_t tableSize_ = 0;                                                        // TableSize = n / 4
    vector<vector<long long>> r_;                                       // random coefficients
    vector<vector<vector<pair<int, uint64_t>>>> tables_; // L x TableSize buckets
    // L × k random projection vectors
    vector<vector<vector<float>>> a_; // [L][k][dim]
    // L × k random offsets t in [0,w)
    vector<vector<float>> t_; // [L][k]
    double w_ = 4.0;
    // static constexpr uint64_t MOD_M = (1ull << 32) - 5;
    uint64_t keyFor(const vector<float> &v, int li) const;

    // Safe modular helpers
    inline static long long mod_ll(long long a, long long m) {
        long long r = a % m;
        return (r < 0) ? r + m : r;
    }

    inline static uint64_t mod_u64(uint64_t a, uint64_t m) {
        return (m == 0) ? 0 : (a % m);
    }

    // Compute base LSH function h_j(p)
    inline long long hij(const vector<float> &v, int li, int j) const {
        double dot = 0.0;
        for (int d = 0; d < Dim; ++d)
            dot += a_[li][j][d] * v[d];
        return (long long)floor((dot + t_[li][j]) / w_);
    }

    // Compute amplified ID(p) = Σ r_j h_j(p) mod M  ref 21
    uint64_t computeID(const vector<float> &v, int li) const;
};
