#pragma once
#include <unordered_map>
#include <vector>
#include "../search_method.hpp"

class Hypercube : public SearchMethod{
public:
    explicit Hypercube(const Arguments &a, const int &b, const vector<VectorData> &c) : SearchMethod(a, b, c) {}
    void buildIndex() override;
    void search(const vector<VectorData> &queries, ofstream &out) override;

private:
    int kproj_;  // d' in the PDF
    float w_;    // Bucket width for LSH

    // LSH hash function parameters ref 18, 24
    vector<vector<float>> proj_;   // Random projection vectors [kproj_][dim_]
    vector<float> shift_;                // Random shifts [kproj_]

    // Bit mapping functions f_j ref 24
    vector<uint64_t> fA_;  // Hash coefficients for f_j [kproj_]
    vector<uint64_t> fB_;  // Hash offsets for f_j [kproj_]

    // Hypercube storage ref 25
    bool denseCube_;  // Use dense or sparse representation
    vector<vector<int>> cubeDense_;  // Dense: [2^kproj_] vertices
    unordered_map<uint64_t, vector<int>> cubeSparse_;  // Sparse

    // Helper functions
    long long hij(const vector<float> &v, int j) const;
    bool fj(long long h, int j) const;
    uint64_t vertexOf(const vector<float> &v) const;
    vector<uint64_t> probesList(uint64_t base, int kproj, 
                                          int maxProbes, int maxHamming) const;
};
