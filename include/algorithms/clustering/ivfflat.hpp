#pragma once
#include "../../search_method.hpp"
#include <vector>

class IVFFlat : public SearchMethod
{
public:
    explicit IVFFlat(const Arguments &a, const int &b, const vector<VectorData> &c) : SearchMethod(a, b, c) {}
    void buildIndex() override;
    void search(const std::vector<VectorData> &queries, std::ofstream &out) override;

    double silhouetteScore();
    std::vector<double> silhouettePerCluster();

protected:
    int Kclusters = Args.kclusters; // kclusters
    int Nprobe = Args.nprobe;
    vector<vector<float>> Centroids; // [k][dim]
    vector<vector<int>> Lists;       // [k] -> ids

    void kmeansWithPP(const vector<vector<float>> &P, int k,
                      unsigned seed,
                      vector<vector<float>> &centroids);
};
