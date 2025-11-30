#pragma once
#include "../../search_method.hpp"
#include <vector>

class IVFFlat : public SearchMethod
{
public:
    explicit IVFFlat(const Arguments &a, const int &b, const vector<VectorData> &c)
        : SearchMethod(a, b, c)
    {
        Kclusters = Args.kclusters;
        Nprobe = Args.nprobe;
    }
    void buildIndex() override;
    void search(const vector<VectorData> &queries, ofstream &out) override;

protected:
    int Kclusters; // kclusters
    int Nprobe;
    double SilhouetteScore = 0.0;
    vector<vector<float>> Centroids; // [k][dim]
    vector<vector<int>> Lists;       // [k] -> ids

    void kmeansWithPP(
        const vector<vector<float>> &P,
        int k,
        vector<vector<float>> &centroids);
    double calculateSilhouetteScore();
};
