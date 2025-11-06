#pragma once
#include <vector>
#include <cstdint>
#include "../../search_method.hpp"


class ClusteringSearchMethod : public SearchMethod
{
public:
    std::vector<double> silhouettePerCluster() const;
    double silhouetteScore() const;
    int findOptimalK(int kmin, int kmax, unsigned seed, int step = 5);

    int findOptimalKEnhanced(int kmin, int kmax, unsigned seed, int step = 5);

protected:
    Dataset data_;
    int dim_ = 0;
    int k_ = 0; // kclusters
    int nprobe_ = 0;

    std::vector<std::vector<float>> centroids_; // [k][dim]
    std::vector<std::vector<int>> lists_;       // [k] -> ids
    void kmeans(const std::vector<std::vector<float>> &points, int k, int iters, unsigned seed);

    void kmeanspp(const std::vector<std::vector<float>> &P, int k,
                  std::vector<std::vector<float>> &centroids, unsigned seed);
    void kmeansWithPP(const std::vector<std::vector<float>> &P, int k,
                      int iters, unsigned seed,
                      std::vector<std::vector<float>> &centroids,
                      std::vector<int> *outAssign = nullptr);
    double computeSilhouetteForK(int k, unsigned seed);
    double l2(const std::vector<float> &a, const std::vector<float> &b) const;
}
