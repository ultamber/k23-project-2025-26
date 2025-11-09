#include "../include/search_method.hpp"

double SearchMethod::l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void SearchMethod::setUpGroundTruth(){
    GroundTruth.reserve(Data.size());

    for(auto &vector : Data){
        NeighborInfo item;
        item.VectorId = vector.id;
        item.DiscoveryTime = 0.0;

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(Data.size());
        for(auto &candidate : Data){
            if (candidate.id == vector.id){
                continue;
            }
            distTrue.emplace_back(l2(vector.values, candidate.values), candidate.id);
        }

        std::nth_element(distTrue.begin(), distTrue.begin() + Args.N, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin() + Args.N);
        item.DiscoveryTime += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t2).count();

        item.Neighbors = std::vector<std::pair<double,int>>(distTrue.begin(), distTrue.begin() + Args.N);
        GroundTruth.emplace_back(item);
    }
}

