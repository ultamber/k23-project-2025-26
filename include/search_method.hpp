#pragma once
#include "arguments.hpp"
#include "dataset.hpp"
#include <fstream>
#include <vector>

class Neighborhood {
public:
    int VectorId;
    double DiscoveryTime;
    std::vector<std::pair<double, int>> Neighbors;
};

class SearchMethod {
public:
    explicit SearchMethod(const Arguments& a, const int b, const std::vector<VectorData> c) : Args(a), Dim(b), Data(c) {}

    virtual void buildIndex() = 0;
    virtual void search(const std::vector<VectorData> &queries, std::ofstream& out) = 0;
    virtual void setUpGroundTruth(const std::vector<VectorData> &queries);

protected:
    Arguments Args;
    int Dim = 0;
    std::vector<VectorData> Data;
    std::vector<Neighborhood> GroundTruth;

    double l2(const std::vector<float> &a, const std::vector<float> &b);

    void calculateGroundTruth(const std::vector<VectorData> &queries, bool storeInFile);

    void readGroundTruthFromFile(const std::vector<VectorData> &queries);
};
