#pragma once
#include "arguments.hpp"
#include "dataset.hpp"
#include <fstream>
#include <utility>
#include <vector>
#include <chrono>

using namespace std::chrono;
using namespace std;

class Neighborhood {
public:
    int VectorId;
    double DiscoveryTime;
    vector<pair<double, int>> Neighbors;
};

class SearchMethod {
public:
    explicit SearchMethod(const Arguments& a, const int b, const vector<VectorData> c) : Args(a), Dim(b), Data(c) {}

    virtual void buildIndex() = 0;
    virtual void search(const vector<VectorData> &queries, ofstream& out) = 0;
    virtual void setUpGroundTruth(const vector<VectorData> &queries);

protected:
    Arguments Args;
    int Dim = 0;
    vector<VectorData> Data;
    vector<Neighborhood> GroundTruth;
    double TotalTrue;
    double TotalApproximation;
    double TotalAF;
    double TotalRecall;
    double QPS;

    double l2(const vector<float> &a, const vector<float> &b);

    void calculateGroundTruth(const vector<VectorData> &queries, bool storeInFile);

    void readGroundTruthFromFile(const vector<VectorData> &queries);

    virtual void calculatePerQueryMetrics(int queryId, int queryIndex, double tApproximate, vector<pair<double, int>> distApproximate, vector<int> rlist, ofstream& out);

    virtual void printSummary(int qCount, ofstream &out);
};
