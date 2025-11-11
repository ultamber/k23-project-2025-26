#include "../include/search_method.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

void SearchMethod::calculateGroundTruth(const std::vector<VectorData> &queries, bool storeInFile) {
    GroundTruth.reserve(queries.size());

    std::cout << "Building Ground Truth" << std::endl;
    for(auto &vector : queries){
        std::cout << "Brute-Force Search for: " << vector.id << std::endl;

        Neighborhood item;
        item.VectorId = vector.id;
        item.DiscoveryTime = 0.0;

        auto t2 = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(Data.size());
        for(auto &candidate : Data) {
            if (candidate.id == vector.id) {
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

    if (!storeInFile)
        return;

    std::ofstream f(Args.gtFile);
    for (const auto& row : GroundTruth) {
        f << row.VectorId << "," << row.DiscoveryTime << ",";
        for (const auto& n : row.Neighbors) {
            f << n.first << "," << n.second;
        }
        f << std::endl;
    }
    f.close();
}

void SearchMethod::readGroundTruthFromFile(const std::vector<VectorData> &queries) {
    std::ifstream f(Args.gtFile);
    std::vector<Neighborhood> fData;

    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);

        Neighborhood row;
        char comma;
        ss >> row.VectorId >> comma >> row.DiscoveryTime;

        while (ss.peek() != EOF) {
            double pairFirst;
            int pairSecond;
            ss >> comma >> pairFirst >> comma >> pairSecond;
            row.Neighbors.push_back({pairFirst, pairSecond});
        }
        fData.push_back(row);
    }
    f.close();

    if (fData.size() != queries.size()){
        std::cout << "Invalid data in groundtruth file" << std::endl;
        calculateGroundTruth(queries, true);
        return;
    }

    GroundTruth = fData;
}

double SearchMethod::l2(const std::vector<float> &a, const std::vector<float> &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void SearchMethod::setUpGroundTruth(const std::vector<VectorData> &queries) {

    if (Args.gtFile.empty()) {
        calculateGroundTruth(queries, false);
        return;
    }

    std::ifstream f(Args.gtFile);
    if (!f.good()) {
        f.close(); 
        calculateGroundTruth(queries, true);
        return;
    }

    readGroundTruthFromFile(queries);
}

void SearchMethod::calculatePerQueryMetrics(int queryId, int queryIndex, double tApproximate, std::vector<std::pair<double, int>> distApproximate, std::vector<int> rlist, std::ofstream& out){

        TotalApproximation += tApproximate;

        // Compute true nearest neighbors for evaluation
        Neighborhood trueNeighborhood;
        for (auto item : GroundTruth){
            if (item.VectorId == queryId){
                trueNeighborhood = item;
                break;
            }
        }
        TotalTrue += trueNeighborhood.DiscoveryTime;

        // Calculate quality metrics
        double AFq = 0, recallq = 0;
        for (int i = 0; i < Args.N; ++i)
        {
            double da = distApproximate[i].first, dt = trueNeighborhood.Neighbors[i].first;
            AFq += (dt > 0 ? da / dt : 1.0);

            // Check if approximate neighbor is in true top-N
            int aid = distApproximate[i].second;
            for (const auto& trueNeighbors : trueNeighborhood.Neighbors)
                if (aid == trueNeighbors.second)
                {
                    recallq += 1;
                    break;
                }
        }

        if (Args.N > 0)
        {
            AFq /= Args.N;
            recallq /= Args.N;
        }
        TotalAF += AFq;
        TotalRecall += recallq;

        // Output results
        out << "Query: " << queryIndex << "\n"
            << std::fixed << std::setprecision(6);
        for (int i = 0; i < Args.N; ++i)
        {
            out << "Nearest neighbor-" << (i + 1) << ": " << distApproximate[i].second << "\n";
            out << "distanceApproximate: " << distApproximate[i].first << "\n";
            out << "distanceTrue: " << trueNeighborhood.Neighbors[i].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
        out << "\n";
};


void SearchMethod::printSummary(int qCount, std::ofstream &out){
    out << std::fixed << std::setprecision(6);
    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << (TotalAF / qCount) << "\n";
    out << "Recall@N: " << (TotalRecall / qCount) << "\n";
    out << "QPS: " << (1.0 / (TotalApproximation / qCount)) << "\n";
    out << "tApproximateAverage: " << (TotalApproximation / qCount) << "\n";
    out << "tTrueAvERAGE: " << (TotalTrue / qCount) << "\n";
};
