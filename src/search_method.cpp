#include "../include/search_method.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
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

