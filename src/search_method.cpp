#include "../include/search_method.hpp"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

void SearchMethod::calculateGroundTruth(const std::vector<VectorData> &queries, bool storeInFile) {
    GroundTruth.reserve(queries.size());

    cout << "Building Ground Truth" << std::endl;
    for(auto &vector : queries){
        cout << "Brute-Force Search for: " << vector.id << endl;

        Neighborhood item;
        item.VectorId = vector.id;
        item.DiscoveryTime = 0.0;

        auto t2 = high_resolution_clock::now();
        std::vector<pair<double, int>> distTrue;
        distTrue.reserve(Data.size());
        for(auto &candidate : Data) {
            if (candidate.id == vector.id) {
                continue;
            }
            distTrue.emplace_back(l2(vector.values, candidate.values), candidate.id);
        }

        nth_element(distTrue.begin(), distTrue.begin() + Args.N, distTrue.end());
        sort(distTrue.begin(), distTrue.begin() + Args.N);
        item.DiscoveryTime += duration<double>(high_resolution_clock::now() - t2).count();

        item.Neighbors = std::vector<pair<double,int>>(distTrue.begin(), distTrue.begin() + Args.N);
        GroundTruth.emplace_back(item);
    }

    if (!storeInFile)
        return;

    ofstream f(Args.gtFile);
    for (const auto& row : GroundTruth) {
        f << row.VectorId << "," << row.DiscoveryTime << ",";
        for (const auto& n : row.Neighbors) {
            f << n.first << "," << n.second;
        }
        f << endl;
    }
    f.close();
}

void SearchMethod::readGroundTruthFromFile(const vector<VectorData> &queries) {
    ifstream f(Args.gtFile);
    vector<Neighborhood> fData;

    string line;
    while (getline(f, line)) {
        stringstream ss(line);

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
        cout << "Invalid data in groundtruth file" << endl;
        calculateGroundTruth(queries, true);
        return;
    }

    GroundTruth = fData;
}

double SearchMethod::l2(const vector<float> &a, const vector<float> &b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return sqrt(s);
}

void SearchMethod::setUpGroundTruth(const vector<VectorData> &queries) {

    if (Args.gtFile.empty()) {
        calculateGroundTruth(queries, false);
        return;
    }

    ifstream f(Args.gtFile);
    if (!f.good()) {
        f.close(); 
        calculateGroundTruth(queries, true);
        return;
    }

    readGroundTruthFromFile(queries);
}

void SearchMethod::calculatePerQueryMetrics(int queryId, int queryIndex, double tApproximate, vector<pair<double, int>> distApproximate, vector<int> rlist, ofstream& out){

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
            << fixed << setprecision(6);
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


void SearchMethod::printSummary(int qCount, ofstream &out){
    out << fixed << std::setprecision(6);
    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << (TotalAF / qCount) << "\n";
    out << "Recall@N: " << (TotalRecall / qCount) << "\n";
    out << "QPS: " << (1.0 / (TotalApproximation / qCount)) << "\n";
    out << "tApproximateAverage: " << (TotalApproximation / qCount) << "\n";
    out << "tTrueAverage: " << (TotalTrue / qCount) << "\n";
};
