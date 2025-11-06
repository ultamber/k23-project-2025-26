#include "../../../include/algorithms/clustering/ivfpq.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <iostream>

/**
 * Computes residual vectors r(x) = x - c(x) for all points ref 49
 */
void IVFPQ::makeResiduals(const Dataset &data, const std::vector<int> &assign,
                          std::vector<std::vector<float>> &residuals)
{
    int N = (int)data.vectors.size(), D = data.dimension;
    residuals.assign(N, std::vector<float>(D, 0.0f));
    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        for (int d = 0; d < D; ++d)
            residuals[i][d] = data.vectors[i].values[d] - Centroids[c][d];
    }
}

/**
 * Trains Product Quantizer on residuals ref 49
 * Split residual into M subspaces, run k-means on each with s=2^nbits centroids
 */
void IVFPQ::trainPQ(const std::vector<std::vector<float>> &R)
{
        int D = Dim, M = M_;

        // slide 49Split r(x) into M parts
        subdim_.assign(M, D / M);
        for (int i = 0; i < D % M; ++i)
            ++subdim_[i]; // Distribute remainder

    codebooks_.assign(M, {});
    unsigned seed = (unsigned)Args.seed;

    int offset = 0;
    for (int m = 0; m < M; ++m)
    {
        int sd = subdim_[m];

        // Extract subspace m from all residuals
        std::vector<std::vector<float>> subspace;
        subspace.reserve(R.size());
        for (const auto &r : R)
            subspace.emplace_back(r.begin() + offset, r.begin() + offset + sd);

        // slide 49 Lloyd's creates clustering with s = 2^nbits centroids
        std::vector<std::vector<float>> cb;

        kmeansWithPP(subspace, Ks_, 20, seed + m, cb, nullptr);
        kmeansWithPP(subspace, Kclusters, 20, (unsigned)Args.seed + m, Centroids, nullptr);
        codebooks_[m] = std::move(cb);

        offset += sd;
    }
}

/**
 * Encodes all residuals using trained PQ ref 49
 */
void IVFPQ::encodeAll(const std::vector<std::vector<float>> &R)
{
    int N = (int)R.size(), M = M_;
    codes_.assign(N, std::vector<uint8_t>(M, 0));

    int offset = 0;
    for (int m = 0; m < M; ++m)
    {
        int sd = subdim_[m];

        // slide 49 For each subspace, find nearest centroid
        for (int i = 0; i < N; ++i)
        {
            int best = 0;
            double bd = std::numeric_limits<double>::infinity();

            for (int k = 0; k < Ks_; ++k)
            {
                double s = 0.0;
                for (int d = 0; d < sd; ++d)
                {
                    double diff = R[i][offset + d] - codebooks_[m][k][d];
                    s += diff * diff;
                }
                if (s < bd)
                {
                    bd = s;
                    best = k;
                }
            }

            // slide 49 code_i(x) = argmin_h ||r_i(x) - c_{i,h}||_2
            codes_[i][m] = (uint8_t)best;
        }
        offset += sd;
    }
}

/**
 * Builds the IVFPQ index ref 49
 */
void IVFPQ::buildIndex(const Dataset &data)
{
    Data = data;
    Dim = data.dimension;

    Kclusters = Args.kclusters;         // Number of coarse cells (k in slides)
    Nprobe = Args.nprobe;       // Number of cells to probe (b in slides)
    M_ = Args.Msubvectors;       // Number of subquantizers (M in slides)
    Ks_ = 1 << Args.nbits;       // Codebook size per subquantizer (s = 2^nbits in slides)

    const int N = (int)Data.vectors.size();
    if (N == 0)
    {
        Centroids.clear();
        Lists.clear();
        return;
    }

    // slide 49, Run Lloyd's on subset X' ~ √n to obtain centroids {c_j}
    int trainN = std::max(Kclusters, (int)std::sqrt((double)N));
    trainN = std::min(trainN, N);

    std::mt19937_64 rng(Args.seed);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(trainN);

    std::vector<std::vector<float>> Ptrain;
    Ptrain.reserve(trainN);
    for (int id : idx)
        Ptrain.push_back(Data.vectors[id].values);

    kmeansWithPP(Ptrain, Kclusters, 40, (unsigned)Args.seed, Centroids, nullptr);

    // slide 49, Assign ALL points to nearest centroid
    Lists.assign(Kclusters, {});
    std::vector<int> fullAssign(N, -1);
    for (int i = 0; i < N; ++i)
    {
        const auto &x = Data.vectors[i].values;
        int best = 0;
        double bd = l2(x, Centroids[0]);
        for (int c = 1; c < Kclusters; ++c)
        {
            double d = l2(x, Centroids[c]);
            if (d < bd)
            {
                bd = d;
                best = c;
            }
        }
        Lists[best].push_back(i);
        fullAssign[i] = best;
    }

    // slide 49,  Compute residual vectors r(x) = x - c(x)
    std::vector<std::vector<float>> R;
    makeResiduals(Data, fullAssign, R);

    // slide 49, Train product quantizer on residuals
    trainPQ(R);

    // slide 49,  Encode ALL residuals with trained PQ
    // PQ(x) = [code_1(x), ..., code_M(x)]
    encodeAll(R);
}

/**
 * Performs IVFPQ search ref 50
 */
void IVFPQ::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "IVFPQ\n\n";

    const double Rrad = Args.R;
    const bool doRange = Args.rangeSearch;
    const int Nret = std::max(1, Args.N);
    const int rerankTop = 100; // Re-rank top T candidates with exact distances

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int counted = 0;
    int Q = (int)queries.vectors.size();

    if (Args.maxQueries > 0)
        Q = std::min(Q, Args.maxQueries);

    for (int qi = 0; qi < Q; ++qi)
    {
        const auto &q = queries.vectors[qi].values;

        // === Approximate search using IVFPQ ===
        auto t0 = high_resolution_clock::now();

        // slide 50Score centroids - compute ||q - c_j||_2
        std::vector<std::pair<double, int>> centroidDists;
        centroidDists.reserve(Kclusters);
        for (int c = 0; c < Kclusters; ++c)
        {
            double s = 0.0;
            for (int d = 0; d < Dim; ++d)
            {
                double diff = q[d] - Centroids[c][d];
                s += diff * diff;
            }
            centroidDists.emplace_back(s, c); // Store squared distance
        }

        // Select top b probes
        int probeCount = std::min(Nprobe, (int)centroidDists.size());
        std::nth_element(centroidDists.begin(),
                        centroidDists.begin() + probeCount,
                        centroidDists.end());
        std::sort(centroidDists.begin(), centroidDists.begin() + probeCount);

        std::vector<std::pair<double, int>> adcCandidates; // (adc_distance_squared, id)
        std::vector<int> rlist;

        // slide 50 For each selected c_j
        for (int pi = 0; pi < probeCount; ++pi)
        {
            int cid = centroidDists[pi].second;

            // Compute residual: r(q) = q - c_j
            std::vector<float> rq(Dim);
            for (int d = 0; d < Dim; ++d)
                rq[d] = q[d] - Centroids[cid][d];

            // Split r(q) = [r_1(q), ..., r_M(q)]
            // Define LUT[i][h] = ||r_i(q) - c_{i,h}||_2 ref 50)
            std::vector<std::vector<double>> LUT(M_, std::vector<double>(Ks_, 0.0));
            int offset = 0;
            for (int m = 0; m < M_; ++m)
            {
                int sd = subdim_[m];
                for (int k = 0; k < Ks_; ++k)
                {
                    double s = 0.0;
                    for (int d = 0; d < sd; ++d)
                    {
                        double diff = rq[offset + d] - codebooks_[m][k][d];
                        s += diff * diff;
                    }
                    LUT[m][k] = s; // Squared distance
                }
                offset += sd;
            }

            // slide 50 Asymmetric Distance Computation (ADC)
            // For x in U, d(q,x) = Σ_i LUT[i][code_i(x)]
            for (int id : Lists[cid])
            {
                double adc_sqr = 0.0;
                for (int m = 0; m < M_; ++m)
                    adc_sqr += LUT[m][codes_[id][m]];
                
                adcCandidates.emplace_back(adc_sqr, id);

                // Range search in PQ space (optional)
                if (doRange)
                {
                    double adc = std::sqrt(adc_sqr);
                    if (adc <= Rrad)
                        rlist.push_back(id);
                }
            }
        }

        // Keep top T candidates for re-ranking
        std::vector<std::pair<double, int>> reranked;
        if (!adcCandidates.empty())
        {
            int T = std::max(Nret, rerankTop);
            T = std::min(T, (int)adcCandidates.size());
            std::nth_element(adcCandidates.begin(),
                           adcCandidates.begin() + T,
                           adcCandidates.end());
            
            // Re-rank top T with exact distances
            reranked.reserve(T);
            for (int i = 0; i < T; ++i)
            {
                int id = adcCandidates[i].second;
                double exactDist = l2(q, Data.vectors[id].values);
                reranked.emplace_back(exactDist, id);
            }
        }

        // slide 50,Return R nearest points
        int keepApprox = std::min(Nret, (int)reranked.size());
        if (keepApprox > 0)
        {
            std::nth_element(reranked.begin(),
                           reranked.begin() + keepApprox,
                           reranked.end());
            std::sort(reranked.begin(), reranked.begin() + keepApprox);
            reranked.resize(keepApprox);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        totalApproxTime += tApprox;

        // === Ground truth for evaluation ===
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(Data.vectors.size());
        for (const auto &v : Data.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);

        int keepTrue = std::min(Nret, (int)distTrue.size());
        if (keepTrue > 0)
        {
            std::nth_element(distTrue.begin(),
                           distTrue.begin() + keepTrue,
                           distTrue.end());
            std::sort(distTrue.begin(), distTrue.begin() + keepTrue);
            distTrue.resize(keepTrue);
        }

        double tTrue = duration<double>(high_resolution_clock::now() - t2).count();
        totalTrueTime += tTrue;

        // === Quality metrics ===
        double AFq = 0.0, Rq = 0.0;
        if (keepApprox > 0 && keepTrue > 0)
        {
            // Approximation Factor
            for (int i = 0; i < keepApprox; ++i)
            {
                double da = reranked[i].first;
                double dt = distTrue[i].first;
                AFq += (dt > 0.0 ? da / dt : 1.0);
            }
            AFq /= keepApprox;

            // Recall@N
            std::unordered_set<int> trueSet;
            for (const auto &p : distTrue)
                trueSet.insert(p.second);
            int hits = 0;
            for (const auto &p : reranked)
                if (trueSet.count(p.second))
                    ++hits;
            Rq = (double)hits / (double)keepTrue;

            totalAF += AFq;
            totalRecall += Rq;
            ++counted;
        }

        // === Output per query ===
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int i = 0; i < keepApprox; ++i)
        {
            out << "Nearest neighbor-" << (i + 1) << ": " << reranked[i].second << "\n";
            out << "distanceApproximate: " << reranked[i].first << "\n";
            out << "distanceTrue: " << distTrue[std::min(i, keepTrue - 1)].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
        out << "\n";
    }

    // === Summary statistics ===
    out << "---- Summary (averages over queries) ----\n";
    out << std::fixed << std::setprecision(6);
    double avgAF = (counted > 0) ? totalAF / counted : 0.0;
    double avgRecall = (counted > 0) ? totalRecall / counted : 0.0;
    double avgApprox = (counted > 0) ? totalApproxTime / counted : 0.0;
    double avgTrue = (counted > 0) ? totalTrueTime / counted : 0.0;
    double qps = (avgApprox > 0.0) ? 1.0 / avgApprox : 0.0;

    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << qps << "\n";
    out << "tApproximateAverage: " << avgApprox << "\n";
    out << "tTrueAverage: " << avgTrue << "\n";
}
