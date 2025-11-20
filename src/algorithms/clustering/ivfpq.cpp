#include "../../../include/algorithms/clustering/ivfpq.hpp"
#include <iostream>
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>

/**
 * Computes residual vectors r(x) = x - c(x) for all points ref 49
 */
void IVFPQ::makeResiduals(const vector<int> &assign, vector<vector<float>> &residuals)
{
    int N = (int)Data.size(), D = Dim;
    residuals.assign(N, vector<float>(D, 0.0f));
    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        for (int d = 0; d < D; ++d)
            residuals[i][d] = Data[i].values[d] - Centroids[c][d];
    }
}

/**
 * Trains Product Quantizer on residuals ref 49
 * Split residual into M subspaces, run k-means on each with s=2^nbits centroids
 */
void IVFPQ::trainPQ(const vector<vector<float>> &R)
{
    // slide 49Split r(x) into M parts
    subdim_.assign(Args.M, Dim / Args.M);
    for (int i = 0; i < Dim % Args.M; ++i)
        ++subdim_[i]; // Distribute remainder

    codebooks_.assign(Args.M, {});

    int offset = 0;
    for (int m = 0; m < Args.M; ++m)
    {
        int sd = subdim_[m];

        // Extract subspace m from all residuals
        vector<vector<float>> subspace;
        subspace.reserve(R.size());
        for (const auto &r : R)
            subspace.emplace_back(r.begin() + offset, r.begin() + offset + sd);

        // slide 49 Lloyd's creates clustering with s = 2^nbits centroids
        vector<vector<float>> cb;

        kmeansWithPP(subspace, Ks_, (unsigned)Args.seed + m, cb);
        kmeansWithPP(subspace, Kclusters, (unsigned)Args.seed + m, Centroids);
        codebooks_[m] = std::move(cb);

        offset += sd;
    }
}

/**
 * Encodes all residuals using trained PQ ref 49
 */
void IVFPQ::encodeAll(const vector<vector<float>> &R)
{
    int N = (int)R.size();
    codes_.assign(N, vector<uint8_t>(Args.M, 0));

    int offset = 0;
    for (int m = 0; m < Args.M; ++m)
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
void IVFPQ::buildIndex()
{
    int N = (int)Data.size();
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
    vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(trainN);

    vector<vector<float>> Ptrain;
    for (int id : idx)
        Ptrain.push_back(Data[id].values);

    kmeansWithPP(Ptrain, Kclusters, (unsigned)Args.seed, Centroids);

    // slide 49, Assign ALL points to nearest centroid
    Lists.assign(Kclusters, {});
    vector<int> fullAssign(N, -1);
    for (int i = 0; i < N; ++i)
    {
        const auto &x = Data[i].values;
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
    vector<vector<float>> R;
    makeResiduals(fullAssign, R);

    // slide 49, Train product quantizer on residuals
    trainPQ(R);

    // slide 49,  Encode ALL residuals with trained PQ
    // PQ(x) = [code_1(x), ..., code_M(x)]
    encodeAll(R);
}

/**
 * Performs IVFPQ search ref 50
 */
void IVFPQ::search(const vector<VectorData> &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "IVFPQ\n\n";

    const double Rrad = Args.R;
    const bool doRange = Args.rangeSearch;
    const int Nret = std::max(1, Args.N);
    const int rerankTop = 100; // Re-rank top T candidates with exact distances
    int Q = (int)queries.size();

    if (Args.maxQueries > 0)
        Q = std::min(Q, Args.maxQueries);

    for (int qi = 0; qi < Q; ++qi)
    {
        const auto &q = queries[qi].values;

        // === Approximate search using IVFPQ ===
        auto t0 = high_resolution_clock::now();

        // slide 50Score centroids - compute ||q - c_j||_2
        vector<std::pair<double, int>> centroidDists;
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

        vector<std::pair<double, int>> adcCandidates; // (adc_distance_squared, id)
        vector<int> rlist;

        // slide 50 For each selected c_j
        for (int pi = 0; pi < probeCount; ++pi)
        {
            int cid = centroidDists[pi].second;

            // Compute residual: r(q) = q - c_j
            vector<float> rq(Dim);
            for (int d = 0; d < Dim; ++d)
                rq[d] = q[d] - Centroids[cid][d];

            // Split r(q) = [r_1(q), ..., r_M(q)]
            // Define LUT[i][h] = ||r_i(q) - c_{i,h}||_2 ref 50)
            vector<vector<double>> LUT(Args.M, vector<double>(Ks_, 0.0));
            int offset = 0;
            for (int m = 0; m < Args.M; ++m)
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
                for (int m = 0; m < Args.M; ++m)
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
        vector<std::pair<double, int>> reranked;
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
                double exactDist = l2(q, Data[id].values);
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
        calculatePerQueryMetrics(queries[qi].id, qi, tApprox, reranked, rlist, out);
    }
    printSummary(Q, out);
}
