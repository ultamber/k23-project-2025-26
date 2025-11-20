#include "../../include/algorithms/lsh.hpp"
#include <ostream>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>


/**
 * Builds the LSH index for the input dataset
 * @param data Input dataset to be indexed
 */
void LSH::buildIndex()
{
    // Set default parameters if not provided
    w_ = Args.w > 0 ? Args.w : 4.0f;
    int L = Args.L > 0 ? Args.L : 10;
    int k = Args.k > 0 ? Args.k : 4;

    // TableSize heuristic ref 20): n/4
    tableSize_ = max<size_t>(1, Data.size() / 4);

    // Random number generators ref 18-19)
    mt19937_64 rng(Args.seed);
    normal_distribution<double> normal(0.0, 1.0);                        // For random projections a
    uniform_real_distribution<float> unif(0.0f, w_);                     // For random shifts t
    uniform_int_distribution<uint32_t> distR(1u, (uint32_t)(MOD_M - 1)); // For hash combination r

    // Allocate hash function parameters
    a_.assign(L, vector<vector<float>>(k, vector<float>(Dim))); // Random projection vectors
    t_.assign(L, vector<float>(k, 0.0f));                                  // Random shifts
    r_.assign(L, vector<long long>(k));                                    // Integer coefficients for hash combination

    // Generate random projections, shifts, and integer coefficients ref 18
    for (int li = 0; li < L; ++li)
    {
        for (int j = 0; j < k; ++j)
        {
            // Generate random projection vector from standard normal
            for (int d = 0; d < Dim; ++d)
                a_[li][j][d] = (float)normal(rng);

            // Generate random shift uniformly in [0, w)
            t_[li][j] = unif(rng);

            // Generate random integer coefficient for hash combination ref 20
            r_[li][j] = (long long)distR(rng);
        }
    }

    // Allocate L hash tables with TableSize buckets each
    tables_.assign(L, vector<vector<pair<int, uint64_t>>>(tableSize_));

    // Insert all data points into hash tables
    for (int id = 0; id < (int)Data.size(); ++id)
    {
        const auto &p = Data[id].values;
        for (int li = 0; li < L; ++li)
        {
            // Compute ID(p) using hash function 
            // ref slides 18, 20-21
            uint64_t IDp = computeID(p, li);

            // Compute bucket index g(p) = ID(p) mod TableSize 
            // ref slide 20
            uint64_t g = IDp % tableSize_;

            // Store both point id and its ID for filtering 
            // ref slide 21
            tables_[li][g].push_back({id, IDp});
        }
    }

    if (getenv("LSH_DEBUG"))
    {
        cerr << "[LSH-DIAG] w=" << w_
                  << " L=" << L << " k=" << k
                  << " TableSize=" << tableSize_
                  << " min_h_seen=" << (min_h_seen_ == LLONG_MAX ? 0 : min_h_seen_)
                  << " neg_h_count=" << neg_h_count_
                  << "\n";
    }
}

/**
 * Computes the ID for a vector using LSH hash functions (slides 18, 20-21)
 * @param v Input vector
 * @param li Index of the hash table
 * @return 64-bit ID value
 */
uint64_t LSH::computeID(const vector<float> &v, int li) const
{
    uint64_t ID = 0;

    // Combine k hash functions 
    // ref slide 20
    for (int j = 0; j < Args.k; ++j)
    {
        // Compute dot product for projection 
        // ref slide 18
        double dot = 0.0;
        for (int d = 0; d < Dim; ++d)
            dot += a_[li][j][d] * v[d];

        // Apply LSH hash function: h(p) = floor((a·p + t)/w)
        // slide 18
        long long hj = (long long)floor((dot + t_[li][j]) / w_);

        // Combine using random coefficients: ID = Σ r_ih_i(p) mod M 
        // slide 20
        ID = (ID + (uint64_t)(r_[li][j] * hj)) % MOD_M;

        // Update diagnostic counters
        if (hj < min_h_seen_)
            min_h_seen_ = hj;
        if (hj < 0)
            ++neg_h_count_;
    }

    return ID;
}

/**
 * Computes the hash key for a vector in a specific hash table
 * @param v Input vector to be hashed
 * @param li Index of the hash table
 * @return 64-bit hash key (bucket index)
 */
uint64_t LSH::keyFor(const vector<float> &v, int li) const
{
    uint64_t IDv = computeID(v, li);
    return IDv % tableSize_; // g(p) = ID(p) mod TableSize ref 20
}

/**
 * Performs LSH search for all queries
 * @param queries Query dataset
 * @param out Output file stream for results
 */
void LSH::search(const vector<VectorData> &queries, ofstream &out)
{

    using namespace chrono;
    out << "LSH\n\n";

    int qCount = (int)queries.size();

    // Optional query limit
    if (Args.maxQueries > 0)
        qCount = min(qCount, Args.maxQueries);

    for (int qi = 0; qi < qCount; ++qi)
    {
        const auto &q = queries[qi].values;
        auto t0 = high_resolution_clock::now();

        // multi probe lsh: collect candidates from multiple buckets
        vector<int> candidates;
        size_t examined = 0;

        // Hard cap on candidates to examine 
        // ref slides 13-14: stop after ~10L to 20L items
        size_t hardCap = Args.rangeSearch ? 20 * Args.L : 10 * Args.L;

        // Probe all L tables
        for (int li = 0; li < Args.L; ++li)
        {
            // compute query's hash value
            //  ref slide 20-21
            uint64_t IDq = computeID(q, li);
            uint64_t gq = IDq % tableSize_;

            // multi-probes: check main bucket + neighboring buckets (delta = -2 to +2)
            // significantly improves recall by finding near-collisions
            for (int delta = -2; delta <= 2; ++delta)
            {
                uint64_t gq2 = (gq + delta + tableSize_) % tableSize_;
                const auto &bucket = tables_[li][gq2];

                for (const auto &pr : bucket)
                {
                    if (delta == 0)
                    {
                        // Exact bucket: use ID filtering 
                        // ref slide 21 
                        if (pr.second == IDq)
                            candidates.push_back(pr.first);
                    }
                    else
                    {
                        // Neighboring buckets: accept all candidates 
                        candidates.push_back(pr.first);
                    }

                    if (++examined > hardCap)
                        break;
                }
                if (examined > hardCap)
                    break;
            }
            if (examined > hardCap)
                break;
        }

        // Deduplicate candidates 
        sort(candidates.begin(), candidates.end());
        candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());

        // Compute actual distances to candidates
        vector<pair<double, int>> distApprox;
        vector<int> rlist; // Range search results
        distApprox.reserve(candidates.size());

        for (int id : candidates)
        {
            double d = l2(q, Data[id].values);
            distApprox.emplace_back(d, id);

            // Range search: collect points within radius R 
            // ref slide 14
            if (Args.rangeSearch && d <= Args.R)
                rlist.push_back(id);
        }

        // Find top N nearest neighbors 
        // ref slide 13
        int N = min(Args.N, (int)distApprox.size());
        if (N > 0)
        {
            nth_element(distApprox.begin(), distApprox.begin() + N, distApprox.end());
            sort(distApprox.begin(), distApprox.begin() + N);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();

        calculatePerQueryMetrics(queries[qi].id, qi, tApprox, distApprox, rlist, out);
    }

    printSummary(qCount, out);
};
