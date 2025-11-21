#include "../../include/algorithms/hypercube.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <vector>

/**
 * Builds the Hypercube index for the input dataset ref 24
 * @param data Input dataset to be indexed
 */
void Hypercube::buildIndex()
{
    // slide 24: d' = floor(log_2 n) - {1,2,3}
    const size_t n = 
        Data.size();
    int dlog = (n > 0) ? (int)floor(log2((double)max<size_t>(1, n))) : 1;
    kproj_ = (Args.kproj > 0) ? Args.kproj : max(1, dlog - 2);
    Dim = Data[0].values.size();

    // slide 18: w ∈ [2, 6], larger for range queries
    w_ = (Args.w > 0) ? Args.w : 4.0f;

    // Random number generators
    mt19937_64 rng(Args.seed);
    normal_distribution<double> normal(0.0, 1.0);    // For LSH projections
    uniform_real_distribution<float> unif(0.0f, w_); // For LSH shifts
    uniform_int_distribution<uint32_t> uni32(1u, 0xffffffffu); // For f_j hash functions

    // slide 24: Generate d' base LSH functions h_j
    // Each h_j(p) = floor((a_j · p + t_j) / )
    proj_.assign(kproj_, vector<float>(Dim));
    shift_.assign(kproj_, 0.0f);
    for (int j = 0; j < kproj_; ++j)
    {
        for (int d = 0; d < Dim; ++d)
            proj_[j][d] = (float)normal(rng);
        shift_[j] = unif(rng);
    }

    // slide 24: Generate bit-mapping functions f_j: Z → {0,1}
    // f_j maps buckets to bits uniformly at random
    // Implementation: use random hash function f_j(h) = (aₕ · h + bₕ) mod 2
    fA_.resize(kproj_);
    fB_.resize(kproj_);
    for (int j = 0; j < kproj_; ++j)
    {
        fA_[j] = (uint64_t)uni32(rng);
        fB_[j] = (uint64_t)uni32(rng);
        if (fA_[j] == 0)
            fA_[j] = 1; // Ensure non-zero
    }

    // slide 25: Choose storage strategy based on d'
    // Dense array for small d' (≤ 24 bits = 16M vertices)
    // Sparse map for large d'
    denseCube_ = (kproj_ <= 24);
    if (denseCube_)
    {
        cubeDense_.assign(1ULL << kproj_, {});
        cubeSparse_.clear();
    }
    else
    {
        cubeSparse_.clear();
        cubeDense_.clear();
    }

    // slide 24: Insert all data points into their vertices
    for (int id = 0; id < (int)n; ++id)
    {
        const auto &v = Data[id].values;
        uint64_t vtx = vertexOf(v);
        if (denseCube_)
            cubeDense_[vtx].push_back(id);
        else
            cubeSparse_[vtx].push_back(id);
    }

    if (getenv("CUBE_DEBUG"))
    {
        cerr << "[CUBE] n=" << n
                  << " dim=" << Dim
                  << " d'=" << kproj_
                  << " w=" << w_
                  << " dense=" << denseCube_
                  << " vertices_allocated=" << (denseCube_ ? (1ULL << kproj_) : cubeSparse_.size())
                  << "\n";
    }
}

/**
 * Computes h_j(v) = floor((a_j · v + t_j) / w) for projection j ref 18
 * @param v Input vector
 * @param j Projection index
 * @return Hash value
 */
long long Hypercube::hij(const vector<float> &v, int j) const
{
    double dot = 0.0;
    for (int d = 0; d < Dim; ++d)
        dot += proj_[j][d] * v[d];
    return (long long)floor((dot + shift_[j]) / w_);
}

/**
 * Maps hash value to bit using function f_j ref 24
 * @param h Hash value
 * @param j Projection index
 * @return Bit value (0 or 1)
 */
bool Hypercube::fj(long long h, int j) const
{
    // Random hash function: (a·h + b) mod 2
    uint64_t hval = (uint64_t)(h >= 0 ? h : -h);
    return ((fA_[j] * hval + fB_[j]) & 1) != 0;
}

/**
 * Maps a vector to a vertex in the hypercube ref 24
 * Vertex label is d'-bit string where bit j = f_j(h_j(v))
 * @param v Input vector
 * @return 64-bit vertex label
 */
uint64_t Hypercube::vertexOf(const vector<float> &v) const
{
    uint64_t key = 0;
    for (int j = 0; j < kproj_; ++j)
    {
        long long hj = hij(v, j);    // Compute LSH hash
        if (fj(hj, j))               // Map to bit using f_j
            key |= (1ULL << j);
    }
    return key;
}

/**
 * Generates vertices to probe in order of increasing Hamming distance ref 24
 * @param base Starting vertex (query's vertex)
 * @param kproj Number of bits (d')
 * @param maxProbes Maximum number of vertices to return
 * @param maxHamming Maximum Hamming distance to probe ref 25
 * @return Vector of vertex labels to probe
 */
vector<uint64_t>
Hypercube::probesList(uint64_t base, int kproj, int maxProbes, int maxHamming) const
{
    vector<uint64_t> out{base};
    if ((int)out.size() >= maxProbes)
        return out;

    // slide 25: Threshold on Hamming distance
    const int Hmax = min(kproj, maxHamming);
    
    // Generate vertices at increasing Hamming distances: 1, 2, ...
    for (int h = 1; h <= Hmax && (int)out.size() < maxProbes; ++h)
    {
        // Generate all combinations of h bit flips
        vector<int> idx(h);
        iota(idx.begin(), idx.end(), 0);
        
        while (true)
        {
            // Flip bits at positions in idx
            uint64_t mask = 0;
            for (int i : idx)
                mask |= (1ULL << i);
            out.push_back(base ^ mask);
            
            if ((int)out.size() >= maxProbes)
                break;
            
            // Next combination in lexicographic order
            int i;
            for (i = h - 1; i >= 0 && idx[i] == i + kproj - h; --i)
                ;
            if (i < 0)
                break;
            ++idx[i];
            for (int j = i + 1; j < h; ++j)
                idx[j] = idx[j - 1] + 1;
        }
    }
    return out;
}

/**
 * Performs Hypercube search for all queries 
 * @param queries Query dataset
 * @param out Output file stream for results
 */
void Hypercube::search(const vector<VectorData> &queries, ofstream &out)
{
    using namespace chrono;
    out << "Hypercube\n\n";

    // slide 25: Search parameters
    int M = Args.M;           // Threshold: max candidates to check in R^d
    int probes = Args.probes; // Threshold: max vertices to probe
    int maxHam = (Args.maxHamming > 0) ? Args.maxHamming : kproj_; // Hamming distance bound
    double R = Args.R;
    bool doRange = Args.rangeSearch;
    int N = Args.N;

    int Q = (int)queries.size();
    
    if (Args.maxQueries > 0)
        Q = min(Q, static_cast<int>(Args.maxQueries));

    for (int qi = 0; qi < Q; ++qi)
    {
        const auto &q = queries[qi].values;
        auto t0 = high_resolution_clock::now();

        // slide 24 Project query to hypercube vertex
        uint64_t base = vertexOf(q);
        
        // slide 24 Check points in same and nearby vertices
        // Generate vertices in increasing Hamming distance
        auto plist = probesList(base, kproj_, probes, maxHam);
        
        // Collect unique candidates from probed vertices
        unordered_set<int> candSet;
        candSet.reserve(min<size_t>((size_t)M, (size_t)4096));
        size_t gathered = 0;
        
        for (auto vtx : plist)
        {
            const vector<int> *bucket = nullptr;
            if (denseCube_)
            {
                bucket = &cubeDense_[vtx];
            }
            else
            {
                auto it = cubeSparse_.find(vtx);
                if (it == cubeSparse_.end())
                    continue;
                bucket = &it->second;
            }
            
            // Add points from this vertex
            for (int id : *bucket)
            {
                if (candSet.insert(id).second && ++gathered >= (size_t)M)
                    break;
            }
            if (gathered >= (size_t)M)
                break;
        }

        // Compute actual distances to candidates
        vector<pair<double, int>> distApprox;
        distApprox.reserve(candSet.size());
        vector<int> rlist;

        for (int id : candSet)
        {
            double d = l2(q, 
                          Data[id].values);
            distApprox.emplace_back(d, id);
            if (doRange && d <= R)
                rlist.push_back(id);
        }

        // slide 24 Return closest candidates or range search results
        int topN = min(N, (int)distApprox.size());
        if (topN > 0)
        {
            nth_element(distApprox.begin(), distApprox.begin() + topN, distApprox.end());
            sort(distApprox.begin(), distApprox.begin() + topN);
            distApprox.resize(topN);
        }
        
        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        calculatePerQueryMetrics(queries[qi].id, qi, tApprox, distApprox, rlist, out);
    }
    printSummary(Q, out);
}
