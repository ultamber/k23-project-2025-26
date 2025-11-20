#include "../../../include/algorithms/clustering/ivfflat.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <vector>

/**
 * Builds the IVF index ref 47
 * 1) Run Lloyd's on √n subset X' to get centroids
 * 2) Assign ALL points to nearest centroid
 * 3) Build inverted lists
 */
void IVFFlat::buildIndex()
{
    const int N = (int)Data.size();
    if (N == 0 || Kclusters <= 0)
    {
        Centroids.clear();
        Lists.clear();
        return;
    }

    // slide 47 Run Lloyd's on √n training subset X'
    int trainN = max(Kclusters, (int)sqrt((double)N)); trainN = min(trainN, N); // Don't exceed dataset size
    mt19937_64 rng(Args.seed);
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    idx.resize(trainN);

    vector<vector<float>> Ptrain;
    Ptrain.reserve(trainN);
    for (int id : idx)
        Ptrain.push_back(Data[id].values);

    // slide 47: Learn centroids C = {c₁, ..., cₖ}
    kmeansWithPP(Ptrain, Kclusters, (unsigned)Args.seed, Centroids);

    // slide 47, Assignment phase - assign ALL points (not just training)
    // to their nearest centroid j* and build inverted lists IL_j
    Lists.assign(Kclusters, {});
    for (int i = 0; i < N; ++i)
    {
        const auto &x = Data[i].values;

        // j*(x) = argmin ||x - c_j||₂
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
        // slide 47, Append (id(x), x) to IL_j*(x)
        Lists[best].push_back(i);
    }
}

/**
 * Performs IVF search ref 48
 * 1) Coarse search: compute ||q - c_j||_2 for all centroids, select top b
 * 2) Fine search: compute distances to candidates in U = ⋃_j belongs to S IL_j
 * 3) Return R nearest
 */
void IVFFlat::search(const vector<VectorData> &queries, ofstream &out)
{
    out << "IVFFlat\n\n";

    int Q = (int)queries.size();
    if (Args.maxQueries > 0)
        Q = min(Q, Args.maxQueries);

    for (int qi = 0; qi < Q; ++qi)
    {
        const auto &q = queries[qi].values;

        // === Approximate search using IVF ===
        auto t0 = high_resolution_clock::now();

        // slide 48 Coarse search - evaluate all cells
        // Compute ||q - c_j||_2 for j = 1, ..., k
        vector<pair<double, int>> centroidDists;
        centroidDists.reserve(Kclusters);
        for (int c = 0; c < Kclusters; ++c)
            centroidDists.emplace_back(l2(q, Centroids[c]), c);

        // Select top b = nprobe_ cells (S ⊂ {1,...,k})
        int probeCount = min(Nprobe, (int)centroidDists.size());
        nth_element(centroidDists.begin(),
                        centroidDists.begin() + probeCount,
                        centroidDists.end());
        sort(centroidDists.begin(), centroidDists.begin() + probeCount);

        // slide 48 Collect candidates from U = ⋃_j∈S IL_j
        vector<int> candidates;
        size_t totalSize = 0;
        for (int i = 0; i < probeCount; ++i)
            totalSize += Lists[centroidDists[i].second].size();
        candidates.reserve(totalSize);

        for (int i = 0; i < probeCount; ++i)
        {
            int cid = centroidDists[i].second;
            const auto &IL = Lists[cid];
            candidates.insert(candidates.end(), IL.begin(), IL.end());
        }

        // slide 48 Compute d(q,x) = ||q - x||_2 for all x ∈ U
        vector<pair<double, int>> distApprox;
        distApprox.reserve(candidates.size());
        vector<int> rlist; // Range search results

        for (int id : candidates)
        {
            double d = l2(q, Data[id].values);
            distApprox.emplace_back(d, id);
            if (Args.rangeSearch && d <= Args.R)
                rlist.push_back(id);
        }

        // slide 48, Return R nearest points from U
        int keepApprox = std::min(Args.N, (int)distApprox.size());
        if (keepApprox > 0)
        {
            std::nth_element(distApprox.begin(),
                           distApprox.begin() + keepApprox,
                           distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin() + keepApprox);
            distApprox.resize(keepApprox);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        calculatePerQueryMetrics(queries[qi].id, qi, tApprox, distApprox, rlist, out);
    }
    printSummary(Q, out);
}

/**
 * Overall silhouette score ref 44-45
 */
double IVFFlat::silhouetteScore()
{
    int N = Data.size();
    int k = Centroids.size();
    if (N == 0 || k <= 1)
        return 0.0;

    vector<int> label(N, -1);
    for (int c = 0; c < k; ++c)
        for (int id : Lists[c])
            label[id] = c;

    double total = 0.0;
    int validPoints = 0;

    for (int i = 0; i < N; ++i)
    {
        int ci = label[i];
        if (ci < 0)
            continue;
        const auto &xi = Data[i].values;

        // slide 44: a(i) = average distance to objects in same cluster
        double a_i = 0.0;
        int sameCount = 0;
        for (int id : Lists[ci])
        {
            if (id == i)
                continue;
            a_i += l2(xi, Data[id].values);
            ++sameCount;
        }
        if (sameCount > 0)
            a_i /= sameCount;
        else
            a_i = 0.0;

        // slide 44: b(i) = average distance to next best cluster
        double b_i = std::numeric_limits<double>::infinity();
        for (int c = 0; c < k; ++c)
        {
            if (c == ci || Lists[c].empty())
                continue;
            double sum = 0.0;
            for (int id : Lists[c])
                sum += l2(xi, Data[id].values);
            double avg = sum / Lists[c].size();
            if (avg < b_i)
                b_i = avg;
        }

        // slide 44: s(i) = (b(i) - a(i)) / max{a(i), b(i)}
        double s_i = 0.0;
        if (a_i < b_i)
            s_i = 1.0 - (a_i / b_i);
        else if (a_i > b_i)
            s_i = (b_i / a_i) - 1.0;
        else
            s_i = 0.0;

        total += s_i;
        ++validPoints;
    }

    return (validPoints > 0) ? total / validPoints : 0.0;
}

/**
 * Per-cluster silhouette averages ref 45
 */
vector<double> IVFFlat::silhouettePerCluster()
{
    int N = Data.size();
    int k = Centroids.size();
    if (N == 0 || k <= 1)
        return {};

    vector<int> label(N, -1);
    for (int c = 0; c < k; ++c)
        for (int id : Lists[c])
            label[id] = c;

    vector<double> clusterSum(k, 0.0);
    vector<int> clusterCount(k, 0);

    for (int i = 0; i < N; ++i)
    {
        int ci = label[i];
        if (ci < 0)
            continue;
        const auto &xi = Data[i].values;

        double a_i = 0.0;
        int sameCount = 0;
        for (int id : Lists[ci])
        {
            if (id == i)
                continue;
            a_i += l2(xi, Data[id].values);
            ++sameCount;
        }
        if (sameCount > 0)
            a_i /= sameCount;
        else
            a_i = 0.0;

        double b_i = std::numeric_limits<double>::infinity();
        for (int c = 0; c < k; ++c)
        {
            if (c == ci || Lists[c].empty())
                continue;
            double sum = 0.0;
            for (int id : Lists[c])
                sum += l2(xi, Data[id].values);
            double avg = sum / Lists[c].size();
            if (avg < b_i)
                b_i = avg;
        }

        double s_i = 0.0;
        if (a_i < b_i)
            s_i = 1.0 - (a_i / b_i);
        else if (a_i > b_i)
            s_i = (b_i / a_i) - 1.0;
        else
            s_i = 0.0;

        clusterSum[ci] += s_i;
        ++clusterCount[ci];
    }

    vector<double> clusterAvg(k, 0.0);
    for (int c = 0; c < k; ++c)
        if (clusterCount[c] > 0)
            clusterAvg[c] = clusterSum[c] / clusterCount[c];
        else
            clusterAvg[c] = 0.0;

    return clusterAvg;
}

/**
 * Lloyd's algorithm with k-means++ initialization
 * Improved version of existing kmeans function
 */
void IVFFlat::kmeansWithPP(
    const vector<vector<float>> &P,
    int k,
    unsigned seed,
    vector<vector<float>> &centroids
) {
    int n = P.size();
    if (n == 0 || k <= 0) return;

    centroids.clear();
    centroids.reserve(k);

    mt19937 gen(seed);
    uniform_int_distribution<> dis(0, n - 1);

    // 1. Pick the first centroid randomly
    centroids.push_back(P[dis(gen)]);

    vector<float> dist2(n, numeric_limits<float>::max());

    for (int c = 1; c < k; ++c) {
        // 2. Compute squared distances to the nearest centroid
        for (int i = 0; i < n; ++i) {
            float d = l2(P[i], centroids.back());
            if (d < dist2[i]) dist2[i] = d;
        }

        // 3. Pick next centroid randomly weighted by distance squared
        float sum = 0.0f;
        for (float d : dist2) sum += d;

        uniform_real_distribution<float> u(0.0f, sum);
        float r = u(gen);

        float cumulative = 0.0f;
        int nextIndex = 0;
        for (int i = 0; i < n; ++i) {
            cumulative += dist2[i];
            if (cumulative >= r) {
                nextIndex = i;
                break;
            }
        }

        centroids.push_back(P[nextIndex]);
    }
}
