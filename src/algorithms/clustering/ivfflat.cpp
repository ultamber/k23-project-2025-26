#include "../../../include/algorithms/clustering/ivfflat.hpp"
#include <iostream>
#include <random>
#include <limits>
#include <algorithm>
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
    Dim = Data[0].values.size();

    // slide 47 Run Lloyd's on √n training subset X'
    int trainN = min(max(Kclusters, (int)sqrt((double)N)), N);
    vector<int> idx(trainN);

    vector<vector<float>> Ptrain;
    Ptrain.reserve(trainN);
    for (int id : idx)
        Ptrain.push_back(Data[id].values);
    kmeansWithPP(Ptrain, Kclusters, (unsigned)Args.seed, Centroids);

    // slide 47, Assignment phase - assign ALL points (not just training)
    // to their nearest centroid j* and build inverted lists IL_j
    Lists.assign(Kclusters, {});
    for (int i = 0; i < N; ++i)
    {
        const auto &x = Data[i].values;

        // j*(x) = argmin ||x - c_j||₂
        int best = 0;
        double bd = numeric_limits<double>::infinity();
        for (int c = 0; c < Kclusters; ++c)
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
    SilhouetteScore = calculateSilhouetteScore();
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
        int keepApprox = min(Args.N, (int)distApprox.size());
        if (keepApprox > 0)
        {
            nth_element(distApprox.begin(),
                           distApprox.begin() + keepApprox,
                           distApprox.end());
            sort(distApprox.begin(), distApprox.begin() + keepApprox);
            distApprox.resize(keepApprox);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        calculatePerQueryMetrics(queries[qi].id, qi, tApprox, distApprox, rlist, out);
    }
    printSummary(Q, out);
    out << "Silhouette Score: " << SilhouetteScore << endl; 
}

/**
 * Silhouette averages ref 45
 */
double IVFFlat::calculateSilhouetteScore()
{
    int N = Data.size();
    int k = Centroids.size();
    double clusterSum = 0.0;
    int clusterCount = 0;
    vector<int> calculated_silhouettes;
    if (N == 0 || k <= 1)
        return {};

    vector<int> point_id_to_centroid(N, -1);
    for (int c = 0; c < k; ++c) {
        if (Lists[c].empty())
            continue;
        for (int id : Lists[c])
            point_id_to_centroid[id] = c;
    }


    for (int i = 0; i < N; ++i) {
        int ci = point_id_to_centroid[i];

        bool already_calculated = false;
        for (int l : calculated_silhouettes){
            if (l == ci){
                already_calculated = true;
                break;
            }
        }

        if (ci < 0 || already_calculated)
            continue;

        const auto &xi = Data[i].values;

        double a_i = 0.0;
        double b_i = 0.0;
        int count = 0;

        for (int id : Lists[ci]) {
            if (id == i)
                continue;
            a_i += l2(xi, Data[id].values);
            ++count;
        }
        if (count > 0)
            a_i /= count;
        else
            a_i = 0.0;

        double closest_neighbor_dist = numeric_limits<double>::infinity();
        int closest_neighbor_centroid = -1;
        for (int c = 0; c < k; c++) {
            if (c == ci || Lists[c].empty()) 
                continue;
            double distance = l2(xi, Data[c].values);
            if (distance < closest_neighbor_dist) {
                closest_neighbor_centroid = c;
                closest_neighbor_dist = distance;
            }
        }

        if (closest_neighbor_centroid != -1) {
            count = 0;
            for (int id : Lists[closest_neighbor_centroid]) {
                if (id == closest_neighbor_centroid)
                    continue;
                b_i += l2(xi, Data[id].values);
                count++;
            }

            if (count > 0)
                b_i /= count;
            else
                b_i = 0;
        }

        double s_i = 0.0;
        if (a_i <= 0.0 && b_i <= 0.0)
            continue;
        s_i = (b_i - a_i) / max(a_i, b_i);

        clusterSum += s_i;
        clusterCount ++;

    }

    return clusterSum / clusterCount;
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
    centroids.reserve(k);

    mt19937 gen(seed);

    // (1) Choose the first centroid uniformly at random
    uniform_int_distribution<int> uni_dist(0, n - 1);

    int first = uni_dist(gen);
    centroids.emplace_back(P[first]);


    vector<double> D(n, 0.0);

    // (2)-(4) Choose the next k-1 centroids
    for (int t = 1; t < k; ++t) {
        double sumD = 0.0;

        // Compute D(i) = distance to nearest chosen centroid
        for (int i = 0; i < n; i++) {
            double d = l2(P[i], centroids[0]);
            for (int c = 0; c < t; c++) {
                d = min(d, l2(P[i], centroids[c]));
            }
            D[i] = d;
            sumD += d;
        }

        // Choose next centroid with probability proportional to D(i)
        uniform_real_distribution<double> dist_double(0.0, sumD);
        double rnd = dist_double(gen);

        double cumulative = 0.0;
        int next = 0;
        for (int i = 0; i < n; i++) {
            cumulative += D[i];
            if (cumulative >= rnd) {
                next = i;
                break;
            }
        }

        centroids.emplace_back(P[next]);
    }
}
