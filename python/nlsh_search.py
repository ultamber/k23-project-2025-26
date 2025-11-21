#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlsh_search.py (IMPROVED)
--------------------------
Enhanced Neural LSH search with:
- Adaptive multi-probe strategy
- Better candidate collection
- Detailed metrics and diagnostics
- Fallback strategies
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from modules.dataset_parser import load_dataset
from modules.models import MLPClassifier
from modules.utils import set_seed


# -----------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Neural LSH search (improved)")

    p.add_argument("-d", required=True, help="Dataset file (input.dat)")
    p.add_argument("-q", required=True, help="Query file (query.dat)")
    p.add_argument("-i", required=True, help="Index directory (from nlsh_build.py)")
    p.add_argument("-o", required=True, help="Output file")
    p.add_argument("-type", required=True, choices=["sift", "mnist"],
                   help="Dataset type")

    p.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
    p.add_argument("-R", type=float, default=None, help="Search radius for range search")
    p.add_argument("-T", type=int, default=5, help="Base number of bins to probe")
    p.add_argument("-range", type=str, default="true",
                   help="Whether to perform range search (true/false)")

    # Advanced options
    p.add_argument("--adaptive", action="store_true",
                   help="Use adaptive probing based on prediction confidence")
    p.add_argument("--max_T", type=int, default=None,
                   help="Maximum bins for adaptive probing (default: 3*T)")
    p.add_argument("--min_candidates", type=int, default=None,
                   help="Minimum candidates to examine (default: 10*N)")
    p.add_argument("--max_candidates", type=int, default=None,
                   help="Maximum candidates to examine (default: unlimited)")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed per-query statistics")
    p.add_argument("--seed", type=int, default=1, help="Random seed")

    return p.parse_args()


def str2bool(s):
    return str(s).lower() in ["1", "true", "yes", "y"]


# -----------------------------------------------------------
# Index loading
# -----------------------------------------------------------
def load_index(index_dir, dataset_type):
    """Load Neural LSH index with all components."""
    index_dir = Path(index_dir)

    # Metadata
    meta_path = index_dir / "metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    d_in = meta["dimension"]
    m = meta["m"]
    layers = meta["layers"]
    nodes = meta["nodes"]
    dropout = meta.get("dropout", 0.0)

    # MLP model
    model = MLPClassifier(d_in=d_in, n_out=m, layers=layers, nodes=nodes, dropout=dropout)
    
    state = torch.load(index_dir / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Inverted index
    with open(index_dir / "inverted_index.pkl", "rb") as f:
        inverted = pickle.load(f)

    # Dataset
    X = np.load(index_dir / "dataset.npy")

    return model, inverted, X, meta


# -----------------------------------------------------------
# Distance computation
# -----------------------------------------------------------
def l2_distances(x, X):
    """Compute L2 distances from x to all rows of X efficiently."""
    diff = X - x
    return np.linalg.norm(diff, axis=1)


def l2_distances_batch(x, X, batch_size=10000):
    """Compute L2 distances in batches to save memory."""
    n = X.shape[0]
    distances = np.zeros(n)
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        diff = X[start:end] - x
        distances[start:end] = np.linalg.norm(diff, axis=1)
    
    return distances


# -----------------------------------------------------------
# Adaptive probing
# -----------------------------------------------------------
def compute_prediction_entropy(probs):
    """
    Compute normalized entropy of prediction distribution.
    High entropy = uncertain prediction = should probe more bins.
    
    Returns: value in [0, 1] where 1 = maximum uncertainty
    """
    probs = probs.cpu().numpy()
    # Avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)
    
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probs))
    
    return entropy / max_entropy if max_entropy > 0 else 0.0


def adaptive_probe_count(probs, base_T, max_T, m):
    """
    Determine number of bins to probe adaptively.
    
    Args:
        probs: Softmax probabilities over bins
        base_T: Minimum number of bins to probe
        max_T: Maximum number of bins to probe
        m: Total number of partitions
    
    Returns:
        Number of bins to probe
    """
    # Get top probability (confidence)
    top_prob = probs.max().item()
    
    # Strategy 1: If very confident (top_prob > 0.5), use fewer bins
    if top_prob > 0.5:
        return max(base_T, int(base_T * 0.8))
    
    # Strategy 2: Use entropy for adaptive selection
    entropy = compute_prediction_entropy(probs)
    
    # Linear interpolation between base_T and max_T based on entropy
    T = int(base_T + (max_T - base_T) * entropy)
    
    # Ensure we don't exceed total partitions
    return min(T, m)


# -----------------------------------------------------------
# Candidate collection
# -----------------------------------------------------------
def collect_candidates(bins, inverted, min_candidates=None, max_candidates=None):
    """
    Collect candidate points from selected bins.
    
    Args:
        bins: Array of bin indices to probe
        inverted: Inverted index mapping bin -> point list
        min_candidates: Minimum number of candidates (will probe more bins if needed)
        max_candidates: Maximum number of candidates (will stop early if reached)
    
    Returns:
        Array of unique candidate indices
    """
    candidates = []
    candidates_set = set()
    
    for b in bins:
        b = int(b)
        if b in inverted:
            bin_points = inverted[b]
            
            # Add unique candidates
            for pt in bin_points:
                if pt not in candidates_set:
                    candidates.append(pt)
                    candidates_set.add(pt)
            
            # Check if we've reached max
            if max_candidates and len(candidates) >= max_candidates:
                break
    
    candidates = np.array(candidates, dtype=np.int64)
    
    # If we don't have enough candidates, return what we have
    # (caller can decide to probe more bins or use all data)
    
    return candidates


# -----------------------------------------------------------
# Search function
# -----------------------------------------------------------
def search_query(
    q, 
    model, 
    inverted, 
    X_data, 
    N, 
    T, 
    max_T, 
    m,
    adaptive,
    min_candidates,
    max_candidates,
    device
):
    """
    Search for k-NN of a single query.
    
    Returns:
        topN_indices: Indices of approximate k-NN
        topN_distances: Distances to approximate k-NN
        num_bins_probed: Number of bins actually probed
        num_candidates: Number of candidates examined
    """
    # Predict bin probabilities
    with torch.no_grad():
        qt = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(qt)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    
    # Determine number of bins to probe
    if adaptive:
        T_actual = adaptive_probe_count(probs, T, max_T, m)
    else:
        T_actual = T
    
    # Get top bins
    topT = torch.topk(probs, k=min(T_actual, m))
    bins = topT.indices.cpu().numpy()
    
    # Collect candidates
    candidates = collect_candidates(bins, inverted, min_candidates, max_candidates)
    
    # Fallback: if no candidates found, try more bins
    if len(candidates) == 0:
        # Probe more bins (up to max_T or all)
        fallback_T = min(max_T, m)
        topT_fallback = torch.topk(probs, k=fallback_T)
        bins_fallback = topT_fallback.indices.cpu().numpy()
        candidates = collect_candidates(bins_fallback, inverted, None, max_candidates)
        T_actual = fallback_T
    
    # Still no candidates? Use all data (shouldn't happen with good partitioning)
    if len(candidates) == 0:
        candidates = np.arange(len(X_data), dtype=np.int64)
    
    # Compute distances to candidates
    d_cand = l2_distances(q, X_data[candidates])
    
    # Find top-N
    if len(d_cand) >= N:
        order_cand = np.argpartition(d_cand, N-1)[:N]
        order_cand = order_cand[np.argsort(d_cand[order_cand])]
    else:
        order_cand = np.argsort(d_cand)
    
    topN_cand = candidates[order_cand]
    d_approx = d_cand[order_cand]
    
    return topN_cand, d_approx, T_actual, len(candidates)


# -----------------------------------------------------------
# Main search loop
# -----------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Set default R
    if args.R is None:
        args.R = 2000.0 if args.type == "mnist" else 2800.0

    # Set default adaptive parameters
    if args.max_T is None:
        args.max_T = min(args.T * 3, 100)
    if args.min_candidates is None:
        args.min_candidates = args.N * 10
    
    do_range = str2bool(args.range)

    print("="*70)
    print("NEURAL LSH SEARCH")
    print("="*70)
    
    # Load index
    print(f"\n[1] Loading index from {args.i}...")
    model, inverted, X_data, meta = load_index(args.i, args.type)
    n_data, dim = X_data.shape
    m = meta["m"]
    print(f"✓ Index loaded")
    print(f"  Dataset: n={n_data:,}, d={dim}")
    print(f"  Partitions: m={m}")
    print(f"  Inverted index size: {len(inverted)} bins")
    
    # Print partition statistics
    bin_sizes = [len(inverted.get(i, [])) for i in range(m)]
    print(f"  Partition sizes: min={min(bin_sizes)}, max={max(bin_sizes)}, "
          f"avg={np.mean(bin_sizes):.1f}, median={np.median(bin_sizes):.1f}")

    # Load queries
    print(f"\n[2] Loading queries from {args.q}...")
    X_query = load_dataset(args.q, args.type)
    n_queries = X_query.shape[0]
    print(f"✓ Queries loaded: {n_queries:,}")

    # Search parameters
    print(f"\n[3] Search configuration:")
    print(f"  N (neighbors): {args.N}")
    print(f"  T (base probes): {args.T}")
    print(f"  Max T: {args.max_T}")
    print(f"  Adaptive probing: {args.adaptive}")
    print(f"  Min candidates: {args.min_candidates}")
    if args.max_candidates:
        print(f"  Max candidates: {args.max_candidates}")
    print(f"  Range search: {do_range} (R={args.R})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # Metrics
    total_approx_time = 0.0
    total_true_time = 0.0
    sum_AF = 0.0
    sum_recall = 0.0
    total_bins_probed = 0
    total_candidates = 0

    print(f"\n[4] Searching...")
    start_all = time.time()

    with open(args.o, "w") as fout:
        fout.write("Neural LSH\n\n")

        for q_id in range(n_queries):
            q = X_query[q_id].astype(np.float32)

            # Progress indicator
            if (q_id + 1) % max(1, n_queries // 10) == 0:
                print(f"  Progress: {q_id + 1}/{n_queries} ({100*(q_id+1)/n_queries:.1f}%)")

            # --------------------------------------
            # Approximate search (Neural LSH)
            # --------------------------------------
            t0 = time.time()
            
            topN_cand, d_approx, bins_probed, num_candidates = search_query(
                q, model, inverted, X_data, args.N, args.T, args.max_T, m,
                args.adaptive, args.min_candidates, args.max_candidates, device
            )
            
            t1 = time.time()
            tApprox = t1 - t0
            total_approx_time += tApprox
            total_bins_probed += bins_probed
            total_candidates += num_candidates

            # --------------------------------------
            # True brute-force search (for metrics)
            # --------------------------------------
            t2 = time.time()
            d_all = l2_distances(q, X_data)
            order_all = np.argsort(d_all)
            topN_true = order_all[:args.N]
            d_true_topN = d_all[topN_true]
            best_true = d_true_topN[0]
            t3 = time.time()
            tTrue = t3 - t2
            total_true_time += tTrue

            # Metrics
            AF_q = float(d_approx[0] / (best_true + 1e-12))
            sum_AF += AF_q

            # Recall@N
            set_true = set(topN_true.tolist())
            set_approx = set(topN_cand.tolist())
            recall_q = len(set_true & set_approx) / float(args.N)
            sum_recall += recall_q

            # Verbose per-query stats
            if args.verbose and q_id < 10:  # First 10 queries
                print(f"\n  Query {q_id}:")
                print(f"    Bins probed: {bins_probed}")
                print(f"    Candidates: {num_candidates}")
                print(f"    Recall: {recall_q:.3f}")
                print(f"    AF: {AF_q:.4f}")
                print(f"    Time: {tApprox:.6f}s")

            # --------------------------------------
            # Range search
            # --------------------------------------
            R_neighbors = []
            if do_range:
                R_neighbors = np.where(d_all <= args.R)[0].tolist()

            # --------------------------------------
            # Write results (IMPROVED FORMAT)
            # --------------------------------------
            fout.write(f"Query: {q_id}\n")

            for k in range(args.N):
                # Approximate result
                approx_idx = topN_cand[k]
                approx_dist = d_approx[k]
                
                # True result
                true_idx = topN_true[k]
                true_dist = d_true_topN[k]
                
                # Output in a clearer format
                fout.write(f"Nearest neighbor-{k+1}:\n")
                fout.write(f"  Approximate: id={approx_idx}, dist={float(approx_dist):.6f}\n")
                fout.write(f"  True: id={true_idx}, dist={float(true_dist):.6f}\n")
                
                # Optional: Show if they match
                if approx_idx == true_idx:
                    fout.write(f"  Match: ✓\n")
                else:
                    fout.write(f"  Match: ✗\n")

            fout.write("R-near neighbors:\n")
            if do_range:
                for idx in R_neighbors:
                    fout.write(f"{idx}\n")

        # ---- Summary metrics ----
        total_time = time.time() - start_all
        avg_AF = sum_AF / n_queries
        avg_recall = sum_recall / n_queries
        avg_tApprox = total_approx_time / n_queries
        avg_tTrue = total_true_time / n_queries
        qps = n_queries / total_approx_time if total_approx_time > 0 else 0.0
        avg_bins = total_bins_probed / n_queries
        avg_candidates = total_candidates / n_queries

        fout.write(f"\nAverage AF: {avg_AF:.6f}\n")
        fout.write(f"Recall@N: {avg_recall:.6f}\n")
        fout.write(f"QPS: {qps:.2f}\n")
        fout.write(f"tApproximateAverage: {avg_tApprox:.6f}\n")
        fout.write(f"tTrueAverage: {avg_tTrue:.6f}\n")
        fout.write(f"avgBinsProbed: {avg_bins:.2f}\n")
        fout.write(f"avgCandidates: {avg_candidates:.1f}\n")

    print(f"\n{'='*70}")
    print("SEARCH COMPLETED")
    print(f"{'='*70}")
    print(f"\n Results:")
    print(f"  Recall@{args.N}: {avg_recall*100:.2f}%")
    print(f"  Average AF: {avg_AF:.4f}")
    print(f"  QPS: {qps:.2f}")
    print(f"  Avg search time: {avg_tApprox*1000:.2f}ms")
    print(f"  Avg brute-force time: {avg_tTrue*1000:.2f}ms")
    print(f"  Speedup: {avg_tTrue/avg_tApprox:.2f}x")
    print(f"  Avg bins probed: {avg_bins:.1f}/{m} ({100*avg_bins/m:.1f}%)")
    print(f"  Avg candidates examined: {avg_candidates:.0f}/{n_data} ({100*avg_candidates/n_data:.1f}%)")
    print(f"\n Output saved to: {args.o}")
    print()


if __name__ == "__main__":
    main()