#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlsh_search.py
--------------
Neural LSH search 
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


def parse_args():
    p = argparse.ArgumentParser(description="Neural LSH search (Assignment Version)")

    p.add_argument("-d", required=True, help="Dataset file")
    p.add_argument("-q", required=True, help="Query file")
    p.add_argument("-i", required=True, help="Index directory")
    p.add_argument("-o", required=True, help="Output file")
    p.add_argument("-type", required=True, choices=["sift", "mnist"], help="Dataset type")

    p.add_argument("-N", type=int, default=1, help="Number of neighbors")
    p.add_argument("-R", type=float, default=None, help="Search radius")
    p.add_argument("-T", type=int, default=5, help="Number of bins to probe")
    p.add_argument("-range", type=str, default="true", help="Range search (true/false)")
    
    p.add_argument("--seed", type=int, default=1, help="Random seed")

    return p.parse_args()


def str2bool(s):
    return str(s).lower() in ["1", "true", "yes", "y"]


def load_index(index_dir):
    """Load Neural LSH index."""
    index_dir = Path(index_dir)

    with open(index_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    d_in = meta["dimension"]
    m = meta["m"]
    layers = meta["layers"]
    nodes = meta["nodes"]
    dropout = meta.get("dropout", 0.0)

    model = MLPClassifier(d_in=d_in, n_out=m, layers=layers, nodes=nodes, dropout=dropout)
    state = torch.load(index_dir / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    with open(index_dir / "inverted_index.pkl", "rb") as f:
        inverted = pickle.load(f)

    X = np.load(index_dir / "dataset.npy")

    return model, inverted, X, meta


def l2_distance(q, x):
    """Compute L2 distance between two vectors."""
    return np.linalg.norm(q - x)

def brute_force_search(q, X_data, N):
    """Vectorized brute-force search."""
    distances = np.linalg.norm(X_data - q, axis=1)
    top_indices = np.argpartition(distances, N-1)[:N]
    top_indices = top_indices[np.argsort(distances[top_indices])]
    top_distances = distances[top_indices]
    return top_indices, top_distances

def neural_lsh_search(q, model, inverted, X_data, T, N, device):
    """
    Neural LSH approximate search.
    Returns indices and distances of N nearest neighbors.
    """
    # Step 1: Predict probabilities
    qt = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(qt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Step 2: Select top-T bins
    top_bins = np.argsort(-probs)[:T]
    
    # Step 3: Collect candidates
    candidates = []
    for b in top_bins:
        b = int(b)
        if b in inverted:
            candidates.extend(inverted[b])
    
    if len(candidates) == 0:
    # Return random sample or empty
        candidates = np.random.choice(len(X_data), min(N*10, len(X_data)), replace=False).tolist()
    
    candidates = list(set(candidates))  # Remove duplicates
    
    # Step 4: Compute distances to candidates
    candidate_vectors = X_data[candidates]
    distances = np.linalg.norm(candidate_vectors - q, axis=1)
    
    # Step 5: Find top-N
    if len(distances) >= N:
        top_idx = np.argpartition(distances, N-1)[:N]
        top_idx = top_idx[np.argsort(distances[top_idx])]
    else:
        top_idx = np.argsort(distances)
    
    result_indices = [candidates[i] for i in top_idx[:N]]
    result_distances = distances[top_idx[:N]]
    
    return result_indices, result_distances


def compute_af(approx_dist, true_dist):
    """Approximation Factor"""
    if true_dist < 1e-12:
        return 1.0
    return approx_dist / true_dist


def compute_recall(approx_neighbors, true_neighbors):
    """Recall@N"""
    set_approx = set(approx_neighbors)
    set_true = set(true_neighbors)
    return len(set_approx & set_true) / len(set_true)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.R is None:
        args.R = 2000.0 if args.type == "mnist" else 2800.0

    do_range = str2bool(args.range)

    print("="*70)
    print("NEURAL LSH SEARCH")
    print("="*70)
    
    # Load index
    print(f"\nLoading index from {args.i}...")
    model, inverted, X_data, meta = load_index(args.i)
    n_data = len(X_data)
    m = meta["m"]
    print(f"Index loaded: n={n_data:,}, m={m}")
    
    # Load queries
    print(f"\nLoading queries from {args.q}...")
    X_query = load_dataset(args.q, args.type)
    n_queries = len(X_query)
    print(f"Queries loaded: {n_queries:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Search
    print(f"\nSearching (N={args.N}, T={args.T}, R={args.R})...")
    
    sum_af = 0.0
    sum_recall = 0.0
    sum_time_approx = 0.0
    sum_time_true = 0.0
    
    results = []
    
    for q_id in range(n_queries):
        q = X_query[q_id]
        
        # Approximate search
        t_start = time.time()
        approx_indices, approx_distances = neural_lsh_search(
            q, model, inverted, X_data, args.T, args.N, device
        )
        t_approx = time.time() - t_start
        
        # True search (brute-force)
        t_start = time.time()
        true_indices, true_distances = brute_force_search(q, X_data, args.N)
        t_true = time.time() - t_start
        
        # Compute metrics
        af = compute_af(approx_distances[0], true_distances[0])
        recall = compute_recall(approx_indices, true_indices)
        
        sum_af += af
        sum_recall += recall
        sum_time_approx += t_approx
        sum_time_true += t_true
        
        # Range search
        r_neighbors = []
        if do_range:
            for idx, dist in zip(approx_indices, approx_distances):
                if dist <= args.R:
                    r_neighbors.append(idx)
        
        results.append({
            'query_id': q_id,
            'approx_indices': approx_indices,
            'approx_distances': approx_distances,
            'true_distances': true_distances,
            'r_neighbors': r_neighbors
        })
        
        if (q_id + 1) % 100 == 0:
            print(f"  Processed {q_id + 1}/{n_queries} queries...")
    
    # Compute averages
    avg_af = sum_af / n_queries
    avg_recall = sum_recall / n_queries
    avg_time_approx = sum_time_approx / n_queries
    avg_time_true = sum_time_true / n_queries
    qps = 1.0 / avg_time_approx if avg_time_approx > 0 else 0
    
    print(f"Search completed")
    print(f"  Recall@{args.N}: {avg_recall*100:.2f}%")
    print(f"  Average AF: {avg_af:.4f}")
    print(f"  QPS: {qps:.2f}")

    # Write output
    print(f"\nWriting results to {args.o}...")
    
    with open(args.o, "w") as fout:
        fout.write("Neural LSH\n\n")
        
        for res in results:
            fout.write(f"Query: {res['query_id']}\n")
            
            for k in range(min(args.N, len(res['approx_indices']))):
                idx = res['approx_indices'][k]
                dist_approx = res['approx_distances'][k]
                dist_true = res['true_distances'][k]
                
                fout.write(f"Nearest neighbor-{k+1}: {idx}\n")
                fout.write(f"distanceApproximate: {dist_approx:.6f}\n")
                fout.write(f"distanceTrue: {dist_true:.6f}\n")
            
            if do_range:
                fout.write("R-near neighbors:\n")
                for idx in res['r_neighbors']:
                    fout.write(f"{idx}\n")
            
            fout.write("\n")
        
        # Summary metrics (required by assignment)
        fout.write(f"Average AF: {avg_af:.6f}\n")
        fout.write(f"Recall@{args.N}: {avg_recall:.6f}\n")
        fout.write(f"QPS: {qps:.6f}\n")
        fout.write(f"tApproximateAverage: {avg_time_approx:.6f}\n")
        fout.write(f"tTrueAverage: {avg_time_true:.6f}\n")

    print(f"Results saved")
    print(f"\nSummary:")
    print(f"  Average AF: {avg_af:.4f}")
    print(f"  Recall@{args.N}: {avg_recall*100:.2f}%")
    print(f"  QPS: {qps:.2f}")


if __name__ == "__main__":
    main()