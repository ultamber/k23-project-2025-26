#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlsh_search.py
--------------
Αναζήτηση k-NN με Neural LSH.

1. Φόρτωση ευρετηρίου (MLP + inverted index + dataset + metadata)
2. Multi-probe επιλογή bins από MLP
3. Ακριβής L2 αναζήτηση μόνο στους υποψήφιους
4. Υπολογισμός true k-NN με brute force
5. Εκτύπωση αποτελεσμάτων σε αρχείο (ίδιο format με 1η εργασία)
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
    p = argparse.ArgumentParser(description="Neural LSH search")

    p.add_argument("-d", required=True, help="Dataset file (input.dat)")
    p.add_argument("-q", required=True, help="Query file (query.dat)")
    p.add_argument("-i", required=True, help="Index directory (from nlsh_build.py)")
    p.add_argument("-o", required=True, help="Output file")
    p.add_argument("-type", required=True, choices=["sift", "mnist"],
                   help="Dataset type")

    p.add_argument("-N", type=int, default=1, help="Number of nearest neighbors")
    p.add_argument("-R", type=float, default=None, help="Search radius for range search")
    p.add_argument("-T", type=int, default=5, help="Number of bins to probe (multi-probe)")
    p.add_argument("-range", type=str, default="true",
                   help="Whether to perform range search (true/false)")

    p.add_argument("--seed", type=int, default=1, help="Random seed")

    return p.parse_args()


def str2bool(s):
    return str(s).lower() in ["1", "true", "yes", "y"]


# -----------------------------------------------------------
# Φόρτωση index
# -----------------------------------------------------------
def load_index(index_dir, dataset_type):
    index_dir = Path(index_dir)

    # metadata
    meta_path = index_dir / "metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)

    d_in = meta["dimension"]
    m = meta["m"]
    layers = meta["layers"]
    nodes = meta["nodes"]

    # MLP model
    model = MLPClassifier(d_in=d_in, n_out=m, layers=layers, nodes=nodes)
    state = torch.load(index_dir / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # inverted index
    with open(index_dir / "inverted_index.pkl", "rb") as f:
        inverted = pickle.load(f)

    # dataset
    X = np.load(index_dir / "dataset.npy")

    return model, inverted, X, meta


# -----------------------------------------------------------
# L2 distances
# -----------------------------------------------------------
def l2_distances(x, X):
    """
    Υπολογίζει L2 αποστάσεις από διάνυσμα x (d,) σε όλα τα rows του X (n,d).
    Επιστρέφει vector (n,).
    """
    diff = X - x
    return np.linalg.norm(diff, axis=1)


# -----------------------------------------------------------
# Κύρια ροή
# -----------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # default R από εκφώνηση, αν δεν δοθεί
    if args.R is None:
        if args.type == "mnist":
            args.R = 2000.0
        else:
            args.R = 2800.0

    do_range = str2bool(args.range)

    print("Loading index...")
    model, inverted, X_data, meta = load_index(args.i, args.type)
    n_data, dim = X_data.shape
    print(f"Dataset in index: n={n_data}, d={dim}")

    print("Loading query set...")
    X_query = load_dataset(args.q, args.type)   # ίδιο parser με το dataset
    n_queries = X_query.shape[0]
    print(f"Queries: {n_queries}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # metrics
    total_approx_time = 0.0
    total_true_time = 0.0
    sum_AF = 0.0
    sum_recall = 0.0

    start_all = time.time()

    with open(args.o, "w") as fout:
        fout.write("Neural LSH\n")

        for q_id in range(n_queries):
            q = X_query[q_id].astype(np.float32)

            # --------------------------------------
            # Approximate search (Neural LSH)
            # --------------------------------------
            t0 = time.time()
            with torch.no_grad():
                qt = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(qt)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                topT = torch.topk(probs, k=args.T)
                bins = topT.indices.cpu().numpy()

            # συλλογή υποψηφίων
            candidates = []
            for b in bins:
                if int(b) in inverted:
                    candidates.extend(inverted[int(b)])
            candidates = np.array(sorted(set(candidates)), dtype=np.int64)

            # fallback: αν δεν βρέθηκαν υποψήφιοι, πάρε όλο το dataset
            if candidates.size == 0:
                candidates = np.arange(n_data, dtype=np.int64)

            # distances μόνο στους υποψήφιους
            d_cand = l2_distances(q, X_data[candidates])
            order_cand = np.argsort(d_cand)
            topN_cand = candidates[order_cand[:args.N]]
            d_approx = d_cand[order_cand[:args.N]]

            t1 = time.time()
            tApprox = t1 - t0
            total_approx_time += tApprox

            # --------------------------------------
            # True brute-force search (για metrics)
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

            # AF: distance of best approx / best true
            AF_q = float(d_approx[0] / (best_true + 1e-12))
            sum_AF += AF_q

            # Recall@N: intersection of sets
            set_true = set(topN_true.tolist())
            set_approx = set(topN_cand.tolist())
            recall_q = len(set_true & set_approx) / float(args.N)
            sum_recall += recall_q

            # --------------------------------------
            # Range search
            # --------------------------------------
            R_neighbors = []
            if do_range:
                for idx in range(n_data):
                    if d_all[idx] <= args.R:
                        R_neighbors.append(idx)

            # --------------------------------------
            # Εκτύπωση αποτελεσμάτων για αυτό το query
            # --------------------------------------
            fout.write(f"Query: {q_id}\n")
            for k, (idx, dA) in enumerate(zip(topN_cand, d_approx), start=1):
                fout.write(f"Nearest neighbor-{k}: {idx}\n")
                fout.write(f"distanceApproximate: {float(dA)}\n")
                fout.write(f"distanceTrue: {float(d_all[idx])}\n")

            fout.write("R-near neighbors:\n")
            if do_range:
                for idx in R_neighbors:
                    fout.write(f"{idx}\n")

        # ---- συνολικά metrics ----
        total_time = time.time() - start_all
        avg_AF = sum_AF / n_queries
        avg_recall = sum_recall / n_queries
        avg_tApprox = total_approx_time / n_queries
        avg_tTrue = total_true_time / n_queries
        qps = n_queries / total_approx_time if total_approx_time > 0 else 0.0

        fout.write(f"Average AF: {avg_AF}\n")
        fout.write(f"Recall@N: {avg_recall}\n")
        fout.write(f"QPS: {qps}\n")
        fout.write(f"tApproximateAverage: {avg_tApprox}\n")
        fout.write(f"tTrueAverage: {avg_tTrue}\n")

    print("Search finished.")
    print(f"Average AF: {avg_AF:.4f}")
    print(f"Recall@N: {avg_recall:.4f}")
    print(f"QPS: {qps:.2f}")
    print(f"Avg tApprox: {avg_tApprox:.6f} s, Avg tTrue: {avg_tTrue:.6f} s")


# -----------------------------------------------------------
if __name__ == "__main__":
    main()
