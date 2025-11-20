#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import torch
from pathlib import Path

from modules.dataset_parser import load_dataset
from modules.graph_utils import build_knn_graph, to_csr, run_kahip , build_symmetric_graph
from modules.models import MLPClassifier, train_model
from modules.utils import set_seed, save_index
from modules.lsh_knn import compute_knn_lsh

def parse_args():
    parser = argparse.ArgumentParser(description="Neural LSH index builder")

    parser.add_argument("-d", required=True, help="Dataset file (.dat)")
    parser.add_argument("-i", required=True, help="Output index directory")
    parser.add_argument("-type", required=True, choices=["sift", "mnist"],
                        help="Dataset type")
    parser.add_argument("--knn", type=int, default=10, help="Number of neighbors for k-NN graph")
    parser.add_argument("-m", type=int, default=100, help="Number of partitions for KaHIP")
    parser.add_argument("--imbalance", type=float, default=0.03, help="Imbalance factor for KaHIP")
    parser.add_argument("--kahip_mode", type=int, default=2, choices=[0, 1, 2],
                        help="KaHIP mode: 0=FAST, 1=ECO, 2=STRONG")
    parser.add_argument("--layers", type=int, default=3, help="Number of MLP layers")
    parser.add_argument("--nodes", type=int, default=64, help="Hidden nodes per layer")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--search_path", default="./bin/search", help="Path to C++ LSH search executable")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    index_dir = Path(args.i)
    index_dir.mkdir(parents=True, exist_ok=True)

    # From 
    print(f"Loading dataset from {args.d} ...")
    X = load_dataset(args.d, args.type)  # np.ndarray (n, d)
    print("⚠ Using DEBUG SUBSET (1000 vectors)")
    X = X[:1000]
    print(f"Loaded dataset shape: {X.shape}")

    print(f"Building k-NN graph using LSH from Project 1...")
    knn_graph = compute_knn_lsh(
    X,
    args.knn,
    search_path=args.search_path,
    dtype=args.type
)

    print(f"Preparing CSR format and running KaHIP (m={args.m}) ...")
    adj = build_symmetric_graph(knn_graph)
    csr_graph = to_csr(adj)
    labels = run_kahip(csr_graph, m=args.m,
                       imbalance=args.imbalance,
                       mode=args.kahip_mode)
    print(f"KaHIP completed — partitions: {len(np.unique(labels))}")

    print("Training MLP classifier ...")
    model = MLPClassifier(d_in=X.shape[1],
                          n_out=args.m,
                          layers=args.layers,
                          nodes=args.nodes,
                          dropout=0.1)
    train_model(model, X, labels,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr)

    print("Saving index files ...")
    save_index(index_dir, model, X, labels,args)

    print("Neural LSH index built successfully.")

if __name__ == "__main__":
    main()
