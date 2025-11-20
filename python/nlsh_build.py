#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nlsh_build.py
-------------
Neural LSH index construction pipeline.

Steps:
1. Load dataset (MNIST or SIFT)
2. Build k-NN graph using LSH from Project 1
3. Symmetrize graph with edge weights
4. Convert to CSR format
5. Run KaHIP for balanced partitioning
6. Train MLP classifier to predict partitions
7. Save index (model + inverted index + metadata)
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from modules.dataset_parser import load_dataset
from modules.graph_utils import build_symmetric_graph, to_csr, run_kahip
from modules.models import MLPClassifier, train_model
from modules.utils import set_seed, save_index
from modules.lsh_knn import compute_knn_lsh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural LSH Index Builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("-d", required=True, help="Dataset file path (.dat, .idx3-ubyte, .fvecs)")
    parser.add_argument("-i", required=True, help="Output index directory")
    parser.add_argument("-type", required=True, choices=["sift", "mnist"],
                        help="Dataset type")
    
    # k-NN graph parameters
    parser.add_argument("--knn", type=int, default=10, 
                        help="Number of neighbors for k-NN graph")
    parser.add_argument("--search_path", default="../bin/search", 
                        help="Path to Project 1 LSH search executable")
    
    # KaHIP partitioning parameters
    parser.add_argument("-m", type=int, default=100, 
                        help="Number of partitions")
    parser.add_argument("--imbalance", type=float, default=0.03, 
                        help="Imbalance factor for KaHIP (0.03 = 3%%)")
    parser.add_argument("--kahip_mode", type=int, default=2, choices=[0, 1, 2],
                        help="KaHIP mode: 0=FAST, 1=ECO, 2=STRONG")
    
    # MLP training parameters
    parser.add_argument("--layers", type=int, default=3, 
                        help="Number of MLP layers")
    parser.add_argument("--nodes", type=int, default=64, 
                        help="Hidden nodes per layer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, 
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Learning rate")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=1, 
                        help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                        help="Use small subset for debugging (1000 vectors)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Create index directory
    index_dir = Path(args.i)
    index_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(" NEURAL LSH INDEX CONSTRUCTION")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    print(f"\n[STEP 1] Loading dataset from {args.d}...")
    X = load_dataset(args.d, args.type)
    
    if args.debug:
        print(f"\n⚠️  DEBUG MODE: Using subset of 1000 vectors")
        X = X[:1000]
    
    n, d = X.shape
    print(f"✓ Dataset loaded: n={n:,} vectors, d={d} dimensions")
    print(f"  Data type: {X.dtype}")
    print(f"  Value range: [{X.min():.2f}, {X.max():.2f}]")
    
    # Normalize data if needed
    if X.max() > 1.0 and args.type == "mnist":
        print(f"  Normalizing MNIST data to [0, 1] range")
        X = X / 255.0
    
    # ========================================================================
    # STEP 2: Build k-NN Graph using LSH
    # ========================================================================
    print(f"\n[STEP 2] Building k-NN graph (k={args.knn}) using LSH...")
    
    try:
        knn_graph = compute_knn_lsh(
            X,
            args.knn,
            search_path=args.search_path,
            dtype=args.type
        )
        print(f"✓ k-NN graph built successfully")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print(f"Falling back to exact k-NN (sklearn)...")
        from modules.lsh_knn import compute_knn_sklearn
        knn_graph = compute_knn_sklearn(X, args.knn)
        print(f"✓ Exact k-NN graph computed")
    
    except Exception as e:
        print(f"\n❌ Error building k-NN graph: {e}")
        print(f"Falling back to exact k-NN (sklearn)...")
        from modules.lsh_knn import compute_knn_sklearn
        knn_graph = compute_knn_sklearn(X, args.knn)
    
    # ========================================================================
    # STEP 3: Symmetrize Graph with Edge Weights
    # ========================================================================
    print(f"\n[STEP 3] Symmetrizing graph and assigning edge weights...")
    adj = build_symmetric_graph(knn_graph)
    print(f"✓ Graph symmetrized")
    
    # ========================================================================
    # STEP 4: Convert to CSR Format
    # ========================================================================
    print(f"\n[STEP 4] Converting to CSR format for KaHIP...")
    csr_graph = to_csr(adj)
    
    # # Validate CSR format
    # try:
    #     validate_csr(csr_graph)
    # except AssertionError as e:
    #     print(f"❌ CSR validation failed: {e}")
    #     raise
    
    # ========================================================================
    # STEP 5: Run KaHIP for Balanced Partitioning
    # ========================================================================
    print(f"\n[STEP 5] Running KaHIP partitioning...")
    labels = run_kahip(
        csr_graph,
        m=args.m,
        imbalance=args.imbalance,
        mode=args.kahip_mode,
        seed=args.seed
    )
    
    # Verify partitioning quality
    unique_labels = np.unique(labels)
    print(f"✓ Partitioning completed")
    print(f"  Unique partitions: {len(unique_labels)}")
    print(f"  Expected: {args.m}")
    
    if len(unique_labels) != args.m:
        print(f"  ⚠️  Warning: Got {len(unique_labels)} partitions instead of {args.m}")
    
    # ========================================================================
    # STEP 6: Train MLP Classifier
    # ========================================================================
    print(f"\n[STEP 6] Training MLP classifier...")
    print(f"  Input dim: {d}")
    print(f"  Output classes: {args.m}")
    print(f"  Hidden layers: {args.layers}")
    print(f"  Nodes per layer: {args.nodes}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    model = MLPClassifier(
        d_in=d,
        n_out=args.m,
        layers=args.layers,
        nodes=args.nodes,
        dropout=args.dropout
    )
    
    train_model(
        model,
        X,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=True
    )
    
    print(f"✓ MLP training completed")
    
    # ========================================================================
    # STEP 7: Save Index
    # ========================================================================
    print(f"\n[STEP 7] Saving index to {index_dir}...")
    save_index(index_dir, model, X, labels, args)
    
    print("\n" + "="*70)
    print("NEURAL LSH INDEX BUILT SUCCESSFULLY")
    print("="*70)
    print(f"\nIndex location: {index_dir.absolute()}")
    print(f"Index files:")
    print(f"  - model.pth (trained MLP)")
    print(f"  - inverted_index.pkl (partition → point mapping)")
    print(f"  - dataset.npy (original data)")
    print(f"  - metadata.json (configuration)")
    print()


if __name__ == "__main__":
    main()