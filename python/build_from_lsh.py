#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_from_lsh.py
-----------------
Complete pipeline: Parse LSH output → Build Neural LSH index
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build Neural LSH index from LSH output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # LSH output parsing
    parser.add_argument("--lsh_output", required=True,
                        help="LSH output file to parse")
    parser.add_argument("--n_queries", type=int, required=True,
                        help="Number of queries (dataset size)")
    parser.add_argument("--k_neighbors", type=int, default=10,
                        help="Number of neighbors per query")
    
    # Index building
    parser.add_argument("-d", "--dataset", required=True,
                        help="Dataset file path")
    parser.add_argument("-i", "--index_dir", required=True,
                        help="Output index directory")
    parser.add_argument("-type", required=True, choices=["mnist", "sift"],
                        help="Dataset type")
    parser.add_argument("-m", type=int, default=50,
                        help="Number of partitions")
    
    # MLP parameters
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    
    # Optional
    parser.add_argument("--knn_graph_output", default="knn_graph.npy",
                        help="Where to save parsed k-NN graph")
    parser.add_argument("--keep_graph", action="store_true",
                        help="Keep k-NN graph file after building")
    
    args = parser.parse_args()
    
    # Step 1: Parse LSH output
    print("="*70)
    print("STEP 1: Parsing LSH output to k-NN graph")
    print("="*70)
    
    parse_cmd = [
        "python", "parse_lsh_output.py",
        "-i", args.lsh_output,
        "-o", args.knn_graph_output,
        "-n", str(args.n_queries),
        "-k", str(args.k_neighbors)
    ]
    
    result = subprocess.run(parse_cmd)
    if result.returncode != 0:
        print("❌ Failed to parse LSH output")
        sys.exit(1)
    
    # Step 2: Build Neural LSH index
    print("\n" + "="*70)
    print("STEP 2: Building Neural LSH index")
    print("="*70)
    
    build_cmd = [
        "python", "nlsh_build.py",
        "-d", args.dataset,
        "-i", args.index_dir,
        "-type", args.type,
        "-m", str(args.m),
        "--knn_graph_file", args.knn_graph_output,
        "--layers", str(args.layers),
        "--nodes", str(args.nodes),
        "--dropout", str(args.dropout),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size)
    ]
    
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        print("❌ Failed to build index")
        sys.exit(1)
    
    # Cleanup
    if not args.keep_graph:
        Path(args.knn_graph_output).unlink()
        print(f"\n🗑️  Removed temporary k-NN graph file")
    
    print("\n" + "="*70)
    print("✓ COMPLETE: Neural LSH index built from LSH output")
    print("="*70)


if __name__ == "__main__":
    main()