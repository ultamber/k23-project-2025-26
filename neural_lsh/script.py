#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Full end-to-end test of Neural LSH using the MNIST/SIFT binary datasets
and the LSH implementation from Project 1 (via /bin/search).
"""

import os
import subprocess
import time
from pathlib import Path
import sys

# Change paths if needed
TYPE = "MNIST"
DATASET_DIR = Path(f"../datasets/{TYPE}")   # or SIFT
BUILD_SCRIPT = "nlsh_build.py"
SEARCH_SCRIPT = "nlsh_search.py"
import argparse
SEARCH_BIN = "../bin/search"              # Project 1 executable


def run_build(args):
    """
    Run nlsh_build.py using LSH-based k-NN graph.
    """
    print("\n Running Neural LSH Build...")

    dataset_file = Path(args.dataset_dir) / "train-images.idx3-ubyte"
    index_dir = args.index_name

    cmd = [
        sys.executable, BUILD_SCRIPT,
        "-d", str(dataset_file),
        "-i", index_dir,
        "-type", args.type,
        "--knn", str(args.knn),
        "-m", str(args.m),
        "--imbalance", str(args.imbalance),
        "--epochs", str(args.epochs),
        "--layers", str(args.layers),
        "--nodes", str(args.nodes),
    ]

    if args.knn_graph_file:
        cmd += ["--knn_graph_file", str(args.knn_graph_file)]
    if args.calculated_output:
        cmd += ["--calculated_output", str(args.calculated_output)]
    if args.search_path:
        cmd += ["--search_path", str(args.search_path)]

    print("Command:", " ".join(cmd))
    if args.dry_run:
        print("Dry-run: skipping build execution")
        return

    subprocess.run(cmd, check=True)
    print("✔ Build phase completed.")


def run_search(args):
    """
    Run nlsh_search.py on actual query.dat dataset.
    """
    print("\n Running Neural LSH Search...")

    dataset_file = Path(args.dataset_dir) / "train-images.idx3-ubyte"
    query_file = Path(args.dataset_dir) / "t10k-images.idx3-ubyte"

    cmd = [
        sys.executable, SEARCH_SCRIPT,
        "-d", str(dataset_file),
        "-q", str(query_file),
        "-i", args.index_name,
        "-o", args.out_file,
        "-type", args.type,
        "-N", str(args.N),
        "-T", str(args.T),
        "-range", "false",
    ]

    if args.max_queries is not None:
        cmd += ["--max-queries", str(args.max_queries)]

    print("Command:", " ".join(cmd))
    if args.dry_run:
        print("Dry-run: skipping search execution")
        return

    subprocess.run(cmd, check=True)
    print("✔ Search phase completed.")


def show_output():
    print("\n Contents of test_output.txt:\n")

    with open("test_output.txt", "r") as f:
        # Print only the first 80 lines to avoid spam
        for i, line in enumerate(f):
            print(line.rstrip())
            if i > 80:
                print("... (file truncated)")
                break


def main():
    start = time.time()

    print("===============================================")
    print("     Neural LSH – Full Pipeline Test (MNIST)   ")
    print("===============================================")
    
    # parse CLI args with sensible defaults
    parser = argparse.ArgumentParser(description="Run full Neural LSH pipeline with optional overrides")
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Dataset directory (default: ../datasets/MNIST)")
    parser.add_argument("--type", default="mnist", choices=["mnist", "sift"], help="Dataset type")
    parser.add_argument("--index-name", default=None, help="Index directory name (auto-generated from params when omitted)")
    parser.add_argument("--knn", type=int, default=25, help="k for k-NN graph")
    parser.add_argument("--m", type=int, default=200, help="Number of partitions (m)")
    parser.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--layers", type=int, default=3, help="MLP layers")
    parser.add_argument("--nodes", type=int, default=128, help="Nodes per layer")
    parser.add_argument("--knn-graph-file",type=str, default="./test_index_111/knn_graph.npy", help="Precomputed k-NN graph file")
    parser.add_argument("--calculated-output", default="../out_ivfflat.txt", help="Precomputed Project1 output file")
    parser.add_argument("--search-path", default=SEARCH_BIN, help="Path to Project1 search binary")
    parser.add_argument("--N", type=int, default=5, help="Search N")
    parser.add_argument("--T", type=int, default=10, help="Search T")
    parser.add_argument("--max-queries", type=int, default=1000, help="Limit number of queries to process (first N)")
    parser.add_argument("--out-file", default=None, help="Search output file (auto-generated when omitted)")
    parser.add_argument("--skip-build", action="store_true", help="Skip the build phase")
    parser.add_argument("--skip-search", action="store_true", help="Skip the search phase")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't run them")
    args = parser.parse_args()

    # If index-name / out-file not provided, generate from key params
    auto_base = f"index_knn{args.knn}_m{args.m}_ep{args.epochs}_layers{args.layers}_nodes{args.nodes}"
    if args.index_name is None:
        args.index_name = auto_base

    if args.out_file is None:
        args.out_file = f"{args.index_name}.txt"

    # run phases based on args
    if not args.skip_build:
        run_build(args)
    else:
        print("Skipping build phase as requested")

    if not args.skip_search:
        run_search(args)
    else:
        print("Skipping search phase as requested")
    # show_output()

    print(f"\nTotal pipeline time: {time.time() - start:.2f} sec")


if __name__ == "__main__":
    main()
