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
import json
import argparse

# Change paths if needed
DATASET_DIR = Path("../datasets/MNIST")   # or SIFT
BUILD_SCRIPT = "nlsh_build.py"
SEARCH_SCRIPT = "nlsh_search.py"
SEARCH_BIN = "../bin/search"              # Project 1 executable


def generate_experiment_name(args):
    """Generate a descriptive name based on key parameters."""
    return f"knn{args.knn}_m{args.m}_ep{args.epochs}_L{args.layers}_N{args.nodes}_imb{args.imbalance}_lr{args.learning_rate}_wd{args.weight_decay}_dr{args.dropout}_batch{args.batch_size}_T{args.T}"


def setup_output_dirs(args):
    """Create output directory structure and return paths."""
    # Base output directory
    output_base = Path(args.output_dir)
    
    # Experiment-specific subdirectory
    experiment_name = generate_experiment_name(args)
    experiment_dir = output_base / experiment_name
    
    # Create directories
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    paths = {
        "experiment_dir": experiment_dir,
        "index_dir": experiment_dir / "index",
        "results_file": experiment_dir / "search_results.txt",
        "metrics_file": experiment_dir / "metrics.json",
        "config_file": experiment_dir / "config.json",
    }
    
    return paths


def save_config(args, paths):
    """Save experiment configuration to JSON."""
    config = {
        "dataset_dir": str(args.dataset_dir),
        "type": args.type,
        "knn": args.knn,
        "m": args.m,
        "imbalance": args.imbalance,
        "epochs": args.epochs,
        "layers": args.layers,
        "nodes": args.nodes,
        "weight_decay": args.weight_decay,
        "lr": args.learning_rate,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "N": args.N,
        "T": args.T,
        "max_queries": args.max_queries,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(paths["config_file"], "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to {paths['config_file']}")


def run_build(args, paths):
    """Run nlsh_build.py using LSH-based k-NN graph."""
    print("\n Running Neural LSH Build...")

    dataset_file = Path(args.dataset_dir) / "train-images.idx3-ubyte"
    index_dir = str(paths["index_dir"])

    cmd = [
        sys.executable, BUILD_SCRIPT,
        "-d", str(dataset_file),
        "-i", index_dir,
        "-t", args.type,
        "--knn", str(args.knn),
        "-m", str(args.m),
        "--imbalance", str(args.imbalance),
        "--epochs", str(args.epochs),
        "--layers", str(args.layers),
        "--nodes", str(args.nodes),
        "--weight_decay", str(args.weight_decay),
        "--lr", str(args.learning_rate),
        "--dropout", str(args.dropout),
        "--batch_size", str(args.batch_size)
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
        return None

    start_time = time.time()
    subprocess.run(cmd, check=True)
    build_time = time.time() - start_time
    
    print(f"✔ Build phase completed in {build_time:.2f} sec.")
    return build_time


def run_search(args, paths):
    """Run nlsh_search.py on actual query.dat dataset."""
    print("\n Running Neural LSH Search...")

    dataset_file = Path(args.dataset_dir) / "train-images.idx3-ubyte"
    query_file = Path(args.dataset_dir) / "t10k-images.idx3-ubyte"

    cmd = [
        sys.executable, SEARCH_SCRIPT,
        "-d", str(dataset_file),
        "-q", str(query_file),
        "-i", str(paths["index_dir"]),
        "-o", str(paths["results_file"]),
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
        return None

    start_time = time.time()
    subprocess.run(cmd, check=True)
    search_time = time.time() - start_time
    
    print(f"✔ Search phase completed in {search_time:.2f} sec.")
    return search_time


def save_metrics(paths, build_time, search_time, total_time):
    """Save timing metrics to JSON."""
    metrics = {
        "build_time_sec": build_time,
        "search_time_sec": search_time,
        "total_time_sec": total_time,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(paths["metrics_file"], "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {paths['metrics_file']}")


def show_output(results_file, max_lines=80):
    """Display contents of results file."""
    print(f"\n Contents of {results_file}:\n")

    if not Path(results_file).exists():
        print("Results file not found.")
        return

    with open(results_file, "r") as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i > max_lines:
                print("... (file truncated)")
                break


def main():
    start = time.time()

    print("===============================================")
    print("     Neural LSH – Full Pipeline Test          ")
    print("===============================================")
    
    parser = argparse.ArgumentParser(description="Run full Neural LSH pipeline with optional overrides")
    
    # Dataset options
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Dataset directory")
    parser.add_argument("--type", default="mnist", choices=["mnist", "sift"], help="Dataset type")
    
    # Output options
    parser.add_argument("--output-dir", default="./experiments", help="Base output directory for all experiments")
    parser.add_argument("--experiment-name", default=None, help="Custom experiment name (auto-generated if omitted)")
    
    # Build parameters
    parser.add_argument("--knn", type=int, default=25, help="k for k-NN graph")
    parser.add_argument("--m", type=int, default=200, help="Number of partitions (m)")
    parser.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--layers", type=int, default=3, help="MLP layers")
    parser.add_argument("--nodes", type=int, default=128, help="Nodes per layer")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay for MLP training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for MLP training")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for MLP training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for MLP training")
    
    # Precomputed files
    parser.add_argument("--knn-graph-file", type=str, default=None, help="Precomputed k-NN graph file")
    parser.add_argument("--calculated-output", default=None, help="Precomputed Project1 output file")
    parser.add_argument("--search-path", default=SEARCH_BIN, help="Path to Project1 search binary")
    
    # Search parameters
    parser.add_argument("--N", type=int, default=5, help="Search N (number of neighbors)")
    parser.add_argument("--T", type=int, default=10, help="Search T (probes)")
    parser.add_argument("--max-queries", type=int, default=1000, help="Limit number of queries")
    
    # Control flags
    parser.add_argument("--skip-build", action="store_true", help="Skip the build phase")
    parser.add_argument("--skip-search", action="store_true", help="Skip the search phase")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't run them")
    parser.add_argument("--show-results", action="store_true", help="Display results file after search")
    
    args = parser.parse_args()

    # Setup output directories
    paths = setup_output_dirs(args)
    print(f"\nExperiment directory: {paths['experiment_dir']}")
    
    # Save configuration
    save_config(args, paths)
    
    # Run phases
    build_time = None
    search_time = None
    
    if not args.skip_build:
        build_time = run_build(args, paths)
    else:
        print("Skipping build phase as requested")

    if not args.skip_search:
        search_time = run_search(args, paths)
    else:
        print("Skipping search phase as requested")

    total_time = time.time() - start
    
    # Save metrics
    if not args.dry_run:
        save_metrics(paths, build_time, search_time, total_time)
    
    if args.show_results and paths["results_file"].exists():
        show_output(paths["results_file"])

    print(f"\n{'='*50}")
    print(f"Experiment completed!")
    print(f"  Output directory: {paths['experiment_dir']}")
    print(f"  Total time: {total_time:.2f} sec")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()