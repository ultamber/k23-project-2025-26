#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parameter sweep script for Neural LSH experiments.
Runs the pipeline with different parameter combinations.
"""

import subprocess
import sys
import itertools
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


# Default parameter grids for experimentation
DEFAULT_PARAM_GRID = {
    "knn": [10, 25, 50],
    "m": [100, 200, 400],
    "epochs": [25, 50, 100],
    "layers": [2, 3, 4],
    "nodes": [64, 128, 256],
    "imbalance": [0.03],
}

# Quick test grid (fewer combinations)
QUICK_PARAM_GRID = {
    "knn": [25],
    "m": [100, 200],
    "epochs": [25, 50],
    "layers": [3],
    "nodes": [128],
    "imbalance": [0.03],
}

# Minimal test grid (for debugging)
MINIMAL_PARAM_GRID = {
    "knn": [25],
    "m": [100],
    "epochs": [10],
    "layers": [2],
    "nodes": [64],
    "imbalance": [0.03],
}


def generate_param_combinations(param_grid):
    """Generate all combinations of parameters."""
    keys = param_grid.keys()
    values = param_grid.values()
    
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def run_experiment(params, base_args, dry_run=False):
    """Run a single experiment with given parameters."""
    cmd = [
        sys.executable, "nlsh_pipeline.py",
        "--output-dir", base_args["output_dir"],
        "--dataset-dir", base_args["dataset_dir"],
        "--type", base_args["type"],
        "--max-queries", str(base_args["max_queries"]),
    ]
    
    # Map grid parameter names to CLI argument names
    param_to_arg = {
        "knn": "--knn",
        "m": "--m",
        "epochs": "--epochs",
        "layers": "--layers",
        "nodes": "--nodes",
        "imbalance": "--imbalance",
        "lr": "--learning-rate",
        "learning_rate": "--learning-rate",
        "weight_decay": "--weight-decay",
        "dropout": "--dropout",
        "batch_size": "--batch-size",
        "T": "--T",
        "N": "--N",
    }
    
    for param_name, param_value in params.items():
        if param_name in param_to_arg:
            cmd += [param_to_arg[param_name], str(param_value)]
        elif param_name == "seed":
            # Skip seed if nlsh_pipeline doesn't support it yet
            pass
        else:
            # Try to pass unknown params with -- prefix
            cmd += [f"--{param_name.replace('_', '-')}", str(param_value)]
    
    # Add N and T from base_args if not in params
    if "N" not in params:
        cmd += ["--N", str(base_args["N"])]
    if "T" not in params:
        cmd += ["--T", str(base_args["T"])]
    
    # Add precomputed file paths
    if base_args.get("knn_graph_file"):
        cmd += ["--knn-graph-file", base_args["knn_graph_file"]]
    if base_args.get("calculated_output"):
        cmd += ["--calculated-output", base_args["calculated_output"]]
    if base_args.get("search_path"):
        cmd += ["--search-path", base_args["search_path"]]
    
    # Add control flags
    if base_args.get("skip_build"):
        cmd += ["--skip-build"]
    if base_args.get("skip_search"):
        cmd += ["--skip-search"]
    if dry_run:
        cmd += ["--dry-run"]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {params}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if base_args.get("verbose", False):
        # Show output in real-time
        result = subprocess.run(cmd)
    else:
        # Capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
    
    elapsed = time.time() - start_time
    
    return {
        "params": params,
        "returncode": result.returncode,
        "elapsed_time": elapsed,
        "success": result.returncode == 0,
    }


def save_sweep_summary(results, output_dir):
    """Save summary of all experiments."""
    summary_file = Path(output_dir) / "sweep_summary.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_time_sec": sum(r["elapsed_time"] for r in results),
        "experiments": results,
    }
    
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSweep summary saved to {summary_file}")
    return summary


def print_summary(summary):
    """Print a formatted summary of the sweep."""
    print("\n" + "="*60)
    print("PARAMETER SWEEP SUMMARY")
    print("="*60)
    print(f"Total experiments:  {summary['total_experiments']}")
    print(f"Successful:         {summary['successful']}")
    print(f"Failed:             {summary['failed']}")
    print(f"Total time:         {summary['total_time_sec']:.2f} sec")
    print("="*60)
    
    if summary['failed'] > 0:
        print("\nFailed experiments:")
        for exp in summary['experiments']:
            if not exp['success']:
                print(f"  - {exp['params']}")


def load_custom_grid(grid_file):
    """Load a custom parameter grid from JSON file."""
    with open(grid_file, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for Neural LSH")
    
    # Sweep configuration
    parser.add_argument("--grid", choices=["default", "quick", "minimal", "custom"], 
                        default="quick", help="Parameter grid to use")
    parser.add_argument("--grid-file", type=str, help="Custom grid JSON file (when --grid=custom)")
    
    # Base experiment settings
    parser.add_argument("--output-dir", default="./sweep_experiments", help="Base output directory")
    parser.add_argument("--dataset-dir", default="../datasets/MNIST", help="Dataset directory")
    parser.add_argument("--type", default="mnist", choices=["mnist", "sift"], help="Dataset type")
    
    # Search parameters (fixed across sweep, unless overridden in grid)
    parser.add_argument("--N", type=int, default=5, help="Search N")
    parser.add_argument("--T", type=int, default=50, help="Search T")
    parser.add_argument("--max-queries", type=int, default=1000, help="Max queries")
    
    # Precomputed files
    parser.add_argument("--knn-graph-file", type=str, default="./computed_graph/knn_graph.npy", help="Precomputed k-NN graph")
    parser.add_argument("--calculated-output", type=str, default="../out_ivfflat.txt", help="Precomputed output file")
    parser.add_argument("--search-path", type=str, default="../bin/search", help="Search binary path")
    
    # Control flags
    parser.add_argument("--skip-build", action="store_true", help="Skip build phase")
    parser.add_argument("--skip-search", action="store_true", help="Skip search phase")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--verbose", action="store_true", help="Show subprocess output")
    parser.add_argument("--max-experiments", type=int, default=None, help="Limit number of experiments")
    
    # Custom parameter overrides (comma-separated values)
    parser.add_argument("--knn-values", type=str, help="Comma-separated knn values, e.g., '10,25,50'")
    parser.add_argument("--m-values", type=str, help="Comma-separated m values")
    parser.add_argument("--epochs-values", type=str, help="Comma-separated epochs values")
    parser.add_argument("--layers-values", type=str, help="Comma-separated layers values")
    parser.add_argument("--nodes-values", type=str, help="Comma-separated nodes values")
    parser.add_argument("--imbalance-values", type=str, help="Comma-separated imbalance values")
    parser.add_argument("--lr-values", type=str, help="Comma-separated learning rate values")
    parser.add_argument("--weight-decay-values", type=str, help="Comma-separated weight decay values")
    parser.add_argument("--dropout-values", type=str, help="Comma-separated dropout values")
    parser.add_argument("--batch-size-values", type=str, help="Comma-separated batch size values")
    parser.add_argument("--T-values", type=str, help="Comma-separated T values")
    
    args = parser.parse_args()
    
    # Select parameter grid
    if args.grid == "default":
        param_grid = DEFAULT_PARAM_GRID.copy()
    elif args.grid == "quick":
        param_grid = QUICK_PARAM_GRID.copy()
    elif args.grid == "minimal":
        param_grid = MINIMAL_PARAM_GRID.copy()
    elif args.grid == "custom":
        if not args.grid_file:
            print("Error: --grid-file required when using --grid=custom")
            sys.exit(1)
        param_grid = load_custom_grid(args.grid_file)
    
    # Apply command-line overrides
    if args.knn_values:
        param_grid["knn"] = [int(x) for x in args.knn_values.split(",")]
    if args.m_values:
        param_grid["m"] = [int(x) for x in args.m_values.split(",")]
    if args.epochs_values:
        param_grid["epochs"] = [int(x) for x in args.epochs_values.split(",")]
    if args.layers_values:
        param_grid["layers"] = [int(x) for x in args.layers_values.split(",")]
    if args.nodes_values:
        param_grid["nodes"] = [int(x) for x in args.nodes_values.split(",")]
    if args.imbalance_values:
        param_grid["imbalance"] = [float(x) for x in args.imbalance_values.split(",")]
    if args.lr_values:
        param_grid["lr"] = [float(x) for x in args.lr_values.split(",")]
    if args.weight_decay_values:
        param_grid["weight_decay"] = [float(x) for x in args.weight_decay_values.split(",")]
    if args.dropout_values:
        param_grid["dropout"] = [float(x) for x in args.dropout_values.split(",")]
    if args.batch_size_values:
        param_grid["batch_size"] = [int(x) for x in args.batch_size_values.split(",")]
    if args.T_values:
        param_grid["T"] = [int(x) for x in args.T_values.split(",")]

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save the parameter grid used
    grid_file = Path(args.output_dir) / "param_grid.json"
    with open(grid_file, "w") as f:
        json.dump(param_grid, f, indent=2)
    
    # Generate combinations
    combinations = list(generate_param_combinations(param_grid))
    total_experiments = len(combinations)
    
    if args.max_experiments:
        combinations = combinations[:args.max_experiments]
    
    print(f"\n{'='*60}")
    print(f"NEURAL LSH PARAMETER SWEEP")
    print(f"{'='*60}")
    print(f"Parameter grid: {args.grid}")
    print(f"Total combinations: {total_experiments}")
    print(f"Running: {len(combinations)} experiments")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")
    
    # Prepare base arguments
    base_args = {
        "output_dir": args.output_dir,
        "dataset_dir": args.dataset_dir,
        "type": args.type,
        "N": args.N,
        "T": args.T,
        "max_queries": args.max_queries,
        "knn_graph_file": args.knn_graph_file,
        "calculated_output": args.calculated_output,
        "search_path": args.search_path,
        "skip_build": args.skip_build,
        "skip_search": args.skip_search,
        "verbose": args.verbose,
    }
    
    # Run experiments
    results = []
    start_time = time.time()
    
    for i, params in enumerate(combinations, 1):
        print(f"\n[{i}/{len(combinations)}] Starting experiment...")
        result = run_experiment(params, base_args, dry_run=args.dry_run)
        results.append(result)
        
        if result["success"]:
            print(f"✔ Completed in {result['elapsed_time']:.2f} sec")
        else:
            print(f"✘ Failed with return code {result['returncode']}")
    
    total_time = time.time() - start_time
    
    # Save and print summary
    summary = save_sweep_summary(results, args.output_dir)
    summary["total_time_sec"] = total_time
    print_summary(summary)


if __name__ == "__main__":
    main()