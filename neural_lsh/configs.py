#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto-Config Neural LSH Test
Automatically generates and tests multiple configurations from parameter ranges.
"""

import subprocess
import time
from pathlib import Path
import sys
from itertools import product
import argparse
import os
import concurrent.futures

# =============================================================================
# PARAMETER RANGES - EDIT THESE TO GENERATE CONFIGS AUTOMATICALLY
# =============================================================================

# Your dataset and script paths
DATASET_DIR = Path("../datasets/MNIST")
BUILD_SCRIPT = "nlsh_build.py" 
SEARCH_SCRIPT = "nlsh_search.py"
SEARCH_BIN = "../bin/search"

# Define parameter ranges to test
PARAM_RANGES = {
    "knn": [25,50],           # Try these k values
    "m": [100,128,256],          # Try these m values  
    "epochs": [50],            # Try these epoch counts
    "layers": [3,4,5],         # Try these layer counts
    "nodes": [64,128,256],           # Try these node counts
    "batch_size": [128,256],
    "lr": [0.01, 0.001, 0.0001],        # Try these learning rates
    "N": [1],               # Try these N values
    "T": [5]              # Try these T values
}

# Fixed parameters (not varied)
FIXED_PARAMS = {
    "imbalance": "0.03",
    "type": "mnist",
    "knn_graph": "./computed_graph/knn_graph.npy",
    "calculated_output": "../search_output.txt"
}

# Limit total configs (safety check)
MAX_CONFIGS = None  # Set to None for no limit

# =============================================================================
# AUTO CONFIG GENERATION - NO NEED TO EDIT
# =============================================================================

def generate_configs():
    """Generate all combinations of parameters."""
    
    # Get all parameter combinations
    param_names = list(PARAM_RANGES.keys())
    param_values = [PARAM_RANGES[name] for name in param_names]
    
    configs = []
    for i, combination in enumerate(product(*param_values)):
        if MAX_CONFIGS and i >= MAX_CONFIGS:
            print(f"⚠️  Limited to {MAX_CONFIGS} configs (there are more combinations)")
            break
            
        config = dict(zip(param_names, combination))
        config["name"] = f"config_{i+1:02d}"
        configs.append(config)
    
    return configs

def run_config(config):
    """Run one configuration."""
    print(f"\n{'='*50}")
    print(f"🧪 Testing: {config['name']}")
    for key, value in config.items():
        if key != "name":
            print(f"   {key}={value}")
    print('='*50)
    
    start_time = time.time()
    
    # Build command
    build_cmd = [
        sys.executable, BUILD_SCRIPT,
        "-d", str(DATASET_DIR/"train-images.idx3-ubyte"),
        "-i", f"{config['name']}_index",
        "-type", FIXED_PARAMS["type"],
        "--knn", str(config["knn"]),
        "-m", str(config["m"]),
        "--imbalance", FIXED_PARAMS["imbalance"],
        "--epochs", str(config["epochs"]),
        "--layers", str(config["layers"]),
        "--nodes", str(config["nodes"]),
        "--search_path", SEARCH_BIN,
        "--knn_graph_file", FIXED_PARAMS["knn_graph"],
        "--calculated_output", FIXED_PARAMS["calculated_output"]
    ]
    
    print("🔨 Running build...")
    build_start = time.time()
    try:
        build_proc = subprocess.run(build_cmd, capture_output=True, text=True)
        build_time = time.time() - build_start
        print(f"✅ Build: {build_time:.1f}s (rc={build_proc.returncode})")
        # Write build logs
        with open(f"{config['name']}_build.log", "w", encoding="utf-8") as bf:
            bf.write(build_proc.stdout or "")
            bf.write("\n--- STDERR ---\n")
            bf.write(build_proc.stderr or "")
        if build_proc.returncode != 0:
            raise subprocess.CalledProcessError(build_proc.returncode, build_cmd, output=build_proc.stdout, stderr=build_proc.stderr)
    except Exception as e:
        build_time = time.time() - build_start
        print(f"❌ Build failed for {config['name']}: {e}")
        return {"name": config["name"], "params": config, "error": str(e)}
    
    # Search command
    search_cmd = [
        sys.executable, SEARCH_SCRIPT,
        "-d", str(DATASET_DIR/"train-images.idx3-ubyte"),
        "-q", str(DATASET_DIR/"t10k-images.idx3-ubyte"),
        "-i", f"{config['name']}_index",
        "-o", f"{config['name']}_results.txt",
        "-type", FIXED_PARAMS["type"],
        "-N", str(config["N"]),
        "-T", str(config["T"]),
        "-range", "false"
    ]
    
    print("🔍 Running search...")
    search_start = time.time()
    try:
        search_proc = subprocess.run(search_cmd, capture_output=True, text=True)
        search_time = time.time() - search_start
        print(f"✅ Search: {search_time:.1f}s (rc={search_proc.returncode})")
        # Write search logs
        with open(f"{config['name']}_search.log", "w", encoding="utf-8") as sf:
            sf.write(search_proc.stdout or "")
            sf.write("\n--- STDERR ---\n")
            sf.write(search_proc.stderr or "")
        if search_proc.returncode != 0:
            raise subprocess.CalledProcessError(search_proc.returncode, search_cmd, output=search_proc.stdout, stderr=search_proc.stderr)
    except Exception as e:
        search_time = time.time() - search_start
        print(f"❌ Search failed for {config['name']}: {e}")
        return {"name": config["name"], "params": config, "error": str(e)}
    
    total_time = time.time() - start_time
    print(f"⏱️  Total: {total_time:.1f}s")
    
    return {
        "name": config["name"],
        "params": config,
        "build_time": build_time,
        "search_time": search_time,
        "total_time": total_time
    }


def run_all_configs(configs, max_workers=None):
    """Run configurations in parallel using a ThreadPoolExecutor.

    Each config will run `run_config` in a separate worker. Returns list of results.
    """
    results = []
    max_workers = max_workers or min(32, (os.cpu_count() or 4))

    print(f"\n➡️  Running {len(configs)} configurations with {max_workers} workers")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_cfg = {ex.submit(run_config, cfg): cfg for cfg in configs}

        for i, fut in enumerate(concurrent.futures.as_completed(future_to_cfg), 1):
            cfg = future_to_cfg[fut]
            try:
                res = fut.result()
                results.append(res)
                print(f"  [{i}/{len(configs)}] Completed: {cfg['name']}")
            except Exception as e:
                print(f"  [{i}/{len(configs)}] Error running {cfg['name']}: {e}")
                results.append({"name": cfg['name'], "params": cfg, "error": str(e)})

    return results

def show_parameter_info():
    """Show what parameters will be tested."""
    print("🔧 Parameter ranges:")
    for param, values in PARAM_RANGES.items():
        print(f"   {param}: {values}")
    
    print(f"\n📐 Fixed parameters:")
    for param, value in FIXED_PARAMS.items():
        print(f"   {param}: {value}")
    
    total_combinations = 1
    for values in PARAM_RANGES.values():
        total_combinations *= len(values)
    
    actual_configs = min(total_combinations, MAX_CONFIGS) if MAX_CONFIGS else total_combinations
    
    print(f"\n📊 Total possible combinations: {total_combinations}")
    print(f"📋 Configurations to test: {actual_configs}")

def main():
    print("🚀 Neural LSH Auto-Config Test")
    print("="*50)
    
    # Show parameter info
    show_parameter_info()
    
    # Generate configurations
    configs = generate_configs()
    print(f"\n✅ Generated {len(configs)} configurations")
    
    parser = argparse.ArgumentParser(description="Run Neural LSH configs")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (defaults to cpu count)")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm and run without prompt")
    args = parser.parse_args()

    if not args.yes:
        response = input(f"\n🤔 Start testing {len(configs)} configurations? (y/N): ")
        if not response.lower().startswith('y'):
            print("👋 Cancelled")
            return

    print(f"\n🏁 Starting experiments...")

    overall_start = time.time()

    # Run in parallel
    results = run_all_configs(configs, max_workers=args.workers)
    
    # Results summary
    print(f"\n{'='*70}")
    print("📊 RESULTS SUMMARY")
    print('='*70)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n⏱️  Top 5 fastest configurations:")
        sorted_results = sorted(successful, key=lambda x: x['total_time'])[:5]
        
        for i, result in enumerate(sorted_results, 1):
            print(f"  {i}. {result['name']}: {result['total_time']:.1f}s")
            print(f"     knn={result['params']['knn']}, m={result['params']['m']}, "
                  f"epochs={result['params']['epochs']}, layers={result['params']['layers']}")
        
        print(f"\n📈 Performance range:")
        times = [r['total_time'] for r in successful]
        print(f"   Fastest: {min(times):.1f}s")
        print(f"   Slowest: {max(times):.1f}s") 
        print(f"   Average: {sum(times)/len(times):.1f}s")
    
    if failed:
        print(f"\n❌ Failed configurations: {[r['name'] for r in failed]}")
    
    total_time = time.time() - overall_start
    print(f"\n🎉 All experiments completed in {total_time:.1f}s")
    print(f"📁 Check *_results.txt files for detailed outputs")

if __name__ == "__main__":
    main()