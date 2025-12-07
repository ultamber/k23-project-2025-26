#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Neural LSH Full Parameter Testing Suite
======================================
Enhanced test script with comprehensive parameter sweeps and automatic testing.
"""

import os
import subprocess
import time
import json
import csv
from pathlib import Path
import sys
import argparse
from itertools import product
import numpy as np

# Default paths
DATASET_DIR = Path("../datasets/MNIST")
BUILD_SCRIPT = "nlsh_build.py"
SEARCH_SCRIPT = "nlsh_search_adaptive.py"  # Use adaptive version
SEARCH_BIN = "../bin/search"
RUNS_DIR = Path("./test_runs1")

class ParameterTester:
    """Comprehensive parameter testing for Neural LSH"""
    
    def __init__(self, args):
        self.args = args
        self.results = []
        self.test_configs = []
        
        # Default parameter ranges based on impact analysis
        self.param_ranges = {
            # High impact parameters (test thoroughly)
            "T_values": [5, 10, 15, 20, 25],
            "m_values": [100, 150, 200, 250, 300],  # For MNIST 60K
            
            # Medium impact parameters  
            "knn_values": [10, 15, 20, 25],
            "epochs_values": [25, 50, 75],
            "layers_values": [2, 3, 4],
            "nodes_values": [64, 128, 192],
            
            # Lower impact parameters
            "weight_decay_values": [0.0, 1e-5, 1e-4],
            "imbalance_values": [0.01, 0.03, 0.05],
            
            # Search parameters
            "N_values": [1, 5, 10],
            
            # Adaptive parameters (if using adaptive search)
            "max_T_values": [15, 20, 25, 30],
            "target_recall_proxy_values": [0.7, 0.8, 0.9],
            "confidence_threshold_values": [0.90, 0.95, 0.98]
        }
        
        # Adjust ranges for dataset size
        if args.type == "sift":
            # SIFT typically has 1M vectors, need different ranges
            self.param_ranges["m_values"] = [500, 750, 1000, 1250, 1500]
            self.param_ranges["knn_values"] = [15, 20, 30, 40]
    
    def generate_test_configs(self):
        """Generate test configurations based on test mode"""
        configs = []
        
        if self.args.test_mode == "full":
            configs = self._generate_full_sweep()
        elif self.args.test_mode == "high_impact":
            configs = self._generate_high_impact_tests()
        elif self.args.test_mode == "adaptive_comparison":
            configs = self._generate_adaptive_comparison()
        elif self.args.test_mode == "quick":
            configs = self._generate_quick_tests()
        elif self.args.test_mode == "custom":
            configs = self._generate_custom_tests()
        
        print(f"Generated {len(configs)} test configurations for {self.args.test_mode} mode")
        return configs
    
    def _generate_high_impact_tests(self):
        """Test high impact parameters: T and m"""
        configs = []
        base_config = self._get_base_config()
        
        # Test T parameter sweep (most important)
        print("Generating T parameter sweep...")
        for T in self.param_ranges["T_values"]:
            config = base_config.copy()
            config.update({
                "T": T,
                "test_name": f"T_sweep_T{T}",
                "description": f"Testing T={T}"
            })
            configs.append(config)
        
        # Test m parameter sweep (second most important)  
        print("Generating m parameter sweep...")
        for m in self.param_ranges["m_values"]:
            config = base_config.copy()
            config.update({
                "m": m,
                "test_name": f"m_sweep_m{m}",
                "description": f"Testing m={m}"
            })
            configs.append(config)
        
        # Test combined high-impact (T=15, different m values)
        print("Generating combined high-impact tests...")
        for m in [150, 200, 250]:
            for T in [10, 15, 20]:
                config = base_config.copy()
                config.update({
                    "m": m,
                    "T": T,
                    "test_name": f"combined_m{m}_T{T}",
                    "description": f"Combined m={m}, T={T}"
                })
                configs.append(config)
        
        return configs
    
    def _generate_adaptive_comparison(self):
        """Compare regular vs adaptive search"""
        configs = []
        base_config = self._get_base_config()
        
        # Regular search with different T values
        for T in [5, 10, 15, 20, 25]:
            config = base_config.copy()
            config.update({
                "T": T,
                "adaptive": False,
                "test_name": f"regular_T{T}",
                "description": f"Regular search T={T}"
            })
            configs.append(config)
        
        # Adaptive search with different max_T values
        for max_T in [15, 20, 25, 30]:
            for target_proxy in [0.8, 0.85, 0.9]:
                config = base_config.copy()
                config.update({
                    "adaptive": True,
                    "max_T": max_T,
                    "target_recall_proxy": target_proxy,
                    "test_name": f"adaptive_maxT{max_T}_proxy{target_proxy}",
                    "description": f"Adaptive max_T={max_T}, proxy={target_proxy}"
                })
                configs.append(config)
        
        return configs
    
    def _generate_quick_tests(self):
        """Quick test with representative parameters"""
        configs = []
        base_config = self._get_base_config()
        
        # Test a few key configurations
        test_params = [
            {"T": 5, "m": 150, "description": "Conservative"},
            {"T": 10, "m": 200, "description": "Default"}, 
            {"T": 15, "m": 200, "description": "Higher T"},
            {"T": 20, "m": 250, "description": "Aggressive"}
        ]
        
        for params in test_params:
            config = base_config.copy()
            config.update(params)
            config["test_name"] = f"quick_{params['description'].lower()}"
            configs.append(config)
        
        # Add one adaptive test
        config = base_config.copy()
        config.update({
            "adaptive": True,
            "max_T": 20,
            "test_name": "quick_adaptive",
            "description": "Quick adaptive test"
        })
        configs.append(config)
        
        return configs
    
    def _generate_full_sweep(self):
        """Full combinatorial sweep (limited to prevent explosion)"""
        configs = []
        base_config = self._get_base_config()
        
        # Limit full sweep to avoid too many combinations
        limited_ranges = {
            "T": [10, 15, 20],
            "m": [150, 200, 250],
            "knn": [15, 20],
            "epochs": [25, 50],
            "layers": [2, 3],
            "nodes": [64, 128]
        }
        
        print("Generating full parameter sweep (limited combinations)...")
        count = 0
        for params in product(*limited_ranges.values()):
            if count >= 50:  # Limit to prevent excessive testing
                break
                
            config = base_config.copy()
            config.update(dict(zip(limited_ranges.keys(), params)))
            config["test_name"] = f"full_{count:03d}"
            config["description"] = f"Full sweep {count}"
            configs.append(config)
            count += 1
        
        return configs
    
    def _generate_custom_tests(self):
        """Custom test configurations"""
        configs = []
        
        # Add your custom configurations here
        custom_configs = [
            {
                "m": 200, "T": 15, "knn": 20, "epochs": 50,
                "test_name": "custom_baseline",
                "description": "Custom baseline configuration"
            },
            {
                "m": 250, "T": 20, "knn": 25, "epochs": 75, "nodes": 128,
                "test_name": "custom_optimized", 
                "description": "Custom optimized configuration"
            }
        ]
        
        base_config = self._get_base_config()
        for custom in custom_configs:
            config = base_config.copy()
            config.update(custom)
            configs.append(config)
        
        return configs
    
    def _get_base_config(self):
        """Get base configuration with default values"""
        return {
            # Build parameters
            "knn": 20,
            "m": 200,
            "epochs": 50,
            "layers": 3,
            "nodes": 128,
            "weight_decay": 0.0,
            "imbalance": 0.03,
            
            # Search parameters
            "N": 5,
            "T": 15,
            "max_queries": self.args.max_queries,
            
            # Adaptive parameters
            "adaptive": False,
            "max_T": 20,
            "min_T": 1,
            "target_recall_proxy": 0.8,
            "confidence_threshold": 0.95,
            
            # Test metadata
            "dataset_dir": self.args.dataset_dir,
            "type": self.args.type,
        }
    
    def run_single_test(self, config):
        """Run a single test configuration"""
        test_name = config["test_name"]
        print(f"\n{'='*60}")
        print(f"Running test: {test_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Create unique index directory
        index_dir = f"test_index_{test_name}"
        output_file = f"results_{test_name}.txt"
        
        start_time = time.time()
        
        try:
            # Build phase
            if not self.args.skip_build:
                self._run_build_phase(config, index_dir)
            
            # Search phase  
            if not self.args.skip_search:
                recall, qps, avg_af, search_time = self._run_search_phase(config, index_dir, output_file)
                
                # Record results
                result = {
                    "test_name": test_name,
                    "config": config,
                    "recall": recall,
                    "qps": qps,
                    "avg_af": avg_af,
                    "search_time": search_time,
                    "total_time": time.time() - start_time,
                    "output_file": output_file
                }
                
                # Add adaptive-specific results if available
                if config.get("adaptive"):
                    adaptive_stats = self._extract_adaptive_stats(output_file)
                    result.update(adaptive_stats)
                
                self.results.append(result)
                print(f"✔ Test {test_name} completed - Recall: {recall:.2f}%, QPS: {qps:.2f}")
            
        except Exception as e:
            print(f"✗ Test {test_name} failed: {e}")
            result = {
                "test_name": test_name,
                "config": config,
                "error": str(e),
                "total_time": time.time() - start_time
            }
            self.results.append(result)
    
    def _run_build_phase(self, config, index_dir):
        """Run build phase with given configuration"""
        dataset_file = Path(config["dataset_dir"]) / "train-images.idx3-ubyte"
        
        cmd = [
            sys.executable, BUILD_SCRIPT,
            "-d", str(dataset_file),
            "-i", index_dir,
            "-type", config["type"],
            "--knn", str(config["knn"]),
            "-m", str(config["m"]),
            "--imbalance", str(config["imbalance"]),
            "--epochs", str(config["epochs"]),
            "--layers", str(config["layers"]),
            "--nodes", str(config["nodes"]),
            "--weight_decay", str(config["weight_decay"]),
            "--calculated_output", "../out_ivfflat.txt",
            "--knn-graph-file","./test_index_111/knn_graph.npy"
        ]
        
        if self.args.search_path:
            cmd += ["--search_path", self.args.search_path]
        
        print(f"Build command: {' '.join(cmd)}")
        if not self.args.dry_run:
            subprocess.run(cmd, check=True)
    
    def _run_search_phase(self, config, index_dir, output_file):
        """Run search phase and extract metrics"""
        dataset_file = Path(config["dataset_dir"]) / "train-images.idx3-ubyte"
        query_file = Path(config["dataset_dir"]) / "t10k-images.idx3-ubyte"
        
        cmd = [
            sys.executable, SEARCH_SCRIPT,
            "-d", str(dataset_file),
            "-q", str(query_file),
            "-i", index_dir,
            "-o", output_file,
            "-type", config["type"],
            "-N", str(config["N"]),
            "-range", "false"
        ]
        
        if config.get("adaptive"):
            cmd.extend([
                "--adaptive",
                "--max_T", str(config["max_T"]),
                "--min_T", str(config["min_T"]),
                "--target_recall_proxy", str(config["target_recall_proxy"]),
                "--confidence_threshold", str(config["confidence_threshold"])
            ])
        else:
            cmd.extend(["-T", str(config["T"])])
        
        if config.get("max_queries"):
            cmd += ["--max-queries", str(config["max_queries"])]
        
        print(f"Search command: {' '.join(cmd)}")
        if self.args.dry_run:
            return 0.0, 0.0, 0.0, 0.0
        
        subprocess.run(cmd, check=True)
        
        # Extract metrics from output file
        return self._extract_metrics(output_file)
    
    def _extract_metrics(self, output_file):
        """Extract metrics from search output file"""
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Extract metrics using string parsing
            recall = 0.0
            qps = 0.0
            avg_af = 0.0
            search_time = 0.0
            
            for line in content.split('\n'):
                if line.startswith("Recall@"):
                    recall = float(line.split(':')[1].strip()) * 100
                elif line.startswith("QPS:"):
                    qps = float(line.split(':')[1].strip())
                elif line.startswith("Average AF:"):
                    avg_af = float(line.split(':')[1].strip())
                elif line.startswith("tApproximateAverage:"):
                    search_time = float(line.split(':')[1].strip())
            
            return recall, qps, avg_af, search_time
            
        except Exception as e:
            print(f"Warning: Could not extract metrics from {output_file}: {e}")
            return 0.0, 0.0, 0.0, 0.0
    
    def _extract_adaptive_stats(self, output_file):
        """Extract adaptive-specific statistics"""
        adaptive_stats = {}
        
        try:
            with open(output_file, 'r') as f:
                content = f.read()
            
            for line in content.split('\n'):
                if line.startswith("avgTUsed:"):
                    adaptive_stats["avg_T_used"] = float(line.split(':')[1].strip())
                elif line.startswith("avgCandidates:"):
                    adaptive_stats["avg_candidates"] = float(line.split(':')[1].strip())
                elif line.startswith("avgCumulativeProb:"):
                    adaptive_stats["avg_cumulative_prob"] = float(line.split(':')[1].strip())
                    
        except Exception as e:
            print(f"Warning: Could not extract adaptive stats: {e}")
        
        return adaptive_stats
    
    def run_all_tests(self):
        """Run all test configurations"""
        configs = self.generate_test_configs()
        
        print(f"\nStarting {len(configs)} tests...")
        start_time = time.time()
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Running test: {config['test_name']}")
            if self.args.max_test_time and (time.time() - start_time) > self.args.max_test_time:
                print(f"Stopping tests - exceeded max time limit of {self.args.max_test_time} seconds")
                break
                
            self.run_single_test(config)
        
        total_time = time.time() - start_time
        print(f"\nAll tests completed in {total_time:.2f} seconds")
        
        # Generate reports
        self.generate_reports()
    
    def generate_reports(self):
        """Generate comprehensive test reports"""
        if not self.results:
            print("No results to report")
            return
        
        # Create reports directory
        reports_dir = Path("test_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Generate CSV report
        self._generate_csv_report(reports_dir)
        
        # Generate summary report
        self._generate_summary_report(reports_dir)
        
        # Generate best configurations report
        self._generate_best_configs_report(reports_dir)
        
        print(f"\nReports generated in {reports_dir}/")
    
    def _generate_csv_report(self, reports_dir):
        """Generate detailed CSV report"""
        csv_file = reports_dir / "detailed_results.csv"
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["test_name", "recall_%", "qps", "avg_af", "search_time", 
                     "total_time", "T", "m", "knn", "epochs", "layers", "nodes", 
                     "weight_decay", "adaptive"]
            
            # Add adaptive columns if any adaptive tests
            has_adaptive = any(r.get("avg_T_used") is not None for r in self.results)
            if has_adaptive:
                header.extend(["avg_T_used", "avg_candidates", "avg_cumulative_prob"])
            
            writer.writerow(header)
            
            # Data rows
            for result in self.results:
                if "error" in result:
                    continue
                    
                config = result["config"]
                row = [
                    result["test_name"],
                    result["recall"], 
                    result["qps"],
                    result["avg_af"],
                    result["search_time"],
                    result["total_time"],
                    config.get("T", "N/A"),
                    config["m"],
                    config["knn"],
                    config["epochs"],
                    config["layers"], 
                    config["nodes"],
                    config["weight_decay"],
                    config.get("adaptive", False)
                ]
                
                if has_adaptive:
                    row.extend([
                        result.get("avg_T_used", "N/A"),
                        result.get("avg_candidates", "N/A"),
                        result.get("avg_cumulative_prob", "N/A")
                    ])
                
                writer.writerow(row)
        
        print(f"Detailed CSV report: {csv_file}")
    
    def _generate_summary_report(self, reports_dir):
        """Generate summary text report"""
        summary_file = reports_dir / "summary_report.txt"
        
        with open(summary_file, 'w') as f:
            f.write("NEURAL LSH PARAMETER TEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            successful_results = [r for r in self.results if "error" not in r]
            
            if not successful_results:
                f.write("No successful tests to summarize.\n")
                return
            
            # Overall statistics
            f.write(f"Total tests run: {len(self.results)}\n")
            f.write(f"Successful tests: {len(successful_results)}\n")
            f.write(f"Failed tests: {len(self.results) - len(successful_results)}\n\n")
            
            # Best configurations
            best_recall = max(successful_results, key=lambda x: x["recall"])
            best_qps = max(successful_results, key=lambda x: x["qps"])
            best_balance = min(successful_results, key=lambda x: abs(x["recall"] - 50) + abs(x["qps"] - 10))
            
            f.write("BEST CONFIGURATIONS:\n")
            f.write("-"*20 + "\n")
            f.write(f"Best Recall: {best_recall['test_name']} - {best_recall['recall']:.2f}%\n")
            f.write(f"Best QPS: {best_qps['test_name']} - {best_qps['qps']:.2f}\n")
            f.write(f"Best Balance: {best_balance['test_name']}\n\n")
            
            # Parameter impact analysis
            f.write("PARAMETER IMPACT ANALYSIS:\n")
            f.write("-"*25 + "\n")
            
            # Group by T values
            if any("T" in r["config"] for r in successful_results):
                f.write("\nT Parameter Impact:\n")
                t_groups = {}
                for result in successful_results:
                    if "T" in result["config"]:
                        t_val = result["config"]["T"]
                        if t_val not in t_groups:
                            t_groups[t_val] = []
                        t_groups[t_val].append(result)
                
                for t_val in sorted(t_groups.keys()):
                    avg_recall = np.mean([r["recall"] for r in t_groups[t_val]])
                    avg_qps = np.mean([r["qps"] for r in t_groups[t_val]])
                    f.write(f"  T={t_val}: Avg Recall={avg_recall:.1f}%, Avg QPS={avg_qps:.1f}\n")
            
            # Group by m values
            f.write("\nm Parameter Impact:\n")
            m_groups = {}
            for result in successful_results:
                m_val = result["config"]["m"]
                if m_val not in m_groups:
                    m_groups[m_val] = []
                m_groups[m_val].append(result)
            
            for m_val in sorted(m_groups.keys()):
                avg_recall = np.mean([r["recall"] for r in m_groups[m_val]])
                avg_qps = np.mean([r["qps"] for r in m_groups[m_val]])
                f.write(f"  m={m_val}: Avg Recall={avg_recall:.1f}%, Avg QPS={avg_qps:.1f}\n")
        
        print(f"Summary report: {summary_file}")
    
    def _generate_best_configs_report(self, reports_dir):
        """Generate best configurations JSON report"""
        best_configs_file = reports_dir / "best_configurations.json"
        
        successful_results = [r for r in self.results if "error" not in r]
        if not successful_results:
            return
        
        # Find best configurations for different criteria
        best_configs = {
            "best_recall": max(successful_results, key=lambda x: x["recall"]),
            "best_qps": max(successful_results, key=lambda x: x["qps"]),
            "best_af": min(successful_results, key=lambda x: x["avg_af"]),
            "best_overall": max(successful_results, key=lambda x: x["recall"] * x["qps"]),
        }
        
        # Make serializable
        for key, result in best_configs.items():
            # Convert numpy types to regular Python types
            for metric_key, value in result.items():
                if isinstance(value, (np.integer, np.floating)):
                    result[metric_key] = float(value)
        
        with open(best_configs_file, 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        print(f"Best configurations: {best_configs_file}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Neural LSH Comprehensive Parameter Testing")
    
    # Original arguments
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR), help="Dataset directory")
    parser.add_argument("--type", default="mnist", choices=["mnist", "sift"], help="Dataset type")
    parser.add_argument("--search-path", default=SEARCH_BIN, help="Path to Project1 search binary")
    parser.add_argument("--max-queries", type=int, default=1000, help="Max queries to test")
    
    # Test control arguments
    parser.add_argument("--full-test", action="store_true", help="Run comprehensive parameter testing")
    parser.add_argument("--test-mode", choices=["quick", "high_impact", "adaptive_comparison", "full", "custom"], 
                       default="quick", help="Test mode to run")
    parser.add_argument("--max-test-time", type=int, help="Maximum total testing time in seconds")
    parser.add_argument("--skip-build", action="store_true", help="Skip build phases")
    parser.add_argument("--skip-search", action="store_true", help="Skip search phases")
    parser.add_argument("--dry-run", action="store_true", help="Print commands but don't run")
    
    # Single test arguments (original functionality)
    parser.add_argument("--single-test", action="store_true", help="Run single test with specified parameters")
    parser.add_argument("--index-name", help="Index directory name")
    parser.add_argument("--knn", type=int, default=20, help="k for k-NN graph")
    parser.add_argument("--m", type=int, default=200, help="Number of partitions")
    parser.add_argument("--imbalance", type=float, default=0.03, help="KaHIP imbalance")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--layers", type=int, default=3, help="MLP layers")
    parser.add_argument("--nodes", type=int, default=128, help="Nodes per layer")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--N", type=int, default=5, help="Search N")
    parser.add_argument("--T", type=int, default=15, help="Search T")
    parser.add_argument("--out-file", help="Output file")
    
    return parser.parse_args()


def run_single_test(args):
    """Run original single test functionality"""
    print("Running single test with specified parameters...")
    
    # Generate index name if not provided
    if args.index_name is None:
        args.index_name = f"index_knn{args.knn}_m{args.m}_ep{args.epochs}"
    
    if args.out_file is None:
        args.out_file = f"{args.index_name}.txt"
    
    # Create a mock config for compatibility
    config = {
        "dataset_dir": args.dataset_dir,
        "type": args.type,
        "knn": args.knn,
        "m": args.m,
        "epochs": args.epochs,
        "layers": args.layers,
        "nodes": args.nodes,
        "weight_decay": args.weight_decay,
        "imbalance": args.imbalance,
        "N": args.N,
        "T": args.T,
        "max_queries": args.max_queries,
        "adaptive": False
    }
    
    start_time = time.time()
    
    # Run build
    if not args.skip_build:
        dataset_file = Path(args.dataset_dir) / "train-images.idx3-ubyte"
        
        cmd = [
            sys.executable, BUILD_SCRIPT,
            "-d", str(dataset_file),
            "-i", args.index_name,
            "-type", args.type,
            "--knn", str(args.knn),
            "-m", str(args.m),
            "--imbalance", str(args.imbalance),
            "--epochs", str(args.epochs),
            "--layers", str(args.layers),
            "--nodes", str(args.nodes),
            "--weight_decay", str(args.weight_decay)
        ]
        
        if args.search_path:
            cmd += ["--search_path", args.search_path]
        
        print("Build command:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    
    # Run search
    if not args.skip_search:
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
            "-range", "false"
        ]
        
        if args.max_queries:
            cmd += ["--max-queries", str(args.max_queries)]
        
        print("Search command:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
    
    total_time = time.time() - start_time
    print(f"Single test completed in {total_time:.2f} seconds")


def main():
    start = time.time()
    args = parse_args()
    
    print("="*70)
    print("    NEURAL LSH COMPREHENSIVE PARAMETER TESTING SUITE")
    print("="*70)
    
    if args.full_test or not args.single_test:
        # Run comprehensive testing
        tester = ParameterTester(args)
        tester.run_all_tests()
    else:
        # Run single test (original functionality)
        run_single_test(args)
    
    total_time = time.time() - start
    print(f"\nTotal execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()