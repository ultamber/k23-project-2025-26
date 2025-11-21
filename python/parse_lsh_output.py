#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_lsh_output.py
-------------------
Parse LSH search output file and extract k-NN graph.

Usage:
    python parse_lsh_output.py -i lsh_output.txt -o knn_graph.npy -n 60000 -k 10
"""

import argparse
import numpy as np
import re
from pathlib import Path


def parse_lsh_output(input_file, n_queries, k):
    """
    Parse LSH output file to extract k-NN graph.
    
    Args:
        input_file: Path to LSH output file
        n_queries: Expected number of queries
        k: Number of neighbors per query
        
    Returns:
        knn_graph: numpy array (n_queries, k) with neighbor indices
    """
    print(f"Parsing LSH output from {input_file}...")
    print(f"Expected: {n_queries} queries, {k} neighbors each")
    
    knn_graph = np.zeros((n_queries, k), dtype=np.int32)
    knn_graph.fill(-1)  # Fill with -1 to detect missing neighbors
    
    current_query = -1
    neighbor_count = 0
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and method header
            if not line or line == "LSH":
                continue
            
            # Parse query ID
            if line.startswith("Query:"):
                match = re.search(r"Query:\s*(\d+)", line)
                if match:
                    current_query = int(match.group(1))
                    neighbor_count = 0
                    
                    # Progress indicator
                    if (current_query + 1) % max(1, n_queries // 10) == 0:
                        print(f"  Progress: {current_query + 1}/{n_queries} "
                              f"({100*(current_query+1)/n_queries:.1f}%)")
                continue
            
            # Parse nearest neighbor
            if line.startswith("Nearest neighbor-"):
                match = re.search(r"Nearest neighbor-\d+:\s*(\d+)", line)
                if match and current_query >= 0 and neighbor_count < k:
                    neighbor_idx = int(match.group(1))
                    knn_graph[current_query, neighbor_count] = neighbor_idx
                    neighbor_count += 1
                continue
    
    # Validate results
    missing_queries = np.where(knn_graph[:, 0] == -1)[0]
    if len(missing_queries) > 0:
        print(f"\n⚠️  Warning: {len(missing_queries)} queries have no neighbors")
        print(f"   First missing: {missing_queries[:10]}")
    
    incomplete_queries = np.where((knn_graph == -1).any(axis=1))[0]
    if len(incomplete_queries) > 0:
        print(f"⚠️  Warning: {len(incomplete_queries)} queries have < {k} neighbors")
        
        # Fill missing neighbors with self-reference
        for qi in incomplete_queries:
            for j in range(k):
                if knn_graph[qi, j] == -1:
                    knn_graph[qi, j] = qi
    
    print(f"\n✓ Parsing completed")
    print(f"  k-NN graph shape: {knn_graph.shape}")
    print(f"  Unique neighbors: {len(np.unique(knn_graph))}")
    
    return knn_graph


def main():
    parser = argparse.ArgumentParser(
        description="Parse LSH output to k-NN graph",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--input", required=True,
                        help="Input LSH output file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output k-NN graph file (.npy)")
    parser.add_argument("-n", "--n_queries", type=int, required=True,
                        help="Number of queries in dataset")
    parser.add_argument("-k", "--k_neighbors", type=int, required=True,
                        help="Number of neighbors per query")
    
    args = parser.parse_args()
    
    # Parse LSH output
    knn_graph = parse_lsh_output(args.input, args.n_queries, args.k_neighbors)
    
    # Save as numpy array
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving k-NN graph to {output_path}...")
    np.save(output_path, knn_graph)
    print(f"✓ Saved: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    
    # Print statistics
    print(f"\n📊 k-NN Graph Statistics:")
    print(f"  Shape: {knn_graph.shape}")
    print(f"  Data type: {knn_graph.dtype}")
    print(f"  Self-loops: {np.sum(knn_graph == np.arange(args.n_queries)[:, None])}")
    
    # Check for duplicates in neighbors
    unique_per_query = [len(np.unique(knn_graph[i])) for i in range(args.n_queries)]
    avg_unique = np.mean(unique_per_query)
    print(f"  Avg unique neighbors per query: {avg_unique:.2f}/{args.k_neighbors}")


if __name__ == "__main__":
    main()