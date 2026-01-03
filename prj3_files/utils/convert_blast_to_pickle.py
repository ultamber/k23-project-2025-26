#!/usr/bin/env python3
"""
Convert BLAST tabular output to pickle format for protein_search.py

Usage:
    python convert_blast_to_pickle.py \
        --blast-file blast_results.txt \
        --database-ids output.ids \
        --query-ids query_vectors.ids \
        --output blast_ground_truth.pkl
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def load_ids(ids_file):
    """Load protein IDs from .ids file."""
    with open(ids_file, 'r') as f:
        ids = [line.strip() for line in f]
    return ids


def get_accession(protein_id):
    """Extract accession from UniProt ID (e.g., sp|P12345|NAME → P12345)."""
    if '|' in protein_id:
        parts = protein_id.split('|')
        if len(parts) >= 2:
            return parts[1]
    return protein_id.split()[0]


def parse_blast_results(blast_file, database_ids, query_ids, N=50):
    """
    Parse BLAST tabular output and convert to indices.
    
    Returns:
        dict: {
            'blast_results_ids': {query_id: [(hit_id, score, evalue), ...]},
            'blast_results_indices': {query_idx: [(hit_idx, score, evalue), ...]},
            'params': {'N': N, 'evalue': 0.01}
        }
    """
    # Create ID to index mappings
    db_id_to_idx = {get_accession(pid): i for i, pid in enumerate(database_ids)}
    query_id_to_idx = {get_accession(pid): i for i, pid in enumerate(query_ids)}
    
    # Parse BLAST results
    blast_results_ids = defaultdict(list)
    blast_results_indices = defaultdict(list)
    
    with open(blast_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 12:
                continue
            
            query_id = parts[0]
            hit_id = parts[1]
            pident = float(parts[2])
            evalue = float(parts[10])
            bitscore = float(parts[11])
            
            # Store by ID
            blast_results_ids[query_id].append((hit_id, bitscore, evalue))
            
            # Convert to indices
            query_acc = get_accession(query_id)
            hit_acc = get_accession(hit_id)
            
            if query_acc in query_id_to_idx and hit_acc in db_id_to_idx:
                query_idx = query_id_to_idx[query_acc]
                hit_idx = db_id_to_idx[hit_acc]
                blast_results_indices[query_idx].append((hit_idx, bitscore, evalue))
    
    return {
        'blast_results_ids': dict(blast_results_ids),
        'blast_results_indices': dict(blast_results_indices),
        'params': {'N': N, 'evalue': 0.01}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert BLAST results to pickle format'
    )
    parser.add_argument('--blast-file', required=True, help='BLAST results (tabular)')
    parser.add_argument('--database-ids', required=True, help='Database .ids file')
    parser.add_argument('--query-ids', required=True, help='Query .ids file')
    parser.add_argument('--output', default='blast_ground_truth.pkl', help='Output pickle file')
    parser.add_argument('--N', type=int, default=50, help='Top-N (default: 50)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Converting BLAST Results to Pickle Format")
    print("="*70)
    
    # Load IDs
    print(f"\n[1] Loading protein IDs...")
    database_ids = load_ids(args.database_ids)
    query_ids = load_ids(args.query_ids)
    print(f"    Database: {len(database_ids)} proteins")
    print(f"    Queries: {len(query_ids)} proteins")
    
    # Parse BLAST results
    print(f"\n[2] Parsing BLAST results...")
    results = parse_blast_results(
        args.blast_file,
        database_ids,
        query_ids,
        args.N
    )
    print(f"    ✓ Found {len(results['blast_results_ids'])} queries with hits")
    print(f"    ✓ Converted {len(results['blast_results_indices'])} to indices")
    
    # Save to pickle
    print(f"\n[3] Saving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"    ✓ Saved!")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nNow run:")
    print(f"  python protein_search.py \\")
    print(f"      -d output.npy \\")
    print(f"      -q query_vectors.npy \\")
    print(f"      -o results_with_blast.txt \\")
    print(f"      -method all \\")
    print(f"      --ground-truth {args.output}")
    print()


if __name__ == '__main__':
    main()