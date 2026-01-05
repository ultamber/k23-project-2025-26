#!/usr/bin/env python3
"""
Convert BLAST tabular output to pickle format for protein_search.py
FIXED VERSION: Better ID matching

Usage:
    python convert_blast_fixed.py \
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


def create_id_mapping(ids_list):
    """
    Create multiple mappings for robust ID matching.
    
    Returns:
        dict: {accession: index, full_id: index}
    """
    mapping = {}
    
    for i, protein_id in enumerate(ids_list):
        # Map full ID
        mapping[protein_id] = i
        
        # Map accession
        accession = get_accession(protein_id)
        mapping[accession] = i
        
        # Map without trailing whitespace/newlines
        mapping[protein_id.strip()] = i
        mapping[accession.strip()] = i
    
    return mapping


def parse_blast_results(blast_file, database_ids, query_ids, N=50):
    """
    Parse BLAST tabular output and convert to indices.
    FIXED: Better ID matching with multiple strategies.
    
    Returns:
        dict: {
            'blast_results_ids': {query_id: [(hit_id, score, evalue), ...]},
            'blast_results_indices': {query_idx: [(hit_idx, score, evalue), ...]},
            'params': {'N': N, 'evalue': 0.01}
        }
    """
    # Create ID to index mappings
    print("    Creating ID mappings...")
    db_mapping = create_id_mapping(database_ids)
    query_mapping = create_id_mapping(query_ids)
    
    print(f"    Database mapping: {len(db_mapping)} entries")
    print(f"    Query mapping: {len(query_mapping)} entries")
    
    # Parse BLAST results
    blast_results_ids = defaultdict(list)
    blast_results_indices = defaultdict(list)
    
    matches_found = 0
    query_misses = set()
    db_misses = set()
    
    with open(blast_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
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
            
            # Try multiple matching strategies
            query_idx = None
            hit_idx = None
            
            # Strategy 1: Direct match
            if query_id in query_mapping:
                query_idx = query_mapping[query_id]
            # Strategy 2: Accession match
            else:
                query_acc = get_accession(query_id)
                if query_acc in query_mapping:
                    query_idx = query_mapping[query_acc]
                else:
                    query_misses.add(query_id)
            
            # Same for hit
            if hit_id in db_mapping:
                hit_idx = db_mapping[hit_id]
            else:
                hit_acc = get_accession(hit_id)
                if hit_acc in db_mapping:
                    hit_idx = db_mapping[hit_acc]
                else:
                    db_misses.add(hit_id)
            
            # If both matched, store
            if query_idx is not None and hit_idx is not None:
                blast_results_indices[query_idx].append((hit_idx, bitscore, evalue))
                matches_found += 1
    
    print(f"    Parsed {line_num} BLAST lines")
    print(f"    Matched {matches_found} alignments")
    
    if query_misses:
        print(f"    ⚠️  {len(query_misses)} unique query IDs not found")
        print(f"       Examples: {list(query_misses)[:3]}")
    
    if db_misses:
        print(f"    ⚠️  {len(db_misses)} unique hit IDs not found")
        print(f"       Examples: {list(db_misses)[:3]}")
    
    return {
        'blast_results_ids': dict(blast_results_ids),
        'blast_results_indices': dict(blast_results_indices),
        'params': {'N': N, 'evalue': 0.01}
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert BLAST results to pickle format (FIXED VERSION)'
    )
    parser.add_argument('--blast-file', required=True, help='BLAST results (tabular)')
    parser.add_argument('--database-ids', required=True, help='Database .ids file')
    parser.add_argument('--query-ids', required=True, help='Query .ids file')
    parser.add_argument('--output', default='blast_ground_truth.pkl', help='Output pickle file')
    parser.add_argument('--N', type=int, default=50, help='Top-N (default: 50)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Converting BLAST Results to Pickle Format (FIXED)")
    print("="*70)
    
    # Load IDs
    print(f"\n[1] Loading protein IDs...")
    database_ids = load_ids(args.database_ids)
    query_ids = load_ids(args.query_ids)
    print(f"    Database: {len(database_ids)} proteins")
    print(f"    Queries: {len(query_ids)} proteins")
    print(f"    Example DB ID: {database_ids[0]}")
    print(f"    Example Query ID: {query_ids[0]}")
    
    # Parse BLAST results
    print(f"\n[2] Parsing BLAST results...")
    results = parse_blast_results(
        args.blast_file,
        database_ids,
        query_ids,
        args.N
    )
    print(f"    Found {len(results['blast_results_ids'])} queries with hits")
    print(f"    Converted {len(results['blast_results_indices'])} to indices")
    
    if len(results['blast_results_indices']) == 0:
        print("\n    ❌ ERROR: No indices matched!")
        print("    This means BLAST IDs don't match your database/query IDs.")
        print("\n    Possible causes:")
        print("    1. Wrong database used for BLAST vs embeddings")
        print("    2. Different FASTA files")
        print("    3. ID format mismatch")
        print("\n    Run debug_blast_ids.py for detailed diagnosis!")
        return
    
    # Save to pickle
    print(f"\n[3] Saving to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    print(f"    Saved!")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)
    print(f"\nConverted {len(results['blast_results_indices'])} queries successfully!")
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