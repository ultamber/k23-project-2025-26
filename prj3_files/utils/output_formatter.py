#!/usr/bin/env python3

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio.Seq import Seq
from utils.evaluation import compute_sequence_identity

try:
    from utils.remote_homolog_detector import RemoteHomologDetector
    REMOTE_HOMOLOG_AVAILABLE = True
except ImportError:
    REMOTE_HOMOLOG_AVAILABLE = False

def get_blast_identity_from_results(
    query_idx: int,
    neighbor_idx: int,
    blast_results: Optional[Dict]
) -> Optional[float]:
    """
    Get BLAST identity for a specific query-neighbor pair from BLAST results.
    
    Args:
        query_idx: Query index
        neighbor_idx: Neighbor index
        blast_results: BLAST results dict
        
    Returns:
        Identity percentage or None
    """
    if not blast_results or 'blast_results_indices' not in blast_results:
        return None
    
    blast_indices = blast_results['blast_results_indices']
    
    if query_idx not in blast_indices:
        return None
    
    # Look for this neighbor in BLAST results
    for hit_idx, score, evalue in blast_indices[query_idx]:
        if hit_idx == neighbor_idx:
            # We have the hit, but BLAST results don't include identity %
            # We would need to parse the original BLAST output
            # For now, return a flag that it was found
            return -1.0  # Flag: found in BLAST but no identity
    
    return None


def is_in_blast_top_n(
    query_idx: int,
    neighbor_idx: int,
    blast_results: Optional[Dict],
    N: int
) -> Optional[bool]:
    """
    Check if neighbor is in BLAST Top-N for this query.
    
    Args:
        query_idx: Query index
        neighbor_idx: Neighbor index
        blast_results: BLAST results dict
        N: Top-N threshold
        
    Returns:
        True if in Top-N, False if not, None if no BLAST data
    """
    if not blast_results or 'blast_results_indices' not in blast_results:
        return None
    
    blast_indices = blast_results['blast_results_indices']
    
    if query_idx not in blast_indices:
        return None
    
    # Get top-N indices from BLAST
    blast_top_n = [hit_idx for hit_idx, _, _ in blast_indices[query_idx][:N]]
    
    return neighbor_idx in blast_top_n


def format_output_txt(
    results: Dict,
    output_file: str,
    database_ids: Optional[List[str]] = None,
    query_ids: Optional[List[str]] = None,
    database_seqs: Optional[List[str]] = None,
    query_seqs: Optional[List[str]] = None,
    blast_results: Optional[Dict] = None,
    N: int = 50,
    display_n: int = 10,
    enable_bio_comments: bool = True,
    uniprot_cache_dir: Optional[str] = 'uniprot_cache'
):
    """
    Format results to assignment-style text output.
    
    Args:
        results: Results dict with method metrics
        output_file: Output file path
        database_ids: Database protein IDs
        query_ids: Query protein IDs
        database_seqs: Database sequences (for identity calculation)
        query_seqs: Query sequences (for identity calculation)
        blast_results: BLAST ground truth results
        N: Top-N for evaluation (e.g., 50)
        display_n: Number of neighbors to display (e.g., 10)
        enable_bio_comments: Generate biological comments using UniProt
        uniprot_cache_dir: Cache directory for UniProt data
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize remote homolog detector if enabled
    detector = None
    print("enabling bio comments:", enable_bio_comments)
    print("remote homolog available:", REMOTE_HOMOLOG_AVAILABLE)
    if enable_bio_comments and REMOTE_HOMOLOG_AVAILABLE:
        try:
            detector = RemoteHomologDetector(cache_dir=uniprot_cache_dir)
            print("  Bio comments enabled (using UniProt)")
        except Exception as e:
            print(f"Could not initialize bio comments: {e}")
            detector = None
    elif enable_bio_comments:
        print("Bio comments unavailable (missing dependencies)")
    
    with open(output_path, 'w') as f:
        # =====================================================================
        # HEADER
        # =====================================================================
        f.write("="*70 + "\n")
        f.write("PROTEIN REMOTE HOMOLOG DETECTION - RESULTS\n")
        f.write("="*70 + "\n\n")
        
        num_queries = results.get('num_queries', 0)
        
        f.write(f"Total Queries: {num_queries}\n")
        f.write(f"N = {N} (Top-N size for Recall@N evaluation)\n")
        f.write(f"Display N = {display_n} (neighbors shown in tables)\n\n")
        
        # =====================================================================
        # [1] METHOD COMPARISON SUMMARY
        # =====================================================================
        f.write("="*70 + "\n")
        f.write("[1] METHOD COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        # Header
        f.write(f"{'Method':<18s} {'Time/query (s)':<18s} {'QPS':<12s} {'Recall@N vs BLAST':<20s}\n")
        f.write("-"*70 + "\n")
        
        methods = results.get('methods', {})
        method_order = ['lsh', 'hypercube', 'ivfflat', 'ivfpq', 'neural-lsh']
        
        for method_name in method_order:
            if method_name not in methods:
                continue
            
            method_data = methods[method_name]
            time_per_query = method_data.get('avg_query_time', 0)
            qps = method_data.get('qps', 0)
            recall = method_data.get('recall_at_n', 0)
            
            # Format method name for display
            display_name = method_name.upper().replace('-', ' ')
            
            f.write(f"{display_name:<18s} {time_per_query:<18.4f} {qps:<12.2f} {recall:<20.4f}\n")
        
        # Add BLAST reference
        if blast_results:
            blast_params = blast_results.get('params', {})
            blast_time = blast_params.get('total_time', 0)
            
            if num_queries > 0 and blast_time > 0:
                blast_time_per_query = blast_time / num_queries
                blast_qps = num_queries / blast_time
            else:
                blast_time_per_query = 0
                blast_qps = 0
            
            f.write(f"{'BLAST (Reference)':<18s} {blast_time_per_query:<18.4f} {blast_qps:<12.2f} {'1.0000 (defines Top-N)':<20s}\n")
        
        f.write("-"*70 + "\n\n")
        
        # =====================================================================
        # [2] TOP-N NEIGHBORS PER METHOD (First Query)
        # =====================================================================
        f.write("="*70 + "\n")
        f.write(f"[2] TOP-{display_n} NEIGHBORS PER METHOD (First Query Only)\n")
        f.write("="*70 + "\n\n")
        
        # Get first query info
        first_query_id = "Unknown"
        if query_ids and len(query_ids) > 0:
            first_query_id = query_ids[0]
        
        f.write(f"Query Protein: {first_query_id}\n")
        f.write(f"(Showing first query as example. All queries evaluated in Recall@N.)\n\n")
        
        # For each method
        for method_name in method_order:
            if method_name not in methods:
                continue
            
            method_data = methods[method_name]
            method_results = method_data.get('results', [])
            
            if not method_results:
                continue
            
            first_query_results = method_results[0] if method_results else []
            
            # Method header
            display_name = method_name.upper().replace('-', ' ')
            f.write(f"Method: {display_name}\n")
            f.write("-"*100 + "\n")
            
            # Column headers
            f.write(f"{'Rank':<6s} {'Neighbor ID':<20s} {'L2 Dist':<12s} {'BLAST ID (%)':<15s} {'In BLAST Top-N?':<18s} {'Bio Comment'}\n")
            f.write("-"*100 + "\n")
            
            # Neighbors
            for rank, (neighbor_idx, distance) in enumerate(first_query_results[:display_n], 1):
                # Get neighbor ID
                if database_ids and neighbor_idx < len(database_ids):
                    neighbor_id = database_ids[neighbor_idx]
                else:
                    neighbor_id = f"Prot_{neighbor_idx}"
                
                # Compute/retrieve BLAST identity
                blast_identity_str = "N/A"
                if database_seqs and query_seqs and neighbor_idx < len(database_seqs):
                    # Compute sequence identity
                    identity = compute_sequence_identity(
                        query_seqs[0],
                        database_seqs[neighbor_idx]
                    )
                    blast_identity_str = f"{identity:.1f}"
                elif blast_results:
                    # Try to get from BLAST results
                    identity = get_blast_identity_from_results(
                        0, neighbor_idx, blast_results
                    )
                    if identity is not None:
                        if identity < 0:
                            blast_identity_str = "In BLAST"
                        else:
                            blast_identity_str = f"{identity:.1f}"
                
                # Check if in BLAST Top-N
                in_blast_top_n_str = "?"
                in_blast_top_n_bool = None
                if blast_results:
                    in_top_n = is_in_blast_top_n(0, neighbor_idx, blast_results, N)
                    if in_top_n is not None:
                        in_blast_top_n_str = "Yes" if in_top_n else "No"
                        in_blast_top_n_bool = in_top_n
                
                # Generate bio comment
                bio_comment = "--"
                blast_identity_float = None
                
                # Parse BLAST identity
                if blast_identity_str not in ["N/A", "In BLAST"]:
                    try:
                        blast_identity_float = float(blast_identity_str)
                    except:
                        pass
                
                # Use detector if available
                if detector and database_ids:
                    try:
                        # Get query and neighbor IDs (UniProt format)
                        query_uniprot_id = first_query_id.split('|')[1] if '|' in first_query_id else first_query_id
                        neighbor_uniprot_id = neighbor_id.split('|')[1] if '|' in neighbor_id else neighbor_id
                        
                        # Quick validation
                        validation = detector.validate_remote_homolog(
                            query_uniprot_id,
                            neighbor_uniprot_id,
                            distance,
                            blast_identity_float
                        )
                        
                        # Generate comment
                        bio_comment = detector.generate_bio_comment(
                            distance,
                            blast_identity_float,
                            in_blast_top_n_bool,
                            validation
                        )
                    except Exception as e:
                        # Fallback to simple logic
                        if blast_identity_float is not None and blast_identity_float < 30:
                            bio_comment = "Remote homolog?"
                else:
                    # Simple logic without detector
                    if blast_identity_float is not None and blast_identity_float < 30:
                        bio_comment = "Remote homolog?"
                
                f.write(f"{rank:<6d} {neighbor_id:<20s} {distance:<12.4f} {blast_identity_str:<15s} {in_blast_top_n_str:<18s} {bio_comment}\n")
            
            f.write("\n")
        
        # =====================================================================
        # FOOTER
        # =====================================================================
        f.write("="*70 + "\n")
        f.write("NOTES:\n")
        f.write("  - BLAST ID (%): Sequence identity percentage\n")
        f.write("  - Remote homologs typically have <30% identity but similar function\n")
        f.write("  - Bio comments require UniProt annotation (future enhancement)\n")
        f.write("="*70 + "\n\n")
        
        f.write("="*70 + "\n")
        f.write("END OF RESULTS\n")
        f.write("="*70 + "\n")


def format_detailed_query_output(
    query_idx: int,
    query_id: str,
    query_seq: Optional[str],
    ann_neighbors: List[Tuple[int, float]],
    database_ids: List[str],
    database_seqs: Optional[List[str]],
    blast_results: Optional[Dict],
    method_name: str,
    N: int = 50,
    display_n: int = 10
) -> str:
    """
    Format detailed output for a single query.
    
    Returns formatted string for inclusion in report.
    """
    lines = []
    
    lines.append(f"\n{'='*70}")
    lines.append(f"Query: {query_id} (Index {query_idx})")
    lines.append(f"Method: {method_name.upper()}")
    lines.append(f"{'='*70}\n")
    
    # Table header
    lines.append(f"{'Rank':<6s} {'Neighbor ID':<20s} {'L2 Dist':<12s} {'Identity (%)':<13s} {'In BLAST?':<12s} {'Comment'}")
    lines.append("-"*90)
    
    # Neighbors
    for rank, (neighbor_idx, distance) in enumerate(ann_neighbors[:display_n], 1):
        # Get neighbor ID
        if neighbor_idx < len(database_ids):
            neighbor_id = database_ids[neighbor_idx]
        else:
            neighbor_id = f"Prot_{neighbor_idx}"
        
        # Compute identity
        identity_str = "N/A"
        if query_seq and database_seqs and neighbor_idx < len(database_seqs):
            identity = compute_sequence_identity(query_seq, database_seqs[neighbor_idx])
            identity_str = f"{identity:.1f}"
        
        # Check BLAST
        in_blast = is_in_blast_top_n(query_idx, neighbor_idx, blast_results, N)
        in_blast_str = "Yes" if in_blast else "No" if in_blast is not None else "?"
        
        # Comment
        comment = "--"
        if identity_str != "N/A":
            try:
                id_val = float(identity_str)
                if id_val < 30 and distance < 10:
                    comment = "Remote homolog candidate"
                elif id_val < 30:
                    comment = "Low identity"
                elif id_val > 70:
                    comment = "High identity"
            except:
                pass
        
        lines.append(f"{rank:<6d} {neighbor_id:<20s} {distance:<12.4f} {identity_str:<13s} {in_blast_str:<12s} {comment}")
    
    return "\n".join(lines)


if __name__ == '__main__':
    """Test the formatter with dummy data."""
    
    # Dummy results
    results = {
        'N': 50,
        'num_queries': 100,
        'methods': {
            'lsh': {
                'avg_query_time': 0.0085,
                'qps': 117.65,
                'recall_at_n': 0.8521,
                'results': [
                    [(10, 0.15), (25, 0.18), (30, 0.20), (40, 0.22), (50, 0.25)],
                    # More queries...
                ]
            },
            'hypercube': {
                'avg_query_time': 0.0070,
                'qps': 142.86,
                'recall_at_n': 0.8234,
                'results': [
                    [(12, 0.16), (20, 0.19), (35, 0.21), (45, 0.23), (55, 0.26)],
                ]
            }
        }
    }
    
    database_ids = [f"sp|P{i:05d}|PROT_{i}" for i in range(100)]
    query_ids = [f"sp|Q{i:05d}|QUERY_{i}" for i in range(10)]
    
    # Test format
    format_output_txt(
        results,
        'test_output.txt',
        database_ids=database_ids,
        query_ids=query_ids,
        N=50,
        display_n=5
    )
    
    print("Test output generated: test_output.txt")