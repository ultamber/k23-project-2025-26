"""
Results writer for saving method-specific output files
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np


def write_method_results(
    method_name: str,
    output_dir: str,
    results: List[List[Tuple[int, float]]],
    metrics: Dict,
    query_ids: Optional[List[str]] = None,
    database_ids: Optional[List[str]] = None,
    blast_results: Optional[Dict] = None,
    pfam_mapping: Optional[Dict] = None,
    query_seqs: Optional[List] = None,
    database_seqs: Optional[List] = None,
    N: int = 50,
    display_n: int = 10
):
    """
    Write results for a single method to a dedicated text file.
    
    Creates: {output_dir}/{method_name}_results.txt
    
    Args:
        method_name: Name of the ANN method
        output_dir: Directory for output files
        results: List of (neighbor_idx, distance) tuples per query
        metrics: Dictionary with timing and performance metrics
        query_ids: List of query protein IDs
        database_ids: List of database protein IDs
        blast_results: BLAST ground truth results
        pfam_mapping: Pfam domain mapping
        query_seqs: Query sequences for identity calculation
        database_seqs: Database sequences for identity calculation
        N: Top-N for Recall@N calculation
        display_n: Number of neighbors to display in output
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    output_file = output_dir / f"{method_name.lower().replace(' ', '_')}_results.txt"
    
    # Extract metrics
    build_time = metrics.get('build_time', 0.0)
    search_time = metrics.get('search_time', 0.0)
    num_queries = metrics.get('num_queries', len(results))
    qps = metrics.get('qps', 0.0)
    avg_query_time = metrics.get('avg_query_time', 0.0)
    recall_at_n = metrics.get('recall_at_n', None)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # =====================================================================
        # HEADER
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write(f"METHOD: {method_name.upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # =====================================================================
        # [1] PERFORMANCE METRICS
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("[1] PERFORMANCE METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Metric':<30} {'Value':<20}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Number of Queries':<30} {num_queries:<20d}\n")
        f.write(f"{'Top-N (for Recall@N)':<30} {N:<20d}\n")
        f.write(f"{'Build Time (s)':<30} {build_time:<20.4f}\n")
        f.write(f"{'Total Search Time (s)':<30} {search_time:<20.4f}\n")
        f.write(f"{'Average Query Time (s)':<30} {avg_query_time:<20.6f}\n")
        f.write(f"{'Queries Per Second (QPS)':<30} {qps:<20.2f}\n")
        
        if recall_at_n is not None:
            f.write(f"{'Recall@N vs BLAST':<30} {recall_at_n:<20.4f}\n")
        else:
            f.write(f"{'Recall@N vs BLAST':<30} {'N/A (no ground truth)':<20}\n")
        
        f.write("-" * 50 + "\n\n")
        
        # =====================================================================
        # [2] PER-QUERY RESULTS
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write(f"[2] PER-QUERY RESULTS (Top-{display_n} Neighbors)\n")
        f.write("=" * 80 + "\n\n")
        
        # Compute per-query recall if BLAST available
        per_query_recalls = []
        if blast_results and 'blast_results_indices' in blast_results:
            blast_indices = blast_results['blast_results_indices']
        else:
            blast_indices = None
        
        for q_idx, query_results in enumerate(results):
            # Get query ID
            if query_ids and q_idx < len(query_ids):
                query_id = query_ids[q_idx]
            else:
                query_id = f"Query_{q_idx}"
            
            # Get query Pfam if available
            query_pfam = None
            query_pfam_desc = ""
            if pfam_mapping:
                query_pfam = _get_pfam_for_id(query_id, pfam_mapping)
                if query_pfam:
                    query_pfam_desc = f" ({query_pfam})"
            
            # Compute per-query recall
            query_recall = None
            if blast_indices and q_idx in blast_indices:
                blast_top_n = set([hit_idx for hit_idx, _, _ in blast_indices[q_idx][:N]])
                ann_top_n = set([idx for idx, _ in query_results[:N]])
                if blast_top_n:
                    query_recall = len(blast_top_n & ann_top_n) / len(blast_top_n)
                    per_query_recalls.append(query_recall)
            
            # Query header
            f.write("-" * 80 + "\n")
            f.write(f"Query {q_idx + 1}: {query_id}{query_pfam_desc}\n")
            if query_recall is not None:
                f.write(f"Recall@{N}: {query_recall:.4f}\n")
            f.write("-" * 80 + "\n")
            
            # Column headers
            f.write(f"{'Rank':<6} {'Neighbor ID':<25} {'L2 Distance':<15} ")
            
            if blast_indices:
                f.write(f"{'In BLAST Top-N?':<18} ")
            
            if pfam_mapping:
                f.write(f"{'Pfam':<12} ")
            
            f.write("{'Comment'}\n")
            f.write("-" * 80 + "\n")
            
            # Get query sequence for identity calculation
            query_seq = _get_sequence(query_seqs, q_idx) if query_seqs else None
            
            # Neighbor rows
            for rank, (neighbor_idx, distance) in enumerate(query_results[:display_n], 1):
                # Get neighbor ID
                if database_ids and neighbor_idx < len(database_ids):
                    neighbor_id = database_ids[neighbor_idx]
                else:
                    neighbor_id = f"Protein_{neighbor_idx}"
                
                # Check if in BLAST Top-N
                in_blast = "?"
                if blast_indices and q_idx in blast_indices:
                    blast_top_n_ids = [hit_idx for hit_idx, _, _ in blast_indices[q_idx][:N]]
                    in_blast = "Yes" if neighbor_idx in blast_top_n_ids else "No"
                
                # Get neighbor Pfam
                neighbor_pfam = ""
                if pfam_mapping:
                    neighbor_pfam = _get_pfam_for_id(neighbor_id, pfam_mapping) or "N/A"
                
                # Generate bio comment
                comment = _generate_comment(
                    query_pfam, neighbor_pfam if pfam_mapping else None,
                    distance, in_blast == "Yes" if in_blast != "?" else None
                )
                
                # Write row
                f.write(f"{rank:<6} {neighbor_id:<25} {distance:<15.6f} ")
                
                if blast_indices:
                    f.write(f"{in_blast:<18} ")
                
                if pfam_mapping:
                    f.write(f"{neighbor_pfam:<12} ")
                
                f.write(f"{comment}\n")
            
            f.write("\n")
        
        # =====================================================================
        # [3] DISTANCE STATISTICS
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("[3] DISTANCE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        # Collect all distances
        all_distances = []
        top1_distances = []
        topN_distances = []
        
        for query_results in results:
            if query_results:
                top1_distances.append(query_results[0][1])
                for idx, dist in query_results[:N]:
                    topN_distances.append(dist)
                for idx, dist in query_results:
                    all_distances.append(dist)
        
        if all_distances:
            f.write(f"{'Statistic':<30} {'Top-1':<15} {'Top-N':<15} {'All':<15}\n")
            f.write("-" * 75 + "\n")
            f.write(f"{'Count':<30} {len(top1_distances):<15} {len(topN_distances):<15} {len(all_distances):<15}\n")
            f.write(f"{'Min Distance':<30} {np.min(top1_distances):<15.6f} {np.min(topN_distances):<15.6f} {np.min(all_distances):<15.6f}\n")
            f.write(f"{'Max Distance':<30} {np.max(top1_distances):<15.6f} {np.max(topN_distances):<15.6f} {np.max(all_distances):<15.6f}\n")
            f.write(f"{'Mean Distance':<30} {np.mean(top1_distances):<15.6f} {np.mean(topN_distances):<15.6f} {np.mean(all_distances):<15.6f}\n")
            f.write(f"{'Std Distance':<30} {np.std(top1_distances):<15.6f} {np.std(topN_distances):<15.6f} {np.std(all_distances):<15.6f}\n")
            f.write(f"{'Median Distance':<30} {np.median(top1_distances):<15.6f} {np.median(topN_distances):<15.6f} {np.median(all_distances):<15.6f}\n")
        
        f.write("\n")
        
        # =====================================================================
        # [4] RECALL STATISTICS (if BLAST available)
        # =====================================================================
        if per_query_recalls:
            f.write("=" * 80 + "\n")
            f.write("[4] RECALL STATISTICS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"{'Statistic':<30} {'Value':<15}\n")
            f.write("-" * 45 + "\n")
            f.write(f"{'Queries with BLAST results':<30} {len(per_query_recalls):<15}\n")
            f.write(f"{'Mean Recall@N':<30} {np.mean(per_query_recalls):<15.4f}\n")
            f.write(f"{'Std Recall@N':<30} {np.std(per_query_recalls):<15.4f}\n")
            f.write(f"{'Min Recall@N':<30} {np.min(per_query_recalls):<15.4f}\n")
            f.write(f"{'Max Recall@N':<30} {np.max(per_query_recalls):<15.4f}\n")
            f.write(f"{'Median Recall@N':<30} {np.median(per_query_recalls):<15.4f}\n")
            
            # Recall distribution
            f.write("\nRecall Distribution:\n")
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            for i in range(len(bins) - 1):
                count = sum(1 for r in per_query_recalls if bins[i] <= r < bins[i+1])
                if i == len(bins) - 2:  # Last bin includes 1.0
                    count = sum(1 for r in per_query_recalls if bins[i] <= r <= bins[i+1])
                pct = 100 * count / len(per_query_recalls)
                f.write(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}]: {count:4d} ({pct:5.1f}%)\n")
            
            f.write("\n")
        
        # =====================================================================
        # FOOTER
        # =====================================================================
        f.write("=" * 80 + "\n")
        f.write("END OF RESULTS\n")
        f.write("=" * 80 + "\n")
    
    print(f"  ✓ Saved {method_name} results to: {output_file}")
    return output_file


def write_comparison_summary(
    output_dir: str,
    all_metrics: Dict[str, Dict],
    N: int = 50
):
    """
    Write a comparison summary of all methods to a single file.
    
    Creates: {output_dir}/comparison_summary.txt
    """
    output_dir = Path(output_dir)
    output_file = output_dir / "comparison_summary.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("METHOD COMPARISON SUMMARY\n")
        f.write("=" * 90 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Recall@N computed with N = {N}\n\n")
        
        # Header
        f.write(f"{'Method':<20} {'Build (s)':<12} {'Search (s)':<12} {'QPS':<12} {'Avg Time (ms)':<15} {'Recall@N':<12}\n")
        f.write("-" * 90 + "\n")
        
        # Sort by Recall@N (descending)
        sorted_methods = sorted(
            all_metrics.items(),
            key=lambda x: x[1].get('recall_at_n', 0) or 0,
            reverse=True
        )
        
        for method_name, metrics in sorted_methods:
            build_time = metrics.get('build_time', 0.0)
            search_time = metrics.get('search_time', 0.0)
            qps = metrics.get('qps', 0.0)
            avg_time_ms = metrics.get('avg_query_time', 0.0) * 1000
            recall = metrics.get('recall_at_n', None)
            
            recall_str = f"{recall:.4f}" if recall is not None else "N/A"
            
            f.write(f"{method_name:<20} {build_time:<12.4f} {search_time:<12.4f} {qps:<12.2f} {avg_time_ms:<15.4f} {recall_str:<12}\n")
        
        # Add BLAST reference
        f.write("-" * 90 + "\n")
        f.write(f"{'BLAST (Reference)':<20} {'--':<12} {'--':<12} {'--':<12} {'--':<15} {'1.0000':<12}\n")
        
        f.write("=" * 90 + "\n\n")
        
        # Speed vs Accuracy analysis
        f.write("SPEED VS ACCURACY ANALYSIS\n")
        f.write("-" * 90 + "\n")
        
        # Find best method for each criterion
        best_qps = max(sorted_methods, key=lambda x: x[1].get('qps', 0))
        best_recall = max(sorted_methods, key=lambda x: x[1].get('recall_at_n', 0) or 0)
        
        f.write(f"Fastest Method (highest QPS): {best_qps[0]} ({best_qps[1].get('qps', 0):.2f} QPS)\n")
        f.write(f"Most Accurate (highest Recall@N): {best_recall[0]} ({best_recall[1].get('recall_at_n', 0):.4f})\n")
        
        f.write("\n")
        f.write("=" * 90 + "\n")
    
    print(f"  ✓ Saved comparison summary to: {output_file}")
    return output_file


def _get_pfam_for_id(protein_id: str, pfam_mapping: Dict) -> Optional[str]:
    """Get Pfam domain for a protein ID."""
    if protein_id in pfam_mapping:
        return pfam_mapping[protein_id].get('pfam')
    
    # Try extracting accession
    if '|' in protein_id:
        parts = protein_id.split('|')
        if len(parts) >= 2:
            accession = parts[1]
            if accession in pfam_mapping:
                return pfam_mapping[accession].get('pfam')
    
    return None


def _get_sequence(seqs, idx):
    """Extract sequence string from various formats."""
    if not seqs or idx >= len(seqs):
        return None
    
    seq = seqs[idx]
    if isinstance(seq, tuple):
        return seq[1] if len(seq) > 1 else seq[0]
    return seq


def _generate_comment(query_pfam, neighbor_pfam, distance, in_blast_top_n):
    """Generate a biological comment based on Pfam and distance."""
    if query_pfam is None or neighbor_pfam is None:
        return "--"
    
    same_family = query_pfam == neighbor_pfam
    
    if same_family:
        if in_blast_top_n is False:
            return f"Remote homolog? ({query_pfam})"
        elif in_blast_top_n is True:
            return f"Same family ({query_pfam})"
        else:
            return f"Same family ({query_pfam})"
    else:
        if distance < 5.0:
            return f"Different family ({neighbor_pfam})"
        return "--"