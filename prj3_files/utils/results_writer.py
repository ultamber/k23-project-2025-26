from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import json
import pickle

from utils.uniprot_client import UniProtClient
from utils.pfam_loader import _get_pfams_for_id

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
    per_query_times: Optional[List[float]] = None,
    blast_identity: Optional[Dict[int, Dict[int, float]]] = None,
    N: int = 50,
    display_n: int = 10,
    save_raw_data: bool = True,
    output_file: Optional[str] = None,
    uniprot_client: Optional[UniProtClient] = None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method_slug = method_name.lower().replace(' ', '_').replace('-', '_')
    
    # Create output filename
    if output_file is None:
        output_file = output_dir / f"{method_slug}_results.txt"
    
    # Extract metrics
    build_time = metrics.get('build_time', 0.0)
    search_time = metrics.get('search_time', 0.0)
    num_queries = metrics.get('num_queries', len(results))
    qps = metrics.get('qps', 0.0)
    avg_query_time = metrics.get('avg_query_time', 0.0)
    recall_at_n = metrics.get('recall_at_n', None)
    
    # Compute per-query metrics
    per_query_recalls = []
    all_distances_per_query = []
    
    if blast_results and 'blast_results_indices' in blast_results:
        blast_indices = blast_results['blast_results_indices']
    else:
        blast_indices = None
    
    # Collect all data
    for q_idx, query_results in enumerate(results):
        query_distances = [dist for idx, dist in query_results]
        all_distances_per_query.append(query_distances)
        
        # Per-query recall
        if blast_indices and q_idx in blast_indices:
            blast_top_n = set([hit[0] for hit in blast_indices[q_idx][:N]])
            ann_top_n = set([idx for idx, _ in query_results[:N]])
            if blast_top_n:
                query_recall = len(blast_top_n & ann_top_n) / len(blast_top_n)
                per_query_recalls.append(query_recall)
            else:
                per_query_recalls.append(None)
        else:
            per_query_recalls.append(None)
    
    # Calculate per-query QPS if times available
    per_query_qps = []
    if per_query_times:
        per_query_qps = [1.0 / t if t > 0 else 0.0 for t in per_query_times]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write(f"METHOD: {method_name.upper()}\n")
        f.write("=" * 90 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("Overall Performance Metrics\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"{'Metric':<35} {'Value':<20}\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Number of Queries':<35} {num_queries:<20d}\n")
        f.write(f"{'Top-N (for Recall@N)':<35} {N:<20d}\n")
        f.write(f"{'Build Time (s)':<35} {build_time:<20.4f}\n")
        f.write(f"{'Total Search Time (s)':<35} {search_time:<20.4f}\n")
        f.write(f"{'Average Query Time (s)':<35} {avg_query_time:<20.6f}\n")
        f.write(f"{'Average Query Time (ms)':<35} {avg_query_time * 1000:<20.4f}\n")
        f.write(f"{'Queries Per Second (QPS)':<35} {qps:<20.2f}\n")
        
        if recall_at_n is not None:
            f.write(f"{'Recall@{N} vs BLAST':<35} {recall_at_n:<20.4f}\n")
        else:
            f.write(f"{'Recall@{N} vs BLAST':<35} {'N/A':<20}\n")
        
        f.write("-" * 55 + "\n\n")
        
        if per_query_times:
            f.write("=" * 90 + "\n")
            f.write("Per-Query Timing\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"{'Query':<10} {'Query ID':<25} {'Time (s)':<15} {'Time (ms)':<15} {'QPS':<15}")
            if per_query_recalls:
                f.write(f"{'Recall@N':<12}")
            f.write("\n")
            f.write("-" * 90 + "\n")
            
            for q_idx in range(len(per_query_times)):
                q_id = query_ids[q_idx] if query_ids and q_idx < len(query_ids) else f"Query_{q_idx}"
                q_time = per_query_times[q_idx]
                q_time_ms = q_time * 1000
                q_qps = per_query_qps[q_idx] if per_query_qps else 0.0
                
                f.write(f"{q_idx + 1:<10} {q_id:<25} {q_time:<15.6f} {q_time_ms:<15.4f} {q_qps:<15.2f}")
                
                if per_query_recalls and q_idx < len(per_query_recalls):
                    recall_val = per_query_recalls[q_idx]
                    if recall_val is not None:
                        f.write(f"{recall_val:<12.4f}")
                    else:
                        f.write(f"{'N/A':<12}")
                f.write("\n")
            
            # Timing statistics
            f.write("-" * 90 + "\n")
            f.write(f"{'STATISTICS':<10} {'':<25} ")
            f.write(f"{np.mean(per_query_times):<15.6f} ")
            f.write(f"{np.mean(per_query_times) * 1000:<15.4f} ")
            f.write(f"{np.mean(per_query_qps):<15.2f}")
            if per_query_recalls:
                valid_recalls = [r for r in per_query_recalls if r is not None]
                if valid_recalls:
                    f.write(f"{np.mean(valid_recalls):<12.4f}")
            f.write(" (mean)\n")
            
            f.write(f"{'':<10} {'':<25} ")
            f.write(f"{np.std(per_query_times):<15.6f} ")
            f.write(f"{np.std(per_query_times) * 1000:<15.4f} ")
            f.write(f"{np.std(per_query_qps):<15.2f}")
            if per_query_recalls:
                valid_recalls = [r for r in per_query_recalls if r is not None]
                if valid_recalls:
                    f.write(f"{np.std(valid_recalls):<12.4f}")
            f.write(" (std)\n\n")
        
        f.write("=" * 90 + "\n")
        f.write("L2 Distances / All neighbours\n")
        f.write("=" * 90 + "\n\n")
        
        for q_idx, query_results in enumerate(results):
            q_id = query_ids[q_idx] if query_ids and q_idx < len(query_ids) else f"Query_{q_idx}"
            
            f.write(f"Query {q_idx + 1}: {q_id}\n")
            f.write(f"  Neighbors found: {len(query_results)}\n")
            
            if query_results:
                distances = [dist for idx, dist in query_results]
                f.write(f"  Distance range: [{min(distances):.6f}, {max(distances):.6f}]\n")
                f.write(f"  Mean distance: {np.mean(distances):.6f}\n")
                
                # Top-10 distances
                f.write(f"  Top-{min(10, len(query_results))} distances: ")
                top_dists = distances[:10]
                f.write(", ".join([f"{d:.4f}" for d in top_dists]))
                if len(distances) > 10:
                    f.write(", ...")
                f.write("\n")
            
            f.write("\n")
        
        f.write("=" * 90 + "\n")
        f.write("Distance Statistics\n")
        f.write("=" * 90 + "\n\n")
        
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
            f.write(f"{'Statistic':<25} {'Top-1':<18} {'Top-N':<18} {'All':<18}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Count':<25} {len(top1_distances):<18} {len(topN_distances):<18} {len(all_distances):<18}\n")
            f.write(f"{'Min':<25} {np.min(top1_distances):<18.6f} {np.min(topN_distances):<18.6f} {np.min(all_distances):<18.6f}\n")
            f.write(f"{'Max':<25} {np.max(top1_distances):<18.6f} {np.max(topN_distances):<18.6f} {np.max(all_distances):<18.6f}\n")
            f.write(f"{'Mean':<25} {np.mean(top1_distances):<18.6f} {np.mean(topN_distances):<18.6f} {np.mean(all_distances):<18.6f}\n")
            f.write(f"{'Std':<25} {np.std(top1_distances):<18.6f} {np.std(topN_distances):<18.6f} {np.std(all_distances):<18.6f}\n")
            f.write(f"{'Median':<25} {np.median(top1_distances):<18.6f} {np.median(topN_distances):<18.6f} {np.median(all_distances):<18.6f}\n")
            f.write(f"{'25th Percentile':<25} {np.percentile(top1_distances, 25):<18.6f} {np.percentile(topN_distances, 25):<18.6f} {np.percentile(all_distances, 25):<18.6f}\n")
            f.write(f"{'75th Percentile':<25} {np.percentile(top1_distances, 75):<18.6f} {np.percentile(topN_distances, 75):<18.6f} {np.percentile(all_distances, 75):<18.6f}\n")
        
        f.write("\n")
        
        valid_recalls = [r for r in per_query_recalls if r is not None]
        if valid_recalls:
            f.write("=" * 90 + "\n")
            f.write(f"[5] Recall@{N} Statistics\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"{'Statistic':<30} {'Value':<15}\n")
            f.write("-" * 45 + "\n")
            f.write(f"{'Queries with BLAST results':<30} {len(valid_recalls):<15}\n")
            f.write(f"{'Mean Recall@N':<30} {np.mean(valid_recalls):<15.4f}\n")
            f.write(f"{'Std Recall@N':<30} {np.std(valid_recalls):<15.4f}\n")
            f.write(f"{'Min Recall@N':<30} {np.min(valid_recalls):<15.4f}\n")
            f.write(f"{'Max Recall@N':<30} {np.max(valid_recalls):<15.4f}\n")
            f.write(f"{'Median Recall@N':<30} {np.median(valid_recalls):<15.4f}\n")
            
            # Recall distribution
            f.write("\nRecall Distribution:\n")
            bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            for i in range(len(bins) - 1):
                count = sum(1 for r in valid_recalls if bins[i] <= r < bins[i+1])
                if i == len(bins) - 2:
                    count = sum(1 for r in valid_recalls if bins[i] <= r <= bins[i+1])
                pct = 100 * count / len(valid_recalls) if valid_recalls else 0
                bar = "█" * int(pct / 5)
                f.write(f"  [{bins[i]:.1f} - {bins[i+1]:.1f}]: {count:4d} ({pct:5.1f}%) {bar}\n")
            
            f.write("\n")
        
        f.write("=" * 90 + "\n")
        f.write(f"Per-Query Neighbor Details (Top-{display_n})\n")
        f.write("=" * 90 + "\n\n")
        
        for q_idx, query_results in enumerate(results):
            q_id = query_ids[q_idx] if query_ids and q_idx < len(query_ids) else f"Query_{q_idx}"
            
            # Query Pfam
            query_pfam_display = None
            if pfam_mapping:
                query_pfams = _get_pfams_for_id(q_id, pfam_mapping) if pfam_mapping else []
                query_pfam_display = query_pfams[0] if query_pfams else None
            
            f.write("-" * 100 + "\n")
            f.write(f"Query {q_idx + 1}: {q_id}")
            if query_pfam_display:
                f.write(f" [Pfam: {query_pfam_display}]")
            f.write("\n")
            
            # Query metrics
            if per_query_times and q_idx < len(per_query_times):
                f.write(f"  Time: {per_query_times[q_idx]:.6f}s ({per_query_times[q_idx]*1000:.2f}ms)")
            if per_query_recalls and q_idx < len(per_query_recalls) and per_query_recalls[q_idx] is not None:
                f.write(f"  | Recall@{N}: {per_query_recalls[q_idx]:.4f}")
            f.write("\n")
            f.write("-" * 100 + "\n")
            
            f.write(f"{'Rank':<6} | {'Neighbor ID':<25} | {'L2 Dist':<10} | {'BLAST ID %':<12} | {'In BLAST?':<12} |")
            if pfam_mapping:
                f.write(f" {'Pfam':<10} |")
            f.write(" Bio comment\n")
            f.write("-" * 100 + "\n")
            
            # Get BLAST Top-N for this query
            blast_top_n_set = set()
            if blast_indices and q_idx in blast_indices:
                blast_top_n_set = set([hit[0] for hit in blast_indices[q_idx][:N]])

            for rank, (neighbor_idx, distance) in enumerate(query_results[:display_n], 1):
                neighbor_id = database_ids[neighbor_idx] if database_ids and neighbor_idx < len(database_ids) else f"Protein_{neighbor_idx}"
                
                # Truncate long IDs for display
                display_neighbor_id = neighbor_id[:22] + "..." if len(neighbor_id) > 25 else neighbor_id
                
                # In BLAST Top-N?
                in_blast = "?"
                if blast_indices and q_idx in blast_indices:
                    in_blast = "Yes" if neighbor_idx in blast_top_n_set else "No"
                
                # BLAST Identity % - NEW
                blast_id_str = "N/A"
                identity = _get_blast_identity(q_idx, neighbor_idx, blast_results, blast_identity)

                if identity is not None:
                    blast_id_str = f"{identity:.1f}%"
                
                # Neighbor Pfam
                neighbor_pfams = _get_pfams_for_id(neighbor_id, pfam_mapping) if pfam_mapping else []
                neighbor_pfam_display = neighbor_pfams[0] if neighbor_pfams else "N/A"
                
                comment = _generate_comment_with_identity(
                    query_pfams=query_pfams,
                    neighbor_pfams=neighbor_pfams,
                    distance=distance,
                    in_blast_top_n=(in_blast == "Yes") if in_blast != "?" else None,
                    blast_identity=identity
                )
                
                # Write row
                f.write(f"{rank:<6} | {display_neighbor_id:<25} | {distance:<10.4f} | {blast_id_str:<12} | {in_blast:<12} |")
                if pfam_mapping:
                    f.write(f" {neighbor_pfam_display:<10} |")
                f.write(f" {comment}\n")
            
            f.write("\n")
        
        f.write("=" * 90 + "\n")
        f.write("End of Results\n")
        f.write("=" * 90 + "\n")
    
    print(f"Saved {method_name} results to: {output_file}")

    if save_raw_data:
        raw_data_dir = output_dir / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        raw_data = {
            'method': method_name,
            'N': N,
            'num_queries': int(num_queries),
            'overall_metrics': {
                'build_time': float(build_time),
                'search_time': float(search_time),
                'qps': float(qps),
                'avg_query_time': float(avg_query_time),
                'recall_at_n': float(recall_at_n) if recall_at_n is not None else None
            },
            'per_query': []
        }
        
        for q_idx in range(len(results)):
            q_id = query_ids[q_idx] if query_ids and q_idx < len(query_ids) else f"Query_{q_idx}"
            
            query_data = {
                'query_idx': int(q_idx),
                'query_id': str(q_id),
                'time_seconds': float(per_query_times[q_idx]) if per_query_times and q_idx < len(per_query_times) else None,
                'qps': float(per_query_qps[q_idx]) if per_query_qps and q_idx < len(per_query_qps) else None,
                'recall_at_n': float(per_query_recalls[q_idx]) if per_query_recalls and q_idx < len(per_query_recalls) and per_query_recalls[q_idx] is not None else None,
                'num_neighbors': int(len(results[q_idx])),
                'neighbors': [
                    {
                        'rank': int(rank + 1),
                        'index': int(idx),
                        'id': str(database_ids[idx]) if database_ids and idx < len(database_ids) else f"Protein_{idx}",
                        'l2_distance': float(dist)
                    }
                    for rank, (idx, dist) in enumerate(results[q_idx])
                ]
            }
            raw_data['per_query'].append(query_data)
        
        # Save JSON
        json_file = raw_data_dir / f"{method_slug}_raw.json"
        with open(json_file, 'w') as f:
            json.dump(raw_data, f, indent=2)
        print(f"Saved raw data to: {json_file}")
        
        # Save distances as numpy array
        distances_file = raw_data_dir / f"{method_slug}_distances.npy"
        max_neighbors = max(len(r) for r in results) if results else 0
        distances_array = np.full((len(results), max_neighbors), np.nan)
        for q_idx, query_results in enumerate(results):
            for n_idx, (_, dist) in enumerate(query_results):
                distances_array[q_idx, n_idx] = dist
        np.save(distances_file, distances_array)
        print(f"Saved distances array to: {distances_file}")
        
        csv_file = raw_data_dir / f"{method_slug}_per_query.csv"
        with open(csv_file, 'w') as f:
            f.write("query_idx,query_id,time_s,time_ms,qps,recall_at_n,num_neighbors,min_dist,max_dist,mean_dist\n")
            for q_idx in range(len(results)):
                q_id = query_ids[q_idx] if query_ids and q_idx < len(query_ids) else f"Query_{q_idx}"
                q_time = float(per_query_times[q_idx]) if per_query_times and q_idx < len(per_query_times) else 0.0
                q_qps = float(per_query_qps[q_idx]) if per_query_qps and q_idx < len(per_query_qps) else 0.0
                q_recall = per_query_recalls[q_idx] if per_query_recalls and q_idx < len(per_query_recalls) else None
                
                dists = [float(d) for _, d in results[q_idx]] if results[q_idx] else [0.0]
                
                # Fix: Handle None recall properly
                recall_str = f"{q_recall:.4f}" if q_recall is not None else "N/A"
                
                f.write(f"{q_idx},{q_id},{q_time:.6f},{q_time*1000:.4f},{q_qps:.2f},")
                f.write(f"{recall_str},")
                f.write(f"{len(results[q_idx])},{min(dists):.6f},{max(dists):.6f},{np.mean(dists):.6f}\n")
        print(f"Saved per-query CSV to: {csv_file}")
    
    return output_file

def _get_blast_identity(
    query_idx: int,
    neighbor_idx: int,
    blast_results: Optional[Dict],
    blast_identity: Optional[Dict[int, Dict[int, float]]]
) -> Optional[float]:

    if blast_identity and query_idx in blast_identity:
        if neighbor_idx in blast_identity[query_idx]:
            return blast_identity[query_idx][neighbor_idx]
    
    if blast_results and 'blast_results_indices' in blast_results:
        blast_indices = blast_results['blast_results_indices']
        if query_idx in blast_indices:
            for hit in blast_indices[query_idx]:
                hit_idx = hit[0]
                if hit_idx == neighbor_idx:
                    # Check if pident is available (4th element)
                    if len(hit) >= 4:
                        return hit[3]  # pident
                    return None  # In BLAST but no pident
    
    return None

def _get_pfam_for_id(protein_id: str, pfam_mapping: Dict) -> Optional[str]:
    if protein_id in pfam_mapping:
        return pfam_mapping[protein_id].get('pfam')
    
    if '|' in protein_id:
        parts = protein_id.split('|')
        if len(parts) >= 2:
            accession = parts[1]
            if accession in pfam_mapping:
                return pfam_mapping[accession].get('pfam')
    
    return None


def _get_sequence(seqs, idx):
    if not seqs or idx >= len(seqs):
        return None
    
    seq = seqs[idx]
    if isinstance(seq, tuple):
        return seq[1] if len(seq) > 1 else seq[0]
    return seq


def _generate_comment(query_pfam, neighbor_pfam, distance, in_blast_top_n):
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


def write_comparison_summary(
    output_dir: str,
    all_metrics: Dict[str, Dict],
    N: int = 50
):
    output_dir = Path(output_dir)
    output_file = output_dir / "comparison_summary.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Method Comparison Summary\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Recall@N computed with N = {N}\n\n")
        
        f.write(f"{'Method':<20} {'Build (s)':<12} {'Search (s)':<12} {'QPS':<12} {'Avg Time (ms)':<15} {'Recall@N':<12}\n")
        f.write("-" * 100 + "\n")
        
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
        
        f.write("-" * 100 + "\n")
        f.write(f"{'BLAST (Reference)':<20} {'--':<12} {'--':<12} {'--':<12} {'--':<15} {'1.0000':<12}\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Analysis\n")
        f.write("-" * 100 + "\n")
        
        best_qps = max(sorted_methods, key=lambda x: x[1].get('qps', 0))
        best_recall = max(sorted_methods, key=lambda x: x[1].get('recall_at_n', 0) or 0)
        
        f.write(f"Fastest (highest QPS):    {best_qps[0]} ({best_qps[1].get('qps', 0):.2f} QPS)\n")
        f.write(f"Most Accurate (Recall@N): {best_recall[0]} ({best_recall[1].get('recall_at_n', 0):.4f})\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"Saved comparison summary to: {output_file}")
    return output_file

def _generate_comment_with_identity(
    query_pfams: List[str],
    neighbor_pfams: List[str],
    distance: float,
    in_blast_top_n: Optional[bool],
    blast_identity: Optional[float],
    fp_dist_thresh: float = 0.30,   # tune needed
) -> str:
    # No Pfam info
    if not query_pfams or not neighbor_pfams:
        if blast_identity is not None and blast_identity < 30:
            return "Low identity (<30%)"
        return "--"

    qset = set(query_pfams)
    nset = set(neighbor_pfams)
    shared = sorted(qset & nset)
    same_family = len(shared) > 0

    rep = shared[0] if shared else neighbor_pfams[0]

    low_identity = blast_identity is not None and blast_identity < 30
    very_low_identity = blast_identity is not None and blast_identity < 20

    if same_family and low_identity:
        return f"REMOTE HOMOLOG (shared {rep})" if very_low_identity else f"Remote homolog? (shared {rep})"

    if same_family and blast_identity is not None and blast_identity >= 30:
        return f"Close homolog (shared {rep})"

    if same_family:
        return f"Same family (shared {rep})"

    # different families
    if in_blast_top_n is False and distance <= fp_dist_thresh:
        return f"Possible FP? (closest domain {rep})"

    return f"Diff family (nearest {rep})"
