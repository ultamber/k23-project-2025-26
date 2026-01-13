from pathlib import Path
from typing import Dict, List, Tuple, Optional
from Bio import Align
from utils.pfam_loader import load_pfam_mapping, get_pfam_for_id, check_same_family, PFAM_DESCRIPTIONS

def compute_sequence_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignments = aligner.align(seq1, seq2)
    
    if not alignments:
        return 0.0
    
    alignment = alignments[0]
    aligned_seq1 = alignment.seqA
    aligned_seq2 = alignment.seqB
    
    # Count matches
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    
    # Identity = matches / min(len(seq1), len(seq2))
    identity = 100.0 * matches / min(len(seq1), len(seq2))
    
    return identity


def get_sequence_from_list(seq_list, idx):
    if not seq_list or idx >= len(seq_list):
        return None
    
    seq = seq_list[idx]
    
    # Handle different formats
    if isinstance(seq, tuple):
        # Format: (id, sequence)
        return seq[1] if len(seq) > 1 else seq[0]
    elif isinstance(seq, str):
        return seq
    else:
        return None


def is_in_blast_top_n(query_idx: int, neighbor_idx: int, 
    blast_results: Optional[Dict], N: int) -> Optional[bool]:
    if not blast_results or 'blast_results_indices' not in blast_results:
        return None
    
    blast_indices = blast_results['blast_results_indices']
    if query_idx not in blast_indices:
        return None
    
    blast_top_n = [hit_idx for hit_idx, _, _ in blast_indices[query_idx][:N]]
    return neighbor_idx in blast_top_n


def format_output_txt(
    results: Dict,
    output_file: str,
    database_ids: Optional[List[str]] = None,
    query_ids: Optional[List[str]] = None,
    database_seqs: Optional[List] = None,
    query_seqs: Optional[List] = None,
    blast_results: Optional[Dict] = None,
    pfam_mapping: Optional[Dict] = None,
    N: int = 50,
    display_n: int = 10
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Print availability info
    if pfam_mapping:
        print(f"  Pfam mapping: {len(pfam_mapping)} proteins loaded")
    else:
        print(f"  Pfam mapping: Not available")

    # Print sequence availability
    if database_seqs:
        print(f"  Database sequences: {len(database_seqs)} loaded")
    else:
        print(f"  Database sequences: Not available (identity % will show N/A)")
    
    if query_seqs:
        print(f"  Query sequences: {len(query_seqs)} loaded")
    else:
        print(f"  Query sequences: Not available (identity % will show N/A)")
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Protein Remote Homolog Detection - Results\n")
        f.write("="*70 + "\n\n")
        
        num_queries = results.get('num_queries', 0)
        
        f.write(f"Total Queries: {num_queries}\n")
        f.write(f"N = {N}\n")
        f.write(f"Display N = {display_n}\n\n")
        f.write("="*70 + "\n")
        f.write("[1] Συνοπτική σύγκριση μεθόδων\n")
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
            
            display_name = method_name.upper().replace('-', ' ')
            f.write(f"{display_name:<18s} {time_per_query:<18.4f} {qps:<12.2f} {recall:<20.4f}\n")
        
        # Add BLAST reference
        if blast_results:
            f.write(f"{'BLAST (Reference)':<18s} {'0.0000':<18s} {'0.00':<12s} {'1.0000':<20s}\n")
        
        f.write("-"*70 + "\n\n")
        f.write("="*70 + "\n")
        f.write(f"[2] TOP-{display_n} γείτονες ανά μέθοδο\n")
        f.write("="*70 + "\n\n")
        
        # Get first query info
        first_query_id = "Unknown"
        if query_ids and len(query_ids) > 0:
            first_query_id = query_ids[0]
        
        f.write(f"Query Protein: {first_query_id}\n")
        f.write(f"(Showing first query as example. All queries evaluated in Recall@N.)\n\n")
        
        # Get first query sequence
        first_query_seq = get_sequence_from_list(query_seqs, 0) if query_seqs else None
        
        # For each method's results
        for method_name in method_order:
            if method_name not in methods:
                continue
            
            method_data = methods[method_name]
            method_results = method_data.get('results', [])
            
            if not method_results:
                continue
            
            first_query_results = method_results[0] if method_results else []
            
            # Get query Pfam
            query_pfam = None
            query_pfam_desc = ""
            if pfam_mapping and query_ids:
                query_pfam = get_pfam_for_id(query_ids[0], pfam_mapping)
                if query_pfam:
                    query_pfam_desc = PFAM_DESCRIPTIONS.get(query_pfam, "")
            
            # Method header with Pfam info
            display_name = method_name.upper().replace('-', ' ')
            f.write(f"Method: {display_name}\n")
            if query_pfam:
                f.write(f"Query Pfam: {query_pfam} ({query_pfam_desc})\n")
            f.write("-"*110 + "\n")
            
            # Column headers - add Pfam column
            f.write(f"{'Rank':<6s} {'Neighbor ID':<20s} {'L2 Dist':<12s} {'Seq ID %':<12s} {'Neighbor Pfam':<15s} {'In BLAST?':<12s} {'Bio Comment'}\n")
            f.write("-"*110 + "\n")
            
            # Process neighbors
            for rank, (neighbor_idx, distance) in enumerate(first_query_results[:display_n], 1):
                # Get neighbor ID
                if database_ids and neighbor_idx < len(database_ids):
                    neighbor_id = database_ids[neighbor_idx]
                else:
                    neighbor_id = f"Prot_{neighbor_idx}"
                
                # Get sequence identity
                blast_identity_str = "N/A"
                blast_identity_float = None
                if first_query_seq and database_seqs and neighbor_idx < len(database_seqs):
                    neighbor_seq = get_sequence_from_list(database_seqs, neighbor_idx)
                    if neighbor_seq:
                        try:
                            identity = compute_sequence_identity(first_query_seq, neighbor_seq)
                            blast_identity_str = f"{identity:.1f}"
                            blast_identity_float = identity
                        except:
                            blast_identity_str = "Error"
                
                # Get neighbor Pfam
                neighbor_pfam = None
                neighbor_pfam_str = "N/A"
                if pfam_mapping:
                    neighbor_pfam = get_pfam_for_id(neighbor_id, pfam_mapping)
                    if neighbor_pfam:
                        neighbor_pfam_str = neighbor_pfam
                
                # Check BLAST Top-N
                in_blast_top_n_str = "?"
                in_blast_top_n_bool = None
                if blast_results:
                    in_top_n = is_in_blast_top_n(0, neighbor_idx, blast_results, N)
                    if in_top_n is not None:
                        in_blast_top_n_str = "Yes" if in_top_n else "No"
                        in_blast_top_n_bool = in_top_n
                
                # Generate bio comment based on Pfam
                bio_comment = generate_pfam_bio_comment(
                    query_pfam=query_pfam,
                    neighbor_pfam=neighbor_pfam,
                    sequence_identity=blast_identity_float,
                    distance=distance,
                    in_blast_top_n=in_blast_top_n_bool
                )
                
                f.write(f"{rank:<6d} {neighbor_id:<20s} {distance:<12.4f} {blast_identity_str:<12s} {neighbor_pfam_str:<15s} {in_blast_top_n_str:<12s} {bio_comment}\n")
            
            f.write("\n")

        f.write("="*70 + "\n")
        f.write("NOTES:\n")
        f.write("  - BLAST ID (%): Sequence identity percentage\n")
        f.write("  - Remote homologs typically have <30% identity but similar function\n")
        if not database_seqs or not query_seqs:
            f.write("  - Sequence files not loaded: Identity % shows N/A\n")
            f.write("  - To enable: provide .fasta files alongside embeddings\n")
        f.write("="*70 + "\n\n")

def generate_pfam_bio_comment(
    query_pfam: Optional[str],
    neighbor_pfam: Optional[str],
    sequence_identity: Optional[float],
    distance: float,
    in_blast_top_n: Optional[bool]
) -> str:

    if query_pfam is None or neighbor_pfam is None:
        return "--"
    
    same_family = (query_pfam == neighbor_pfam)
    low_identity = sequence_identity is not None and sequence_identity < 30
    very_low_identity = sequence_identity is not None and sequence_identity < 20
    
    if same_family and low_identity:
        if very_low_identity:
            return f"REMOTE HOMOLOG ({query_pfam})"
        else:
            return f"Remote homolog? ({query_pfam})"
    
    # Same family, higher identity - close homolog
    if same_family and sequence_identity is not None and sequence_identity >= 30:
        return f"Close homolog ({query_pfam})"
    
    # Same family, no identity info
    if same_family:
        return f"Same family ({query_pfam})"
    
    # Different family but close in embedding space - potential false positive
    if not same_family and distance < 3.0:
        return f"Diff family ({neighbor_pfam}) - FP?"
    
    # Different family
    if not same_family:
        return f"Diff: {neighbor_pfam}"
    
    return "--"