import argparse
import numpy as np
from pathlib import Path
import json
import time
import pickle
import tempfile
from typing import Dict, List, Tuple, Optional

from methods.lsh import EuclideanLSH
from methods.ivfflat import IVFFlat
from methods.ivfpq import IVFPQ
from methods.neural_lsh import NeuralLSH
from methods.hypercube import Hypercube
from utils.evaluation import PerformanceTracker
from utils.output_formatter import format_output_txt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protein Search with ANN Methods",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-d', '--database',
        help='Database vectors (.dat, .npy)'
    )
    parser.add_argument(
        '-q', '--queries',
        help='Query file (.fasta)'
    )
    parser.add_argument(
        '-o', '--output',
        default='results.txt',
        help='Output file'
    )
    parser.add_argument(
        '-method', '--method',
        choices=['lsh', 'hypercube', 'ivfflat', 'ivfpq', 'neural-lsh', 'neural', 'all'],
        default='all',
        help='ANN method to use (default: all)'
    )

    parser.add_argument(
        '--embeddings', '-e',
        help='Database embeddings file (.npy) - alternative to -d'
    )
    parser.add_argument(
        '--query-embeddings',
        help='Query embeddings file (.npy) - alternative to -q'
    )
    parser.add_argument(
        '--ground-truth', '-g',
        help='BLAST ground truth file (.pkl)'
    )

    parser.add_argument(
        '--N',
        type=int,
        default=50,
        help='Number of neighbors to retrieve (default: 50)'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        help='Maximum number of queries to process (for testing)'
    )

    parser.add_argument(
        '--run-blast',
        action='store_true',
        help='Automatically run BLAST if ground truth not provided'
    )
    parser.add_argument(
        '--blast-evalue',
        type=float,
        default=0.01,
        help='BLAST E-value threshold (default: 0.01)'
    )
    parser.add_argument(
        '--blast-threads',
        type=int,
        default=8,
        help='BLAST threads (default: 8)'
    )

    parser.add_argument(
        '--embed-batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    parser.add_argument(
        '--embed-device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for embedding generation (default: cuda)'
    )

    # LSH
    parser.add_argument('--lsh-L', type=int, default=10, help='LSH: number of hash tables')
    parser.add_argument('--lsh-k', type=int, default=4, help='LSH: hash functions per table')
    parser.add_argument('--lsh-w', type=float, default=4.0, help='LSH: bucket width')
    
    # Hypercube
    parser.add_argument('--hc-kproj', type=int, help='Hypercube: projection dimension')
    parser.add_argument('--hc-w', type=float, default=4.0, help='Hypercube: bucket width')
    parser.add_argument('--hc-max-probes', type=int, default=100, help='Hypercube: max probes')
    
    # IVF
    parser.add_argument('--ivf-n-clusters', type=int, help='IVF: number of clusters')
    parser.add_argument('--ivf-n-probe', type=int, default=10, help='IVF: number of probes')
    parser.add_argument('--ivf-M', type=int, default=8, help='IVF-PQ: number of subvectors')
    
    # Neural LSH
    parser.add_argument('--nlsh-m', type=int, default=100, help='Neural LSH: partitions')
    parser.add_argument('--nlsh-k', type=int, default=25, help='Neural LSH: k-NN')
    parser.add_argument('--nlsh-T', type=int, default=10, help='Neural LSH: probes')
    parser.add_argument('--nlsh-epochs', type=int, default=50, help='Neural LSH: training epochs')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def detect_file_type(filepath: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()
    
    if ext in ['.fasta', '.fa', '.faa']:
        return 'fasta'
    elif ext in ['.npy']:
        return 'npy'
    elif ext in ['.dat']:
        return 'dat'
    elif ext in ['.pkl', '.pickle']:
        return 'pkl'
    else:
        return 'unknown'


def load_embeddings(filepath: str) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    path = Path(filepath)

    if path.suffix == '.npy':
        embeddings = np.load(filepath)
    elif path.suffix == '.dat':
        try:
            embeddings = np.load(filepath)
        except:
            embeddings = np.loadtxt(filepath)
    else:
        raise ValueError(f"Unknown embedding format: {path.suffix}")
    
    ids = None
    ids_file = path.with_suffix('.ids')
    if ids_file.exists():
        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f]
    
    sequences = None
    fasta_file = path.with_suffix('.fasta')
    if fasta_file.exists():
        from utils.fasta_loader import load_fasta
        seqs, seq_ids = load_fasta(str(fasta_file))
        sequences = seqs
    
    return embeddings, ids, sequences


def embed_fasta(
    fasta_file: str,
    output_prefix: str = None,
    batch_size: int = 32,
    device: str = 'cuda'
    ) -> Tuple[np.ndarray, List[str]]:
    
    # Use protein_embed.py module
    from protein_embed import ESM2Embedder
    
    embedder = ESM2Embedder(
        model_name='facebook/esm2_t6_8M_UR50D',
        device=device
    )
    
    # Generate embeddings
    if output_prefix:
        # Save to file
        embedder.embed_fasta(
            fasta_file=fasta_file,
            output_prefix=output_prefix,
            batch_size=batch_size
        )
        
        # Load back
        embeddings = np.load(f"{output_prefix}.npy")
        with open(f"{output_prefix}.ids", 'r') as f:
            ids = [line.strip() for line in f]
    else:
        # Generate in memory (no save)
        embeddings, ids = embedder.embed_fasta_memory(
            fasta_file=fasta_file,
            batch_size=batch_size
        )
    
    return embeddings, ids


def run_blast_search(
    database_fasta: str,
    query_fasta: str,
    database_ids: Optional[List[str]] = None,
    query_ids: Optional[List[str]] = None,
    N: int = 50,
    evalue: float = 0.01,
    threads: int = 8
) -> Dict:
    print(f"\n{'='*70}")
    print(f"RUNNING BLAST SEARCH")
    print(f"{'='*70}")
    
    from utils.blast_runner import BLASTRunner
    
    blast = BLASTRunner(
        db_fasta=database_fasta,
        evalue_threshold=evalue
    )
    
    # Build database
    blast.build_database(database_fasta)
    
    # Run search
    results = blast.search_fasta(
        query_fasta=query_fasta,
        N=N,
        num_threads=threads
    )
    
    # Convert to indices if IDs provided
    blast_results_indices = None
    if database_ids and query_ids:
        
        db_id_to_index = {pid: i for i, pid in enumerate(database_ids)}
        query_id_to_index = {pid: i for i, pid in enumerate(query_ids)}
        
        from utils.fasta_loader import get_accession
        
        blast_results_indices = {}
        for query_id, hits in results.items():
            query_acc = get_accession(query_id)
            if query_acc in query_id_to_index:
                query_idx = query_id_to_index[query_acc]
                hit_indices = []
                for hit_id, score, evalue in hits:
                    hit_acc = get_accession(hit_id)
                    if hit_acc in db_id_to_index:
                        hit_idx = db_id_to_index[hit_acc]
                        hit_indices.append((hit_idx, score, evalue))
                if hit_indices:
                    blast_results_indices[query_idx] = hit_indices
    
    # Cleanup
    blast.cleanup()
    
    print(f"BLAST search completed")
    print(f"{'='*70}\n")
    
    return {
        'blast_results_ids': results,
        'blast_results_indices': blast_results_indices,
        'params': {'N': N, 'evalue': evalue}
    }

def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    
    print("="*70)
    print("Protein Search with ANN Methods")
    print("="*70)

    print(f"\nResolving database...")
    
    database_embeddings = None
    database_ids = None
    database_seqs = None
    database_fasta = None
    
    if args.database:
        db_type = detect_file_type(args.database)
        
        if db_type == 'fasta':
            # Need to embed
            database_fasta = args.database
            print(f"Database: FASTA file ({args.database})")
            print(f"Will generate embeddings...")
            
            # Generate embeddings
            temp_prefix = tempfile.mkdtemp(prefix='db_vectors_')
            database_embeddings, database_ids = embed_fasta(
                args.database,
                output_prefix=temp_prefix,
                batch_size=args.embed_batch_size,
                device=args.embed_device
            )
            
            # Load sequences
            from utils.fasta_loader import load_fasta
            database_seqs, _ = load_fasta(args.database)
        
        elif db_type in ['npy', 'dat']:
            # Load embeddings
            database_embeddings, database_ids, database_seqs = load_embeddings(
                args.database
            )
            print(f"Database: {database_embeddings.shape}")
        
        else:
            raise ValueError(f"Unknown database format: {args.database}")
    
    # Or use --embeddings
    elif args.embeddings:
        database_embeddings, database_ids, database_seqs = load_embeddings(
            args.embeddings
        )
        print(f"Database: {database_embeddings.shape}")
    
    else:
        raise ValueError("Must provide either -d/--database or --embeddings")

    print(f"\nResolving queries...")
    
    query_embeddings = None
    query_ids = None
    query_seqs = None
    query_fasta = None
    
    # Check -q/--queries argument
    if args.queries:
        q_type = detect_file_type(args.queries)
        
        if q_type == 'fasta':
            # Need to embed
            query_fasta = args.queries
            print(f"Queries: FASTA file ({args.queries})")
            print(f"Will generate embeddings...")
            
            # Generate embeddings
            temp_prefix = tempfile.mkdtemp(prefix='query_vectors_')
            query_embeddings, query_ids = embed_fasta(
                args.queries,
                output_prefix=temp_prefix,
                batch_size=args.embed_batch_size,
                device=args.embed_device
            )
            
            # Load sequences
            from utils.fasta_loader import load_fasta
            query_seqs, _ = load_fasta(args.queries)
        
        elif q_type in ['npy', 'dat']:
            # Load embeddings
            query_embeddings, query_ids, query_seqs = load_embeddings(
                args.queries
            )
            print(f"Queries: {query_embeddings.shape}")
        
        else:
            raise ValueError(f"Unknown query format: {args.queries}")
    
    # Or use --query-embeddings
    elif args.query_embeddings:
        query_embeddings, query_ids, query_seqs = load_embeddings(
            args.query_embeddings
        )
        print(f"Queries: {query_embeddings.shape}")
    
    else:
        raise ValueError("Must provide either -q/--queries or --query-embeddings")
    
    # Limit queries if requested
    if args.max_queries:
        query_embeddings = query_embeddings[:args.max_queries]
        if query_ids:
            query_ids = query_ids[:args.max_queries]
        if query_seqs:
            query_seqs = query_seqs[:args.max_queries]
        print(f"Limited to {len(query_embeddings)} queries")
    
    print(f"\nLoading BLAST ground truth...")
    
    blast_results = None
    
    if args.ground_truth:
        # Load existing ground truth
        with open(args.ground_truth, 'rb') as f:
            blast_results = pickle.load(f)
        print(f"Loaded from {args.ground_truth}")
    
    elif args.run_blast and database_fasta and query_fasta:
        # Generate BLAST ground truth on-the-fly
        print(f"No ground truth provided, running BLAST...")
        blast_results = run_blast_search(
            database_fasta=database_fasta,
            query_fasta=query_fasta,
            database_ids=database_ids,
            query_ids=query_ids,
            N=args.N,
            evalue=args.blast_evalue,
            threads=args.blast_threads
        )
    
    else:
        print(f"No ground truth available (use --run-blast or --ground-truth)")

    print(f"\nRunning ANN methods...")
    print(f"Method: {args.method}")
    print(f"N: {args.N}")
    
    tracker = PerformanceTracker()
    all_results = {}
    
    methods_to_run = []
    if args.method == 'all':
        methods_to_run = ['lsh', 'hypercube', 'ivf-flat', 'ivf-pq', 'neural-lsh']
    else:
        # Handle aliases
        method_map = {
            'neural': 'neural-lsh',
        }
        methods_to_run = [method_map.get(args.method, args.method)]
    
    for method in methods_to_run:
        print(f"\n  Running {method.upper()}...")
        
        # Build index
        if method == 'lsh':
            index = EuclideanLSH(
                L=args.lsh_L,
                k=args.lsh_k,
                w=args.lsh_w,
                seed=args.seed
            )
        
        elif method == 'hypercube':
            index = Hypercube(
                kproj=args.hc_kproj,
                w=args.hc_w,
                max_probes=args.hc_max_probes,
                seed=args.seed
            )
        
        elif method == 'ivfflat':
            index = IVFFlat(
                n_clusters=args.ivf_n_clusters,
                n_probe=args.ivf_n_probe,
                seed=args.seed
            )
        
        elif method == 'ivfpq':
            index = IVFPQ(
                n_clusters=args.ivf_n_clusters,
                n_probe=args.ivf_n_probe,
                M=args.ivf_M,
                seed=args.seed
            )
        
        elif method == 'neural-lsh':
            index = NeuralLSH(
                m=args.nlsh_m,
                k_neighbors=args.nlsh_k,
                seed=args.seed
            )
        
        # Build or load index 
        tracker.start_build(method)
        if method == 'neural-lsh':
            index.build_index(database_embeddings, epochs=args.nlsh_epochs)
        else:
            index.build_index(database_embeddings)
        tracker.end_build(method)
        
        # Search
        tracker.start_search(method)
        if method == 'neural-lsh':
            results = index.batch_search(query_embeddings, N=args.N, T=args.nlsh_T)
        else:
            results = index.batch_search(query_embeddings, N=args.N)
        tracker.end_search(method, len(query_embeddings))
        
        all_results[method] = results
        
        print(f"  Completed in {tracker.metrics[method]['search_time']:.2f}s")
        print(f"  QPS: {tracker.metrics[method]['qps']:.2f}")
    
    if blast_results and 'blast_results_indices' in blast_results:
        print(f"\nEvaluating against BLAST...")
        
        blast_indices = blast_results['blast_results_indices']
        
        for method, results in all_results.items():
            # Compute recall@N
            recall = compute_recall_at_n(results, blast_indices, args.N)
            tracker.metrics[method]['recall_at_n'] = recall
            
            print(f"{method}: Recall@{args.N} = {recall:.4f}")
    
    print(f"\nSaving results...")
    
    output_path = Path(args.output)
    
    # Prepare results dict
    results_dict = {
        'N': args.N,
        'num_queries': len(query_embeddings),
        'methods': {}
    }
    
    for method in all_results.keys():
        results_dict['methods'][method] = {
            **tracker.metrics[method],
            'results': all_results[method]
        }
    
    # Check output format
    if output_path.suffix == '.txt':
        # Format as text
        format_output_txt(
            results_dict,
            args.output,
            database_ids=database_ids,
            query_ids=query_ids,
            database_seqs=database_seqs,
            query_seqs=query_seqs,
            blast_results=blast_results,
            N=args.N,
            display_n=10
        )
        print(f"Results saved to {args.output} (formatted text)")
    
    else:
        # Save as JSON (directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save comparison
        comparison_file = output_path / 'comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        print(f"Results saved to {output_path}/")
    
    print(f"\n{'='*70}")
    print("Protein Search Completed")
    print(f"{'='*70}\n")


def compute_recall_at_n(
    ann_results: List[List[Tuple[int, float]]],
    blast_results: Dict[int, List[Tuple[int, float, float]]],
    N: int
) -> float:

    total_recall = 0.0
    num_queries = 0
    
    for query_idx, ann_neighbors in enumerate(ann_results):
        if query_idx not in blast_results:
            continue
        
        # Get BLAST top-N indices
        blast_top_n = set([hit_idx for hit_idx, _, _ in blast_results[query_idx][:N]])
        
        if not blast_top_n:
            continue
        
        # Get ANN top-N indices
        ann_top_n = set([idx for idx, _ in ann_neighbors[:N]])
        
        # Compute recall
        intersection = len(blast_top_n & ann_top_n)
        recall = intersection / len(blast_top_n)
        
        total_recall += recall
        num_queries += 1
    
    if num_queries == 0:
        return 0.0
    
    return total_recall / num_queries


if __name__ == '__main__':
    main()