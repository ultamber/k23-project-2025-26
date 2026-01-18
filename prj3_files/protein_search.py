import argparse
import numpy as np
from pathlib import Path
import json
import pickle
import tempfile
from typing import Dict, List, Tuple, Optional
from cpp_wrapper import CppSearchWrapper
from utils.fasta_loader import get_accession, load_fasta
from utils.evaluation import PerformanceTracker, compute_recall_at_n
from utils.output_formatter import format_output_txt
from utils.results_writer import write_method_results, write_comparison_summary
from utils.blast_runner import BLASTRunner
from utils.protein_fvecs import export_for_cpp, load_fvecs
from utils.uniprot_client import UniProtClient

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
    parser.add_argument(
        '--pfam-map',
        help='Pfam mapping file (.tsv) for biological evaluation'
    )
    parser.add_argument(
        '--export-cpp',
        help='Export embeddings to directory in .fvecs format for C++ search'
    )
    parser.add_argument(
        '--cpp-binary',
        default='../bin/search',
        help='Path to C++ search binary (default: ../bin/search)'
    )
    parser.add_argument(
        '--cpp-nlsh-script',
        default='../neural_lsh/nlsh_search.py',
        help='Path to Neural LSH Python script'
    )
    parser.add_argument(
        '--export-fvecs',
        help='Export embeddings to .fvecs format at this directory (for C++ use)'
    )
    parser.add_argument(
        '--display-n',
        type=int,
        default=10,
        help='Number of top results to display/save per query (default: 10)'
    )
    # LSH
    parser.add_argument('--lsh-L', type=int, default=50, help='LSH: number of hash tables')
    parser.add_argument('--lsh-k', type=int, default=6, help='LSH: hash functions per table')
    parser.add_argument('--lsh-w', type=float, default=1.0, help='LSH: bucket width')
    # Hypercube
    parser.add_argument('--hc-kproj', default=12, type=int, help='Hypercube: projection dimension')
    parser.add_argument('--hc-w', type=float, default=1.5, help='Hypercube: bucket width')
    parser.add_argument('--hc-max-probes', type=int, default=2, help='Hypercube: max probes')
    parser.add_argument('--hc-M', type=int, default=5000, help='Hypercube: max candidates')
    # IVF
    parser.add_argument('--ivf-n-clusters', default=1000, type=int, help='IVF: number of clusters')
    parser.add_argument('--ivf-n-probe', type=int, default=50, help='IVF: number of probes')
    parser.add_argument('--ivf-M', type=int, default=16, help='IVF-PQ: number of subvectors')
    parser.add_argument('--ivf-nbits', type=int, default=8, help='IVF-PQ: bits per subvector')
    # Neural LSH
    parser.add_argument('--nlsh-m', type=int, default=400, help='Neural LSH: partitions')
    parser.add_argument('--nlsh-k', type=int, default=25, help='Neural LSH: k-NN')
    parser.add_argument('--nlsh-hidden-dims', type=list, default=[128, 128], help='Neural LSH: hidden dimensions')
    parser.add_argument('--nlsh-T', type=int, default=50, help='Neural LSH: probes')
    parser.add_argument('--nlsh-epochs', type=int, default=15, help='Neural LSH: training epochs')
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--metric', choices=['L2', 'cosine'], default='L2',
                    help='Distance metric (default: L2)')
    return parser.parse_args()


def detect_file_type(filepath: str) -> str:
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext in ['.fasta', '.fa', '.faa']:
        return 'fasta'

    if ext in ['.npy']:
        return 'npy'

    if ext in ['.dat']:
        return 'dat'

    if ext in ['.pkl', '.pickle']:
        return 'pkl'

    return 'unknown'


def load_embeddings(filepath: str) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[Tuple[str, str]]]]:
    path = Path(filepath)

    if path.suffix == '.npy':
        embeddings = np.load(filepath)
    elif path.suffix == '.fvecs':
        embeddings = load_fvecs(filepath)
    elif path.suffix == '.dat':
        try:
            embeddings = np.load(filepath)
        except:
            embeddings = np.loadtxt(filepath)
    else:
        raise ValueError(f"Unknown embedding format: {path.suffix}")

    ids = None
    txt_file = path.with_suffix('.txt')
    ids_file = path.with_suffix('.ids')

    if txt_file.exists():
        with open(txt_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
    elif ids_file.exists():
        with open(ids_file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]

    sequences = None
    fasta_file = path.with_suffix('.fasta')
    if fasta_file.exists():
        from utils.fasta_loader import load_fasta
        sequences = load_fasta(str(fasta_file))

    return embeddings, ids, sequences

def embed_fasta(
    fasta_file: str,
    output_prefix: str = "",
    batch_size: int = 32,
    device: str = 'auto'
) -> Tuple[np.ndarray, List[str]]:
    from protein_embed import ESM2Embedder

    # Create embedder
    embedder = ESM2Embedder(
        model_name='esm2_t6_8M_UR50D',
        device=device
    )

    # Load FASTA sequences
    sequences = load_fasta(fasta_file)

    # Generate embeddings
    embeddings, ids = embedder.embed_sequences(
        sequences=sequences,
        batch_size=batch_size,
        show_progress=True
    )

    # Save if output_prefix specified
    if output_prefix != "":
        embedder.save_embeddings(embeddings, ids, output_prefix)

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

    blast = BLASTRunner(
        db_fasta=database_fasta,
        evalue_threshold=evalue
    )

    blast.build_database(database_fasta)

    results = blast.search_fasta(
        query_fasta=query_fasta,
        N=N,
        num_threads=threads
    )

    blast_results_indices = None
    if database_ids and query_ids:
        overlap = set(map(get_accession, database_ids)) & set(map(get_accession, query_ids))
        if overlap:
            print(f"WARNING: {len(overlap)} query accessions also appear in database (possible leakage)")
        db_id_to_index = {}
        db_acc_to_index = {}
        for i, pid in enumerate(database_ids):
            db_id_to_index[pid] = i
            acc = get_accession(pid)
            db_id_to_index[acc] = i
            db_acc_to_index[acc] = i
            if '|' in pid:
                parts = pid.split('|')
                if len(parts) >= 3:
                    entry_name = parts[2].split()[0]
                    db_id_to_index[entry_name] = i  # Map UniProt entry name to index
        query_id_to_index = {}
        for i, pid in enumerate(query_ids):
            query_id_to_index[pid] = i
            query_id_to_index[get_accession(pid)] = i  # Map query accession to index

        blast_results_indices = {}
        for query_id, hits in results.items():
            query_acc = get_accession(query_id)
            if query_acc in query_id_to_index:
                query_idx = query_id_to_index[query_acc]
                hit_indices = []

                for hit in hits:
                    hit_id = hit[0]
                    score = hit[1]
                    evalue_val = hit[2]
                    pident = hit[3] if len(hit) > 3 else None

                    hit_acc = get_accession(hit_id)
                    if hit_acc in db_id_to_index:
                        hit_idx = db_id_to_index[hit_acc]
                        if pident is not None:
                            hit_indices.append((hit_idx, score, evalue_val, pident))
                        else:
                            hit_indices.append((hit_idx, score, evalue_val))

                if hit_indices:
                    blast_results_indices[query_idx] = hit_indices

    blast.cleanup()

    print(f"BLAST search completed")

    return {
        'blast_results_ids': results,
        'blast_results_indices': blast_results_indices,
        'params': {'N': N, 'evalue': evalue}
    }

def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Protein Search with ANN Methods\n")

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
            temp_prefix = tempfile.mkdtemp(prefix='db_vectors_')  # Create temp directory for embeddings
            database_embeddings, database_ids = embed_fasta(
                args.database,
                output_prefix=temp_prefix,
                batch_size=args.embed_batch_size,
                device=args.embed_device
            )

            # Load sequences
            database_seqs = load_fasta(args.database)

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

    pfam_mapping = None
    if args.pfam_map:
        from utils.pfam_loader import load_pfam_mapping
        pfam_mapping = load_pfam_mapping(args.pfam_map)
    print(f"\nResolving queries...")
    uniprot_client = UniProtClient(cache_dir="uniprot_cache")
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
            temp_prefix = tempfile.mkdtemp(prefix='query_vectors_')  # Create temp directory for query embeddings
            query_embeddings, query_ids = embed_fasta(
                args.queries,
                output_prefix=temp_prefix,
                batch_size=args.embed_batch_size,
                device=args.embed_device
            )

            # Load sequences
            query_seqs = load_fasta(args.queries)

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

    if args.export_cpp:
        export_for_cpp(
            output_dir_str=args.export_cpp,
            database_embeddings=database_embeddings,
            database_ids=database_ids,
            query_embeddings=query_embeddings,
            query_ids=query_ids
        )
        print(f"\\nExported to {args.export_cpp}/ for C++ processing")

        if not args.method:
            print("Use --method to also run Python search, or use C++ binary")
            return
    # Normalize embeddings - does nothing if already normalized (from embedder) - for safety 
    database_embeddings = database_embeddings / (np.linalg.norm(database_embeddings, axis=1, keepdims=True) + 1e-8)  # L2 normalize database vectors
    query_embeddings = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)  # L2 normalize query vectors

    # Database ID mappings
    db_id_to_index = {}
    db_acc_to_index = {}
    if database_ids:
        for i, pid in enumerate(database_ids):
            db_id_to_index[pid] = i
            acc = get_accession(pid)
            db_id_to_index[acc] = i
            db_acc_to_index[acc] = i
            if '|' in pid:
                parts = pid.split('|')
                if len(parts) >= 3:
                    entry_name = parts[2].split()[0]
                    db_id_to_index[entry_name] = i  # Map UniProt entry name
        print(f"Created database ID mappings: {len(database_ids)} proteins")
    # Query ID mappings
    query_id_to_index = {}
    query_acc_to_index = {}
    if query_ids:
        for i, pid in enumerate(query_ids):
            query_id_to_index[pid] = i
            acc = get_accession(pid)
            query_acc_to_index[acc] = i  # Map query accession
        print(f"Created query ID mappings: {len(query_ids)} proteins")

    print(f"\nLoading BLAST ground truth...")

    blast_results = None
    blast_identity = None

    if args.ground_truth:
        with open(args.ground_truth, 'rb') as f:
            blast_results = pickle.load(f)
        print(f"Loaded from {args.ground_truth}")

        blast_identity = _extract_blast_identity(
            blast_results, 
            db_acc_to_index, 
            query_acc_to_index
        )
        if blast_identity:
            print(f"Extracted BLAST identity for {len(blast_identity)} queries")

    elif args.run_blast and database_fasta and query_fasta:
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

        blast_identity = _extract_blast_identity(
            blast_results,
            db_acc_to_index,
            query_acc_to_index
        )
    else:
        print(f"No ground truth available (use --run-blast or --ground-truth)")
    blast_idx = blast_results.get("blast_results_indices", {})
    if blast_idx:
        q0 = 0
        print(f"\nRunning ANN methods...")
        print(f"Method: {args.method}")
        print(f"N: {args.N}")

    tracker = PerformanceTracker()
    all_results = {}

    methods_to_run = []
    if args.method == 'all':
        methods_to_run = ['lsh', 'hypercube', 'ivfflat', 'ivfpq', 'neural-lsh']
    else:
        # Handle aliases
        method_map = {
            'neural': 'neural-lsh',
            'lsh': 'lsh',
            'hypercube':'hypercube',
            'ivfflat':'ivfflat',
            'ivfpq': 'ivfpq'
        }
        methods_to_run = [method_map.get(args.method, args.method)]

    print(f"\n  Using C++ implementations ({args.cpp_binary})")
    wrapper = CppSearchWrapper(
        binary_path=args.cpp_binary,
        nlsh_script=args.cpp_nlsh_script,
        dataset_type='sift'
    )

    for method in methods_to_run:
        if method == None:
            method = ""

        print(f"\n  Running {method.upper()} (C++)...")
        result = {}
        try:
            if method == 'lsh':
                result = wrapper.search_lsh(
                    database=database_embeddings,
                    queries=query_embeddings,
                    L=args.lsh_L,
                    k=args.lsh_k,
                    w=args.lsh_w,
                    N=args.N
                )
            elif method == 'hypercube':
                result = wrapper.search_hypercube(
                    database=database_embeddings,
                    queries=query_embeddings,
                    kproj=args.hc_kproj,
                    w=args.hc_w,
                    M=args.hc_M,
                    probes=args.hc_max_probes,
                    N=args.N
                )
            elif method == 'ivfflat':
                result = wrapper.search_ivfflat(
                    database=database_embeddings,
                    queries=query_embeddings,
                    kclusters=args.ivf_n_clusters,
                    nprobe=args.ivf_n_probe,
                    N=args.N
                )
            elif method == 'ivfpq':
                result = wrapper.search_ivfpq(
                    database=database_embeddings,
                    queries=query_embeddings,
                    kclusters=args.ivf_n_clusters,
                    nprobe=args.ivf_n_probe,
                    Msub=args.ivf_M,
                    nbits=args.ivf_nbits,
                    N=args.N
                )
            elif method in ['neural-lsh', 'neural']:
                # Neural LSH needs index directory
                result = wrapper.search_neural_lsh(
                    database=database_embeddings,
                    queries=query_embeddings,
                    T=args.nlsh_T,
                    N=args.N,

                )

            # Store results
            all_results[method] = result['results']

            # Update tracker metrics
            tracker.metrics[method] = {
                'build_time': 0,
                'search_time': result['elapsed'],
                'qps': len(query_embeddings) / result['elapsed'] if result['elapsed'] > 0 else 0,
                **result['summary']
            }

            print(f"  Completed in {result['elapsed']:.2f}s")
            print(f"  QPS: {tracker.metrics[method]['qps']:.2f}")
            if 'recall_at_n' in result['summary']:
                print(f"  Recall@N: {result['summary']['recall_at_n']:.4f}")

        except Exception as e:
            print(f"  ERROR running {method}: {e}")
            continue

    wrapper.cleanup()

    if blast_results and 'blast_results_indices' in blast_results:
        blast_indices = blast_results['blast_results_indices']
        for method, results in all_results.items():
            recall = compute_recall_at_n(results, blast_indices, args.N)
            tracker.metrics[method]['recall_at_n'] = recall
            print(f"  {method}: Recall@{args.N} = {recall:.4f}")
    else:
        print("WARNING: Skipping BLAST evaluation - no blast_results_indices!")
    print(f"\nSaving results")
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
        output_dir = output_path.parent / output_path.stem
        # Format as text
        format_output_txt(
            results_dict,
            args.output,
            database_ids=database_ids,
            query_ids=query_ids,
            database_seqs=database_seqs,
            query_seqs=query_seqs,
            blast_results=blast_results,
            pfam_mapping=pfam_mapping,
            N=args.N,
            display_n=10
        )
        print(f"Results saved to {args.output} (formatted text)")

    else:
        # Save as JSON (directory)
        output_path.mkdir(parents=True, exist_ok=True)
        output_dir = output_path
        # Save comparison
        comparison_file = output_path / 'comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"Results saved to {output_path}/")

    # Prepare individual method result files
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f""

    # Write individual method results and format names
    for method_name, method_results in all_results.items():
        method_slug = method_name.lower().replace(' ', '_').replace('-', '_')
        if(method_name == "lsh"):
            args.lsh_params = f"L{args.lsh_L}_k{args.lsh_k}_w{args.lsh_w}_"
            output_file = output_dir / f"{method_slug}_lsh_{args.lsh_params}results.txt"
        elif(method_name == "hypercube"):
            args.hc_params = f"kproj{args.hc_kproj}_w{args.hc_w}_M{args.hc_M}_probes{args.hc_max_probes}_"
            output_file = output_dir / f"{method_slug}_hc_{args.hc_params}results.txt"
        elif(method_name == "ivfflat"):
            args.ivfflat_params = f"nclusters{args.ivf_n_clusters}_nprobe{args.ivf_n_probe}_"
            output_file = output_dir / f"{method_slug}_ivfflat_{args.ivfflat_params}results.txt"
        elif(method_name == "ivfpq"):
            args.ivfpq_params = f"nclusters{args.ivf_n_clusters}_nprobe{args.ivf_n_probe}_M{args.ivf_M}_nbits{args.ivf_nbits}_"
            output_file = output_dir / f"{method_slug}_ivfpq_{args.ivfpq_params}results.txt"
        elif(method_name == "neural-lsh"):
            args.nlsh_params = f"m{args.nlsh_m}_k{args.nlsh_k}_epochs{args.nlsh_epochs}_"
            output_file = output_dir / f"{method_slug}_nlsh_{args.nlsh_params}results.txt"

        write_method_results(
            method_name=method_name,
            output_dir=str(output_dir),
            results=method_results,
            metrics=tracker.metrics.get(method_name, {}),
            query_ids=query_ids,
            database_ids=database_ids,
            blast_results=blast_results,
            pfam_mapping=pfam_mapping,
            blast_identity=blast_identity,
            query_seqs=query_seqs,
            database_seqs=database_seqs,
            per_query_times=tracker.get_per_query_times(method_name),
            N=args.N,
            display_n=args.display_n,
            save_raw_data=True,
            output_file=output_file.as_posix(),
            uniprot_client=uniprot_client,
            uniprot_delay=0.2 # rate limit
        )

    write_comparison_summary(
        output_dir=str(output_dir),
        all_metrics=tracker.metrics,
        N=args.N
    )

    print("Protein Search Completed")


def _extract_blast_identity(
    blast_results: Optional[Dict],
    db_acc_to_index: Dict[str, int],
    query_acc_to_index: Dict[str, int]
) -> Optional[Dict[int, Dict[int, float]]]:

    if not blast_results:
        return None

    blast_identity = {}

    if 'blast_results_ids' in blast_results:
        for query_id, hits in blast_results['blast_results_ids'].items():
            query_acc = get_accession(query_id)

            if query_acc not in query_acc_to_index:
                continue

            q_idx = query_acc_to_index[query_acc]
            blast_identity[q_idx] = {}

            for hit in hits:
                if len(hit) >= 4:
                    hit_id, bitscore, evalue, pident = hit[0], hit[1], hit[2], hit[3]  # Unpack hit with identity
                elif len(hit) >= 3:
                    hit_id, bitscore, evalue = hit[0], hit[1], hit[2]  # Unpack hit without identity
                    pident = None
                else:
                    continue  # Skip malformed hits

                hit_acc = get_accession(hit_id)

                if hit_acc in db_acc_to_index:
                    n_idx = db_acc_to_index[hit_acc]
                    if pident is not None:
                        blast_identity[q_idx][n_idx] = pident

    if not blast_identity or all(len(v) == 0 for v in blast_identity.values()):
        return None

    return blast_identity

if __name__ == '__main__':
    main()
