import subprocess
import tempfile
import os
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import time
import re
from utils.protein_fvecs import save_fvecs, load_fvecs

def parse_cpp_output(output_file: str) -> Dict:

    results = {
        'method': None,
        'queries': [],
        'summary': {}
    }
    
    current_query = None
    in_r_neighbors = False  # Flag for R-near neighbors section
    
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                in_r_neighbors = False
                continue
            
            # First non-empty line that's not a query is the method name
            if results['method'] is None and not line.startswith('Query'):
                results['method'] = line
                continue
            
            # Start of new query section
            if line.startswith('Query:'):
                if current_query is not None:
                    results['queries'].append(current_query)  # Save previous query
                current_query = {
                    'query_idx': int(line.split(':')[1].strip()),
                    'neighbors': [],
                    'distances_approx': [],
                    'distances_true': [],
                    'r_neighbors': []
                }
                in_r_neighbors = False
                continue
            
            # Parse neighbor indices
            if line.startswith('Nearest neighbor-'):
                idx = int(line.split(':')[1].strip())
                current_query['neighbors'].append(idx)
                continue
            
            # Parse approximate distances
            if line.startswith('distanceApproximate:'):
                current_query['distances_approx'].append(float(line.split(':')[1].strip()))
                continue
            
            # Parse true distances
            if line.startswith('distanceTrue:'):
                current_query['distances_true'].append(float(line.split(':')[1].strip()))
                continue
            
            # R-near neighbors section (for range search)
            if line.startswith('R-near neighbors:'):
                in_r_neighbors = True
                continue
            
            if in_r_neighbors and line.isdigit():
                current_query['r_neighbors'].append(int(line))
                continue
            
            # Summary statistics section
            if line.startswith('Average AF:'):
                results['summary']['avg_af'] = float(line.split(':')[1].strip())
            elif line.startswith('Recall@N:'):
                results['summary']['recall_at_n'] = float(line.split(':')[1].strip())
            elif line.startswith('QPS:'):
                results['summary']['qps'] = float(line.split(':')[1].strip())
            elif line.startswith('tApproximateAverage:'):
                results['summary']['avg_time_approx'] = float(line.split(':')[1].strip())
            elif line.startswith('tTrueAverage:'):
                results['summary']['avg_time_true'] = float(line.split(':')[1].strip())
            elif line.startswith('Silhouette Score:'):
                results['summary']['silhouette'] = float(line.split(':')[1].strip())
    
    if current_query is not None:
        results['queries'].append(current_query)
    
    return results


def convert_to_ann_results(parsed: Dict) -> List[List[Tuple[int, float]]]:
    """
    Convert parsed C++ output to standard ANN results format.
    
    Standard format: List[List[Tuple[int, float]]] where each inner list
    contains (neighbor_index, distance) tuples for one query.
    """
    results = []
    for query in parsed['queries']:
        query_results = []
        for i, idx in enumerate(query['neighbors']):
            # Use approximate distance if available, otherwise 0.0
            dist = query['distances_approx'][i] if i < len(query['distances_approx']) else 0.0
            query_results.append((idx, dist))
        results.append(query_results)
    return results

class CppSearchWrapper:

    def __init__(
        self,
        binary_path: str = "../bin/search",
        nlsh_script: str = "../neural_lsh/nlsh_search.py",
        dataset_type: str = "sift",
        temp_dir: Optional[str] = None
    ):

        self.binary_path = binary_path
        self.nlsh_script = nlsh_script
        self.dataset_type = dataset_type
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix='cpp_search_')
        
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    def _prepare_data(
        self,
        database: np.ndarray,
        queries: np.ndarray,
        database_file: Optional[str] = None,
        query_file: Optional[str] = None
    ) -> Tuple[str, str]:

        # Database
        if isinstance(database, np.ndarray):
            db_path = database_file or os.path.join(self.temp_dir, 'database.fvecs')
            save_fvecs(db_path, database)
        else:
            db_path = database
        
        # Queries
        if isinstance(queries, np.ndarray):
            q_path = query_file or os.path.join(self.temp_dir, 'queries.fvecs')
            save_fvecs(q_path, queries)
        else:
            q_path = queries
        
        return db_path, q_path
    
    def _run_command(self, cmd: List[str], verbose: bool = True) -> Tuple[str, str, float]:
        """Run a command and return stdout, stderr, and elapsed time."""
        if verbose:
            print(f"  Running: {' '.join(cmd)}")
        
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode != 0:
            print(f"  ERROR: Command failed with code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            raise RuntimeError(f"C++ search failed: {result.stderr}")
        
        return result.stdout, result.stderr, elapsed
    
    def search_lsh(
        self,
        database,
        queries,
        L: int = 10,
        k: int = 4,
        w: float = 1.0,
        N: int = 50,
        R: float = None,
        range_search: bool = False,
        output_file: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:

        db_path, q_path = self._prepare_data(database, queries)
        output_file = output_file or os.path.join(self.temp_dir, 'lsh_results.txt')
        
        # Build command line arguments for C++ binary
        cmd = [
            self.binary_path,
            '-d', db_path,
            '-q', q_path,
            '-o', output_file,
            '-type', self.dataset_type,
            '-lsh',
            '-L', str(L),
            '-k', str(k),
            '-w', str(w),
            '-N', str(N)
        ]
        
        # Optional range search parameters
        if range_search and R is not None:
            cmd.extend(['-R', str(R), '-range', 'true'])
        
        # Optional ground truth for evaluation
        if ground_truth_file:
            cmd.extend(['-gt', ground_truth_file])
        
        # Execute command and parse results
        stdout, stderr, elapsed = self._run_command(cmd, verbose)
        parsed = parse_cpp_output(output_file)
        results = convert_to_ann_results(parsed)
        
        return {
            'results': results,
            'summary': parsed['summary'],
            'method': 'LSH',
            'params': {'L': L, 'k': k, 'w': w, 'N': N},
            'elapsed': elapsed,
            'output_file': output_file
        }
    
    def search_hypercube(
        self,
        database,
        queries,
        kproj: int = 14,
        M: int = 5000,
        probes: int = 100,
        w: float = 1.5,
        N: int = 50,
        R: float = None,
        range_search: bool = False,
        output_file: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:

        db_path, q_path = self._prepare_data(database, queries)
        output_file = output_file or os.path.join(self.temp_dir, 'hypercube_results.txt')
        
        cmd = [
            self.binary_path,
            '-d', db_path,
            '-q', q_path,
            '-o', output_file,
            '-type', self.dataset_type,
            '-hypercube',
            '-kproj', str(kproj),
            '-M', str(M),
            '-probes', str(probes),
            '-w', str(w),
            '-N', str(N)
        ]
        
        if range_search and R is not None:
            cmd.extend(['-R', str(R), '-range', 'true'])
        
        if ground_truth_file:
            cmd.extend(['-gt', ground_truth_file])
        
        stdout, stderr, elapsed = self._run_command(cmd, verbose)
        parsed = parse_cpp_output(output_file)
        results = convert_to_ann_results(parsed)
        
        return {
            'results': results,
            'summary': parsed['summary'],
            'method': 'Hypercube',
            'params': {'kproj': kproj, 'M': M, 'probes': probes, 'w': w, 'N': N},
            'elapsed': elapsed,
            'output_file': output_file
        }
    
    def search_ivfflat(
        self,
        database,
        queries,
        kclusters: int = 100,
        nprobe: int = 10,
        N: int = 50,
        R: float = None,
        range_search: bool = False,
        output_file: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:

        db_path, q_path = self._prepare_data(database, queries)
        output_file = output_file or os.path.join(self.temp_dir, 'ivfflat_results.txt')
        
        cmd = [
            self.binary_path,
            '-d', db_path,
            '-q', q_path,
            '-o', output_file,
            '-type', self.dataset_type,
            '-ivfflat',
            '-kclusters', str(kclusters),
            '-nprobe', str(nprobe),
            '-N', str(N)
        ]
        
        if range_search and R is not None:
            cmd.extend(['-R', str(R), '-range', 'true'])
        
        if ground_truth_file:
            cmd.extend(['-gt', ground_truth_file])
        
        stdout, stderr, elapsed = self._run_command(cmd, verbose)
        parsed = parse_cpp_output(output_file)
        results = convert_to_ann_results(parsed)
        
        return {
            'results': results,
            'summary': parsed['summary'],
            'method': 'IVFFlat',
            'params': {'kclusters': kclusters, 'nprobe': nprobe, 'N': N},
            'elapsed': elapsed,
            'output_file': output_file
        }
    
    def search_ivfpq(
        self,
        database,
        queries,
        kclusters: int = 100,
        nprobe: int = 10,
        Msub: int = 8,
        nbits: int = 8,
        N: int = 50,
        R: float = None,
        range_search: bool = False,
        output_file: Optional[str] = None,
        ground_truth_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        db_path, q_path = self._prepare_data(database, queries)
        output_file = output_file or os.path.join(self.temp_dir, 'ivfpq_results.txt')
        
        cmd = [
            self.binary_path,
            '-d', db_path,
            '-q', q_path,
            '-o', output_file,
            '-type', self.dataset_type,
            '-ivfpq',
            '-kclusters', str(kclusters),
            '-nprobe', str(nprobe),
            '-Msub', str(Msub),
            '-nbits', str(nbits),
            '-N', str(N)
        ]
        
        if range_search and R is not None:
            cmd.extend(['-R', str(R), '-range', 'true'])
        
        if ground_truth_file:
            cmd.extend(['-gt', ground_truth_file])
        
        stdout, stderr, elapsed = self._run_command(cmd, verbose)
        parsed = parse_cpp_output(output_file)
        results = convert_to_ann_results(parsed)
        
        return {
            'results': results,
            'summary': parsed['summary'],
            'method': 'IVFPQ',
            'params': {'kclusters': kclusters, 'nprobe': nprobe, 'Msub': Msub, 'nbits': nbits, 'N': N},
            'elapsed': elapsed,
            'output_file': output_file
        }
    
    def search_neural_lsh(
        self,
        database,
        queries,
        N: int = 5,
        T: int = 75,
        output_file: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict:
        db_path, q_path = self._prepare_data(database, queries)
        output_file = output_file or os.path.join(self.temp_dir, 'nlsh_results.txt')
        index_dir = os.path.join(self.temp_dir, 'nlsh_index')
        #call nlsh build first , best parameters from previous experiments
        cmd_build = [
            'python', '../neural_lsh/nlsh_build.py',
            '-d', db_path,
            '-i', index_dir,
            '--type', self.dataset_type,
            '--knn', '25',
            '-m', '50',
            '--method', 'ivfflat',
            '--epochs', '50',
            '--layers', '4',
            '--nodes', '512',
            '--batch_size', '256',
            '--lr', '0.01'    
        ]
        cmd = [
            'python', self.nlsh_script,
            '-d', db_path,
            '-q', q_path,
            '-i', index_dir,
            '-o', output_file,
            '-type', self.dataset_type,
            '-N', str(N),
            '-T', str(T)
        ]
        stoud, stderr, elapsed = self._run_command(cmd_build, verbose)
        stdout, stderr, elapsed = self._run_command(cmd, verbose)
        parsed = parse_cpp_output(output_file)
        results = convert_to_ann_results(parsed)
        
        return {
            'results': results,
            'summary': parsed['summary'],
            'method': 'NeuralLSH',
            'params': {'N': N, 'T': T , 'k': 25, 'm': 50, 'epochs': 50, 'hidden_dims': [128, 64]},
            'elapsed': elapsed,
            'output_file': output_file
        }
    
    def search(
        self,
        method: str,
        database,
        queries,
        N: int = 50,
        verbose: bool = True,
        **kwargs
    ) -> Dict:

        method = method.lower().replace('_', '-')
        
        if method == 'lsh':
            return self.search_lsh(database, queries, N=N, verbose=verbose, **kwargs)
        elif method == 'hypercube':
            return self.search_hypercube(database, queries, N=N, verbose=verbose, **kwargs)
        elif method == 'ivfflat':
            return self.search_ivfflat(database, queries, N=N, verbose=verbose, **kwargs)
        elif method == 'ivfpq':
            return self.search_ivfpq(database, queries, N=N, verbose=verbose, **kwargs)
        elif method in ['neural-lsh', 'nlsh', 'neural']:
            return self.search_neural_lsh(database, queries, N=N, verbose=verbose, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def cleanup(self):
        import shutil
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)  # Remove temp directory


def run_cpp_method(
    method: str,
    database_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    args,
    binary_path: str = "../bin/search",
    verbose: bool = True
) -> Tuple[List[List[Tuple[int, float]]], Dict]:

    wrapper = CppSearchWrapper(binary_path=binary_path, dataset_type='sift')
    
    try:
        if method == 'lsh':
            result = wrapper.search_lsh(
                database=database_embeddings,
                queries=query_embeddings,
                L=args.lsh_L,
                k=args.lsh_k,
                w=args.lsh_w,
                N=args.N,
                verbose=verbose
            )
        elif method == 'hypercube':
            result = wrapper.search_hypercube(
                database=database_embeddings,
                queries=query_embeddings,
                kproj=args.hc_kproj,
                w=args.hc_w,
                M=args.hc_M,
                probes=args.hc_max_probes,
                N=args.N,
                verbose=verbose
            )
        elif method == 'ivfflat':
            result = wrapper.search_ivfflat(
                database=database_embeddings,
                queries=query_embeddings,
                kclusters=args.ivf_n_clusters,
                nprobe=args.ivf_n_probe,
                N=args.N,
                verbose=verbose
            )
        elif method == 'ivfpq':
            result = wrapper.search_ivfpq(
                database=database_embeddings,
                queries=query_embeddings,
                kclusters=args.ivf_n_clusters,
                nprobe=args.ivf_n_probe,
                Msub=args.ivf_M,
                nbits=args.ivf_nbits,
                N=args.N,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert C++ results to the metrics format expected by protein_search.py
        metrics = {
            'build_time': 0,  # C++ build time included in search time
            'search_time': result['elapsed'],
            'qps': len(query_embeddings) / result['elapsed'] if result['elapsed'] > 0 else 0,
            **result['summary']  # Include parsed summary statistics
        }
        
        return result['results'], metrics
    
    finally:
        wrapper.cleanup()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test C++ search wrapper")
    parser.add_argument('-d', '--database', required=True, help='Database .fvecs file')
    parser.add_argument('-q', '--queries', required=True, help='Query .fvecs file')
    parser.add_argument('-m', '--method', default='lsh', 
                        choices=['lsh', 'hypercube', 'ivfflat', 'ivfpq'])
    parser.add_argument('-b', '--binary', default='../bin/search', help='C++ binary path')
    parser.add_argument('-N', type=int, default=50, help='Number of neighbors')
    
    args = parser.parse_args()
    
    # Load data from fvecs files
    print(f"Loading database from {args.database}")
    database = load_fvecs(args.database)
    print(f"Database shape: {database.shape}")
    
    print(f"Loading queries from {args.queries}")
    queries = load_fvecs(args.queries)
    print(f"Queries shape: {queries.shape}")
    
    # Run search using the wrapper
    wrapper = CppSearchWrapper(binary_path=args.binary)
    
    result = wrapper.search(
        method=args.method,
        database=database,
        queries=queries,
        N=args.N
    )
    
    print(f"\nResults:")
    print(f"  Method: {result['method']}")
    print(f"  Elapsed: {result['elapsed']:.3f}s")
    print(f"  Summary: {result['summary']}")
    print(f"  Num queries: {len(result['results'])}")
    
    if result['results']:
        print(f"\n  First query neighbors: {result['results'][0][:5]}")
    
    wrapper.cleanup()