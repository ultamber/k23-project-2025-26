import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import combinations


class Hypercube:

    def __init__(
        self,
        kproj: int = None,
        w: float = 4.0,
        max_probes: int = 100,
        max_hamming: int = 3,
        M: int = 1000,  # Max candidates to examine
        seed: int = 42
    ):

        self.kproj = kproj
        self.w = w
        self.max_probes = max_probes
        self.max_hamming = max_hamming
        self.M = M
        self.seed = seed
        
        self.dim = None
        self.n = None
        self.proj_ = None
        self.shift_ = None
        self.fA_ = None
        self.fB_ = None
        self.cube_ = None
        
        self.rng = np.random.RandomState(seed)
    
    def build_index(self, embeddings: np.ndarray):

        self.n, self.dim = embeddings.shape
        self.embeddings = embeddings
        
        if self.kproj is None:
            dlog = int(np.floor(np.log2(max(1, self.n)))) if self.n > 0 else 1
            self.kproj = max(1, dlog - 2)

        self.kproj = min(self.kproj, 24)
        
        print(f"Building Hypercube index:")
        print(f"  N = {self.n} proteins")
        print(f"  D = {self.dim} dimensions")
        print(f"  d' = {self.kproj} (hypercube dimension)")
        print(f"  w = {self.w}")
        print(f"  max_probes = {self.max_probes}")
        print(f"  max_hamming = {self.max_hamming}")
        
        self._generate_hash_functions()

        self.cube_ = defaultdict(list)

        print("  Inserting points...")
        for idx in range(self.n):
            vec = embeddings[idx]
            vertex = self._vertex_of(vec)
            self.cube_[vertex].append(idx)
        
        print(f"Hypercube index built!")
        print(f"  Total vertices occupied: {len(self.cube_)}")
    
    def _generate_hash_functions(self):

        self.proj_ = self.rng.randn(self.kproj, self.dim).astype(np.float32)
        self.shift_ = self.rng.uniform(0, self.w, size=self.kproj).astype(np.float32)
        
        
        self.fA_ = self.rng.randint(1, 2**32 - 1, size=self.kproj, dtype=np.uint64)
        self.fB_ = self.rng.randint(0, 2**32 - 1, size=self.kproj, dtype=np.uint64)
    
    def _h_ij(self, v: np.ndarray, j: int) -> int:

        dot = np.dot(self.proj_[j], v)
        return int(np.floor((dot + self.shift_[j]) / self.w))
    
    def _f_j(self, h: int, j: int) -> bool:

        h_val = abs(h)
        return ((self.fA_[j] * h_val + self.fB_[j]) & 1) != 0
    
    def _vertex_of(self, v: np.ndarray) -> int:

        key = 0
        for j in range(self.kproj):
            h_j = self._h_ij(v, j)
            if self._f_j(h_j, j):
                key |= (1 << j)
        return key
    
    def _probes_list(self, base: int) -> List[int]:

        probes = [base]
        
        if len(probes) >= self.max_probes:
            return probes

        H_max = min(self.kproj, self.max_hamming)

        
        for h in range(1, H_max + 1):
            if len(probes) >= self.max_probes:
                break
            
            for positions in combinations(range(self.kproj), h):
                mask = 0
                for pos in positions:
                    mask |= (1 << pos)
                
                probes.append(base ^ mask)
                
                if len(probes) >= self.max_probes:
                    break
        
        return probes
    
    def search(
        self,
        query: np.ndarray,
        N: int = 10
    ) -> List[Tuple[int, float]]:

        base = self._vertex_of(query)
        probe_list = self._probes_list(base)
        candidates = set()
        gathered = 0
        
        for vertex in probe_list:
            if vertex not in self.cube_:
                continue
            
            for idx in self.cube_[vertex]:
                if idx not in candidates:
                    candidates.add(idx)
                    gathered += 1
                    
                    if gathered >= self.M:
                        break
            
            if gathered >= self.M:
                break
        
        distances = []
        for idx in candidates:
            dist = np.linalg.norm(query - self.embeddings[idx])
            distances.append((idx, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:N]
    
    def batch_search(
        self,
        queries: np.ndarray,
        N: int = 10,
        verbose: bool = True
    ) -> List[List[Tuple[int, float]]]:

        results = []
        num_queries = len(queries)
        
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(range(num_queries), desc="Hypercube search")
        else:
            iterator = range(num_queries)
        
        for qi in iterator:
            result = self.search(queries[qi], N=N)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        bucket_sizes = [len(v) for v in self.cube_.values()]
        
        hamming_weights = {}
        for vertex in self.cube_.keys():
            weight = bin(vertex).count('1')
            hamming_weights[weight] = hamming_weights.get(weight, 0) + 1
        
        return {
            'n_points': self.n,
            'kproj': self.kproj,
            'total_vertices': 2**self.kproj,
            'occupied_vertices': len(self.cube_),
            'occupancy_rate': len(self.cube_) / (2**self.kproj),
            'avg_bucket_size': np.mean(bucket_sizes) if bucket_sizes else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
            'hamming_weight_distribution': hamming_weights,
        }


if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path
    import time
    
    parser = argparse.ArgumentParser(
        description="Hypercube Search for Protein Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument('-d', '--database', required=True, help='Database embeddings (.npy)')
    parser.add_argument('-q', '--queries', required=True, help='Query embeddings (.npy)')
    parser.add_argument('-o', '--output', required=True, help='Output results file (.pkl)')
    
    # Search parameters
    parser.add_argument('--N', type=int, default=50, help='Number of neighbors (default: 50)')
    
    # Hypercube parameters
    parser.add_argument('--kproj', type=int, help='Projection dimension (auto if not set)')
    parser.add_argument('--w', type=float, default=4.0, help='Width parameter (default: 4.0)')
    parser.add_argument('--max-probes', type=int, default=100, help='Max probes (default: 100)')
    parser.add_argument('--max-hamming', type=int, default=3, help='Max Hamming distance (default: 3)')
    
    # Options
    parser.add_argument('--max-queries', type=int, help='Limit number of queries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save-index', help='Save built index to file')
    parser.add_argument('--load-index', help='Load pre-built index from file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Ηypercube Search for Protein Embeddings")
    print("="*70)
    
    print(f"\nLoading embeddings...")
    db_embeddings = np.load(args.database)
    query_embeddings = np.load(args.queries)
    
    print(f"  Database: {db_embeddings.shape}")
    print(f"  Queries: {query_embeddings.shape}")
    
    if args.max_queries:
        query_embeddings = query_embeddings[:args.max_queries]
        print(f"  Limited to {len(query_embeddings)} queries")
    
    # Build or load index
    if args.load_index and Path(args.load_index).exists():
        print(f"\nLoading pre-built index from {args.load_index}...")
        with open(args.load_index, 'rb') as f:
            hc = pickle.load(f)
        print(f"  Index loaded")
    else:
        print(f"\Building Hypercube index...")
        print(f"  kproj={args.kproj or 'auto'}, w={args.w}")
        print(f"  max_probes={args.max_probes}, max_hamming={args.max_hamming}")
        
        hc = Hypercube(
            kproj=args.kproj,
            w=args.w,
            max_probes=args.max_probes,
            max_hamming=args.max_hamming,
            seed=args.seed
        )
        
        t0 = time.time()
        hc.build_index(db_embeddings)
        build_time = time.time() - t0
        
        print(f"  Index built in {build_time:.2f}s")
        
        # Save index if requested
        if args.save_index:
            print(f"  Saving index to {args.save_index}...")
            Path(args.save_index).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_index, 'wb') as f:
                pickle.dump(hc, f)
            print(f"  Index saved")
    
    # Search
    print(f"\nSearching {len(query_embeddings)} queries...")
    
    t0 = time.time()
    results = hc.batch_search(query_embeddings, N=args.N, verbose=True)
    search_time = time.time() - t0
    
    qps = len(query_embeddings) / search_time if search_time > 0 else 0
    
    print(f"  Search completed in {search_time:.2f}s")
    print(f"  QPS: {qps:.2f}")
    
    # Save results
    print(f"\nSaving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'results': results,
        'params': {
            'method': 'hypercube',
            'kproj': hc.kproj,
            'w': args.w,
            'max_probes': args.max_probes,
            'max_hamming': args.max_hamming,
            'N': args.N,
            'num_queries': len(query_embeddings),
            'num_database': len(db_embeddings),
        },
        'metrics': {
            'search_time': search_time,
            'qps': qps,
            'avg_query_time': search_time / len(query_embeddings),
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"  Results saved to {output_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Database: {len(db_embeddings)} proteins")
    print(f"Queries: {len(query_embeddings)}")
    print(f"Top-N: {args.N}")
    print(f"Hypercube dimension: {hc.kproj}")
    print(f"QPS: {qps:.2f}")
    print(f"Average query time: {search_time/len(query_embeddings)*1000:.2f} ms")
    
    # Sample results
    print(f"\nSample results (first query):")
    for i, (idx, dist) in enumerate(results[0][:5], 1):
        print(f"  {i}. Index {idx}: distance = {dist:.4f}")
    
    print(f"\n{'='*70}")
    print("Hypercube Search Completed")
    print(f"{'='*70}\n")