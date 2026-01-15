from itertools import combinations
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import time

# TODO : change metric to support cosine similarity

class LSH:

    def __init__(
        self,
        L: int = 10,
        k: int = 4,
        w: float = 4.0,
        tableSize: int = None,
        seed: int = 42,
        metric: str = 'L2'
    ):

        self.L = L
        self.k = k
        self.w = w
        self.tableSize = tableSize
        self.seed = seed
        self.metric = metric.lower()
        
        self.dim = None
        self.n = None
        self.a_ = None
        self.t_ = None
        self.r_ = None
        self.tables_ = None
        
        self.MOD_M = 2**31 - 1
        
        self.rng = np.random.RandomState(seed)
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            return vectors / max(norm, 1e-8)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return vectors / norms

    def build_index(self, embeddings: np.ndarray):

        self.n, self.dim = embeddings.shape
        self.embeddings = embeddings.astype(np.float32)
        # Normalize for cosine similarity
        if self.metric == 'cosine':
            self.embeddings = self._normalize(self.embeddings)

        if self.tableSize is None:
            self.tableSize = max(1, self.n // 4)
        
        print(f"Building LSH index:")
        print(f"N = {self.n} proteins")
        print(f"D = {self.dim} dimensions")
        print(f"L = {self.L} tables")
        print(f"k = {self.k} hash functions")
        print(f"w = {self.w}")
        print(f"tableSize = {self.tableSize}")
        
        self._generate_hash_functions()
        
        # Initialize hash tables
        self.tables_ = [defaultdict(list) for _ in range(self.L)]
        
        # Insert all points into hash tables
        print("  Inserting points...")
        for idx in range(self.n):
            vec = self.embeddings[idx]
            for li in range(self.L):
                # Compute ID(p) using hash function
                ID_p = self._compute_ID(vec, li)
                
                # Compute bucket index g(p) = ID(p) mod tableSize
                g = ID_p % self.tableSize
                
                # Store (point_id, ID) for filtering
                self.tables_[li][g].append((idx, ID_p))
        
        print(f"LSH index built!")
    
    def _generate_hash_functions(self):
        if self.metric == 'cosine':
            # SimHash: random hyperplanes
            self.a_ = self.rng.randn(self.L, self.k, self.dim).astype(np.float32)
            # No shifts or random coefficients needed for SimHash
        else:
            # p-stable LSH for L2
            self.a_ = self.rng.randn(self.L, self.k, self.dim).astype(np.float32)
            self.t_ = self.rng.uniform(0, self.w, size=(self.L, self.k)).astype(np.float32)
            self.r_ = self.rng.randint(1, self.MOD_M, size=(self.L, self.k), dtype=np.int64)
    
    def _compute_ID(self, v: np.ndarray, li: int) -> int:
        if self.metric == 'cosine':
            # SimHash: sign of dot product
            ID = 0
            for j in range(self.k):
                dot = np.dot(self.a_[li][j], v)
                if dot >= 0:
                    ID |= (1 << j)
            return ID
        else:
            # p-stable LSH for L2
            ID = 0
            for j in range(self.k):
                dot = np.dot(self.a_[li][j], v)
                h_j = int(np.floor((dot + self.t_[li][j]) / self.w))
                ID = (ID + self.r_[li][j] * h_j) % self.MOD_M
        return ID
    
    def _compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return 1.0 - sim
        else:
            return np.linalg.norm(v1 - v2)
        
    def _key_for(self, v: np.ndarray, li: int) -> int:

        ID_v = self._compute_ID(v, li)
        return ID_v % self.tableSize
    
    def search(
        self,
        query: np.ndarray,
        N: int = 10,
        multi_probe: bool = True,
        probe_range: int = 2
    ) -> List[Tuple[int, float]]:
        if self.metric == 'cosine':
            query = self._normalize(query)
        candidates = []
        examined = 0
        
        # Hard cap on candidates (from C++: 10L to 20L)
        hard_cap = 10 * self.L
        
        # Probe all L tables
        for li in range(self.L):
            # Compute query's hash value
            ID_q = self._compute_ID(query, li)
            g_q = ID_q % self.tableSize
            
            # Multi-probe: check main bucket + neighboring buckets
            if multi_probe and self.metric == 'l2':
                probe_deltas = range(-probe_range, probe_range + 1)
            elif multi_probe and self.metric == 'cosine':
                # For SimHash, probe nearby Hamming distances
                probe_deltas = []
                for h in range(1, probe_range + 1):
                    for positions in combinations(range(self.k), h):
                        delta = 0
                        for pos in positions:
                            delta |= (1 << pos)
                        probe_deltas.append(delta)
            else:
                probe_deltas = [0]
            
            for delta in probe_deltas:
                g_q2 = (g_q + delta) % self.tableSize
                
                if g_q2 not in self.tables_[li]:
                    continue
                
                bucket = self.tables_[li][g_q2]
                
                for idx, ID_p in bucket:
                    # Neighboring buckets: accept all candidates
                    if delta != 0:
                        candidates.append(idx)
                        examined += 1
                    # Exact bucket: use ID filtering
                    elif ID_p == ID_q:
                        candidates.append(idx)
                        examined += 1
                    
                    if examined > hard_cap:
                        break
                
                if examined > hard_cap:
                    break
            
            if examined > hard_cap:
                break
        
        # Deduplicate candidates
        candidates = list(set(candidates))
        
        # Compute actual distances
        distances = []
        for idx in candidates:
            dist = self._compute_distance(query, self.embeddings[idx])
            distances.append((idx, dist))
        
        # Sort by distance and return top N
        distances.sort(key=lambda x: x[1])
        return distances[:N]
    
    def batch_search(
        self,
        queries: np.ndarray,
        N: int = 10,
        multi_probe: bool = True,
        verbose: bool = True
    ) -> List[List[Tuple[int, float]]]:

        results = []
        num_queries = len(queries)
        
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(range(num_queries), desc="LSH search")
        else:
            iterator = range(num_queries)
        
        for qi in iterator:
            result = self.search(queries[qi], N=N, multi_probe=multi_probe)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        bucket_sizes = []
        for li in range(self.L):
            for bucket in self.tables_[li].values():
                bucket_sizes.append(len(bucket))
        
        return {
            'n_points': self.n,
            'n_tables': self.L,
            'k_functions': self.k,
            'table_size': self.tableSize,
            'avg_bucket_size': np.mean(bucket_sizes) if bucket_sizes else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
            'total_buckets': sum(len(table) for table in self.tables_),
        }


if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path
    import time
    
    parser = argparse.ArgumentParser(
        description="Euclidean LSH Search for Protein Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required
    parser.add_argument('-d', '--database', required=True, help='Database embeddings (.npy)')
    parser.add_argument('-q', '--queries', required=True, help='Query embeddings (.npy)')
    parser.add_argument('-o', '--output', required=True, help='Output results file (.pkl)')
    
    # Search parameters
    parser.add_argument('--N', type=int, default=50, help='Number of neighbors (default: 50)')
    
    # LSH parameters
    parser.add_argument('--L', type=int, default=10, help='Number of hash tables (default: 10)')
    parser.add_argument('--k', type=int, default=4, help='Hash functions per table (default: 4)')
    parser.add_argument('--w', type=float, default=4.0, help='Width parameter (default: 4.0)')
    
    # Options
    parser.add_argument('--max-queries', type=int, help='Limit number of queries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save-index', help='Save built index to file')
    parser.add_argument('--load-index', help='Load pre-built index from file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Lsh for Protein Embeddings")
    print("="*70)
    
    print(f"\nLoading embeddings...")
    db_embeddings = np.load(args.database)
    query_embeddings = np.load(args.queries)
    
    print(f"Database: {db_embeddings.shape}")
    print(f"Queries: {query_embeddings.shape}")
    
    if args.max_queries:
        query_embeddings = query_embeddings[:args.max_queries]
        print(f"Limited to {len(query_embeddings)} queries")
    
    # Build or load index
    if args.load_index and Path(args.load_index).exists():
        print(f"\nLoading pre-built index from {args.load_index}...")
        with open(args.load_index, 'rb') as f:
            lsh = pickle.load(f)
        print(f"Index loaded")
    else:
        print(f"\nBuilding LSH index...")
        print(f"L={args.L}, k={args.k}, w={args.w}")
        
        lsh = LSH(L=args.L, k=args.k, w=args.w, seed=args.seed)
        
        t0 = time.time()
        lsh.build_index(db_embeddings)
        build_time = time.time() - t0
        
        print(f"Index built in {build_time:.2f}s")
        
        # Save index if requested
        if args.save_index:
            print(f"Saving index to {args.save_index}...")
            Path(args.save_index).parent.mkdir(parents=True, exist_ok=True)
            with open(args.save_index, 'wb') as f:
                pickle.dump(lsh, f)
            print(f"Index saved")
    
    # Search
    print(f"\nSearching {len(query_embeddings)} queries...")
    
    t0 = time.time()
    results = lsh.batch_search(query_embeddings, N=args.N, verbose=True)
    search_time = time.time() - t0
    
    qps = len(query_embeddings) / search_time if search_time > 0 else 0
    
    print(f"Search completed in {search_time:.2f}s")
    print(f"QPS: {qps:.2f}")
    
    print(f"\nSaving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'results': results,
        'params': {
            'method': 'lsh',
            'L': args.L,
            'k': args.k,
            'w': args.w,
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
    
    print(f"Results saved to {output_path}")
    
    
    print("Summary:")
    print(f"{'='*70}")
    print(f"Database: {len(db_embeddings)} proteins")
    print(f"Queries: {len(query_embeddings)}")
    print(f"Top-N: {args.N}")
    print(f"QPS: {qps:.2f}")
    print(f"Avg query time: {search_time/len(query_embeddings)*1000:.2f} ms")
    
    print(f"\nSample results (first query):")
    for i, (idx, dist) in enumerate(results[0][:5], 1):
        print(f"{i}. Index {idx}: distance = {dist:.4f}")
    
    
    print("LSH Search Completed")
    