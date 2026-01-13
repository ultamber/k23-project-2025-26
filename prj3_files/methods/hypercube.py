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
        M: int = 1000,
        metric: str = 'L2',
        seed: int = 42
    ):
        self.kproj = kproj
        self.w = w
        self.max_probes = max_probes
        self.max_hamming = max_hamming
        self.M = M
        self.metric = metric.lower()
        self.seed = seed
        
        self.dim = None
        self.n = None
        self.proj_ = None
        self.shift_ = None
        self.fA_ = None
        self.fB_ = None
        self.cube_ = None
        self.embeddings = None
        
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
        
        if self.kproj is None:
            dlog = int(np.floor(np.log2(max(1, self.n)))) if self.n > 0 else 1
            self.kproj = max(1, dlog - 2)
        self.kproj = min(self.kproj, 24)
        
        print(f"Building Hypercube index:")
        print(f"  N = {self.n} proteins")
        print(f"  D = {self.dim} dimensions")
        print(f"  d' = {self.kproj} (hypercube dimension)")
        print(f"  metric = {self.metric}")
        if self.metric == 'l2':
            print(f"  w = {self.w}")
        print(f"  max_probes = {self.max_probes}")
        print(f"  max_hamming = {self.max_hamming}")
        
        self._generate_hash_functions()
        
        self.cube_ = defaultdict(list)
        
        print("  Inserting points...")
        for idx in range(self.n):
            vec = self.embeddings[idx]
            vertex = self._vertex_of(vec)
            self.cube_[vertex].append(idx)
        
        print(f"Hypercube index built!")
        print(f"Total vertices occupied: {len(self.cube_)}")
    
    def _generate_hash_functions(self):
        self.proj_ = self.rng.randn(self.kproj, self.dim).astype(np.float32)
        
        if self.metric == 'l2':
            self.shift_ = self.rng.uniform(0, self.w, size=self.kproj).astype(np.float32)
        
        self.fA_ = self.rng.randint(1, 2**32 - 1, size=self.kproj, dtype=np.uint64)
        self.fB_ = self.rng.randint(0, 2**32 - 1, size=self.kproj, dtype=np.uint64)
    
    def _h_ij(self, v: np.ndarray, j: int) -> int:
        dot = np.dot(self.proj_[j], v)
        if self.metric == 'cosine':
            # For cosine: use sign of dot product
            return 1 if dot >= 0 else 0
        else:
            return int(np.floor((dot + self.shift_[j]) / self.w))
    
    def _f_j(self, h: int, j: int) -> bool:
        if self.metric == 'cosine':
            # For cosine: h is already 0 or 1
            return h == 1
        else:
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
    
    def _compute_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if self.metric == 'cosine':
            sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return 1.0 - sim
        else:
            return np.linalg.norm(v1 - v2)
    
    def search(
        self,
        query: np.ndarray,
        N: int = 10
    ) -> List[Tuple[int, float]]:
        
        # Normalize query for cosine similarity
        if self.metric == 'cosine':
            query = self._normalize(query)
        
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
            dist = self._compute_distance(query, self.embeddings[idx])
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