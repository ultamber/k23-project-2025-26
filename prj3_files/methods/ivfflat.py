import numpy as np
from typing import List, Tuple
import time

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
class IVFFlat:
    def __init__(self, n_clusters=None, n_probe=10, metric='L2', seed=42):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS required pip install faiss-cpu")
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.metric = metric.lower()
        self.seed = seed
        self.dim = None
        self.n = None
        self.index = None
        self.embeddings = None
        self.build_time = None
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        return vectors / norms
    
    def build_index(self, embeddings):
        t0 = time.time()
        self.n, self.dim = embeddings.shape
        self.embeddings = embeddings.astype(np.float32)
        
        if self.n_clusters is None:
            self.n_clusters = max(16, int(np.sqrt(self.n)))
            self.n_clusters = min(self.n_clusters, self.n // 10, 10000)
        
        print(f"Building IVF-Flat index:")
        print(f"  N = {self.n} proteins")
        print(f"  D = {self.dim} dimensions")
        print(f"  k = {self.n_clusters} clusters")
        print(f"  n_probe = {self.n_probe}")
        print(f"  metric = {self.metric}")
        
        # Choose metric
        if self.metric == 'cosine':
            # For cosine similarity: normalize vectors + use inner product
            self.embeddings = self._normalize(self.embeddings)
            quantizer = faiss.IndexFlatIP(self.dim)  # Inner Product
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, self.n_clusters, faiss.METRIC_INNER_PRODUCT
            )
        else:
            # L2 (Euclidean) distance
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dim, self.n_clusters, faiss.METRIC_L2
            )
        
        train_size = min(self.n, max(self.n_clusters * 40, int(np.sqrt(self.n))))
        train_indices = np.random.RandomState(self.seed).choice(self.n, train_size, replace=False)
        train_data = self.embeddings[train_indices]
        
        print(f"  Training k-means on {train_size} points...")
        self.index.train(train_data)
        print(f"  Adding all {self.n} points...")
        self.index.add(self.embeddings)
        self.index.nprobe = self.n_probe
        
        self.build_time = time.time() - t0
        print(f"IVF-Flat index built in {self.build_time:.2f}s!")
    
    def search(self, query, N=10):
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Normalize query for cosine similarity
        if self.metric == 'cosine':
            query = self._normalize(query)
        
        distances, indices = self.index.search(query, N)
        
        results = []
        for i in range(len(indices[0])):
            if indices[0][i] == -1:
                continue
            
            if self.metric == 'cosine':
                # Convert inner product to cosine distance: 1 - similarity
                dist = 1.0 - distances[0][i]
            else:
                # FAISS returns squared L2, convert to L2
                dist = float(np.sqrt(distances[0][i]))
            
            results.append((int(indices[0][i]), dist))
        
        return results
    
    def batch_search(self, queries, N=10, verbose=True):
        queries = queries.astype(np.float32)
        
        # Normalize queries for cosine similarity
        if self.metric == 'cosine':
            queries = self._normalize(queries)
        
        distances, indices = self.index.search(queries, N)
        
        results = []
        for qi in range(len(queries)):
            query_results = []
            for i in range(len(indices[qi])):
                if indices[qi][i] == -1:
                    continue
                
                if self.metric == 'cosine':
                    dist = 1.0 - distances[qi][i]
                else:
                    dist = float(np.sqrt(distances[qi][i]))
                
                query_results.append((int(indices[qi][i]), dist))
            
            results.append(query_results)
        
        return results