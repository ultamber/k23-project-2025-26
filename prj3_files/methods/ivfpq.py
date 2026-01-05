import numpy as np
from typing import List, Tuple
import time

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class IVFPQ:
    def __init__(self, n_clusters=None, n_probe=10, M=8, nbits=8, metric='L2', seed=42):
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS required! pip install faiss-cpu")
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.M = M
        self.nbits = nbits
        self.metric = metric
        self.seed = seed
        self.dim = None
        self.n = None
        self.index = None
        self.embeddings = None
        self.build_time = None
    
    def build_index(self, embeddings):
        t0 = time.time()
        self.n, self.dim = embeddings.shape
        self.embeddings = embeddings.astype(np.float32)
        
        if self.n_clusters is None:
            self.n_clusters = max(16, int(np.sqrt(self.n)))
            self.n_clusters = min(self.n_clusters, self.n // 10, 10000)
        
        print(f"Building IVF-PQ index:")
        print(f"  N = {self.n} proteins")
        print(f"  D = {self.dim} dimensions")
        print(f"  k = {self.n_clusters} clusters")
        print(f"  M = {self.M} subquantizers")
        print(f"  n_probe = {self.n_probe}")
        
        quantizer = faiss.IndexFlatL2(self.dim)
        self.index = faiss.IndexIVFPQ(quantizer, self.dim, self.n_clusters, self.M, self.nbits, faiss.METRIC_L2)
        
        train_size = min(self.n, max(self.n_clusters * 40, int(np.sqrt(self.n))))
        train_indices = np.random.RandomState(self.seed).choice(self.n, train_size, replace=False)
        train_data = self.embeddings[train_indices]
        
        print(f"  Training k-means + PQ on {train_size} points...")
        self.index.train(train_data)
        print(f"  Adding all {self.n} points...")
        self.index.add(self.embeddings)
        self.index.nprobe = self.n_probe
        
        self.build_time = time.time() - t0
        print(f"IVF-PQ index built in {self.build_time:.2f}s!")
    
    def search(self, query, N=10):
        distances, indices = self.index.search(query, N)
        return [(int(indices[0][i]), float(np.sqrt(distances[0][i])))
                for i in range(len(indices[0])) if indices[0][i] != -1]
    
    def batch_search(self, queries, N=10, verbose=True):
        queries = queries.astype(np.float32)
        distances, indices = self.index.search(queries, N)
        results = []
        for qi in range(len(queries)):
            query_results = [(int(indices[qi][i]), float(np.sqrt(distances[qi][i])))
                        for i in range(len(indices[qi])) if indices[qi][i] != -1]
            results.append(query_results)
        return results