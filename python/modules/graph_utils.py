from sklearn.neighbors import NearestNeighbors
import numpy as np
import random
import signal
from contextlib import contextmanager

# -------------------------------------------------------
# Try import KaHIP – if it fails, use fallback
# -------------------------------------------------------
try:
    import kahip
    KAHIP_AVAILABLE = True
    print("[graph_utils] KaHIP library found ")
except Exception:
    KAHIP_AVAILABLE = False
    print("[graph_utils] KaHIP not available — using fallback partitioner.")

# -------------------------------------------------------
# Fallback partitioner
# -------------------------------------------------------
def fallback_partition(csr, m):
    """Simple round-robin partitioning fallback"""
    n = len(csr["vwgt"])
    labels = np.zeros(n, dtype=np.int32)
    for i in range(n):
        labels[i] = i % m
    
    # Calculate edgecut for reporting
    edgecut = 0
    for i in range(n):
        start = csr["xadj"][i]
        end = csr["xadj"][i + 1]
        bi = labels[i]
        for idx in range(start, end):
            j = csr["adjncy"][idx]
            if labels[j] != bi:
                w = csr["adjcwgt"][idx]
                edgecut += w
    edgecut = edgecut // 2
    
    print(f"[graph_utils] Fallback partitioner used. m = {m}, edgecut = {edgecut}")
    return labels


import signal
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# -------------------------------------------------------
# Main KaHIP wrapper function
# -------------------------------------------------------
def run_kahip(csr, m=100, imbalance=0.03, mode=2, seed=1, timeout=30):
    """
    Run KaHIP partitioner or fallback if not available.
    
    Args:
        csr: Dict with keys 'xadj', 'adjncy', 'adjcwgt', 'vwgt'
        m: Number of partitions
        imbalance: Allowed imbalance (default 0.03)
        mode: KaHIP mode (0=fast, 1=eco, 2=strong)
        seed: Random seed
        timeout: Timeout in seconds (default 30)
    
    Returns:
        Array of partition labels (length n)
    """
    if not KAHIP_AVAILABLE:
        return fallback_partition(csr, m)

    print("Running KaHIP partitioner…")
    print(f"  n = {len(csr['vwgt'])}, m = {m}, imbalance = {imbalance}, mode = {mode}")

    try:
        # Try with timeout to prevent hanging
        with time_limit(timeout):
            # KaHIP expects 9 positional arguments:
            # 1. vwgt (vertex weights)
            # 2. xadj (adjacency index pointer)
            # 3. adjcwgt (edge weights)
            # 4. adjncy (adjacency list)
            # 5. nblocks (int)
            # 6. imbalance (float)
            # 7. suppress_output (bool)
            # 8. seed (int)
            # 9. mode (int)
            edgecut, blocks = kahip.kaffpa(
                csr["vwgt"],      # already numpy array
                csr["xadj"],      # already numpy array
                csr["adjcwgt"],   # already numpy array
                csr["adjncy"],    # already numpy array
                m,                # nblocks (int)
                imbalance,        # imbalance (float)
                True,             # suppress_output (bool)
                seed,             # seed (int)
                mode              # mode (int)
            )
            print(f"✓ KaHIP completed: edgecut = {edgecut}")
            return np.array(blocks, dtype=np.int32)

    except TimeoutException:
        print(f" KaHIP timed out after {timeout} seconds")
        print("  Switching to fallback partitioner...")
        return fallback_partition(csr, m)
    
    except Exception as e:
        print(f" KaHIP crashed: {e}")
        print("  Switching to fallback partitioner...")
        return fallback_partition(csr, m)


def build_knn_graph(X, k=10):
    """
    Build k-NN graph.
    Returns indices: shape (n, k) where indices[i] = k nearest neighbors of X[i].
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices


def build_symmetric_graph(knn_indices):
    """
    Build symmetric weighted graph from k-NN indices.
    Mutual edges get weight 2, one-sided edges get weight 1.
    """
    n, k = knn_indices.shape
    adj = [{} for _ in range(n)]  # adjacency as dicts for merging

    for i in range(n):
        for j in knn_indices[i]:
            # one-sided edge i→j
            adj[i].setdefault(j, 1)

    # check mutual edges
    for i in range(n):
        for j in list(adj[i].keys()):
            if i in adj[j]:
                # mutual → upgrade both edges to weight 2
                adj[i][j] = 2
                adj[j][i] = 2

    return adj


def to_csr(adj):
    """
    Convert adjacency list to CSR format for KaHIP.
    
    Args:
        adj: List of dicts, where adj[i] = {neighbor: weight, ...}
    
    Returns:
        Dict with keys: 'xadj', 'adjncy', 'adjcwgt', 'vwgt'
    """
    n = len(adj)
    xadj = [0]
    adjncy = []
    adjcwgt = []

    for i in range(n):
        neighs = adj[i]
        adjncy.extend(neighs.keys())
        adjcwgt.extend(neighs.values())
        xadj.append(len(adjncy))

    vwgt = [1] * n  # uniform vertex weights

    return {
        "xadj": np.array(xadj, dtype=np.int64),
        "adjncy": np.array(adjncy, dtype=np.int32),
        "adjcwgt": np.array(adjcwgt, dtype=np.int32),
        "vwgt": np.array(vwgt, dtype=np.int32),
    }