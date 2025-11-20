# graph_utils.py
from sklearn.neighbors import NearestNeighbors
import numpy as np
try:
    import kahip
    kaffpa = kahip.kaffpa
except Exception:
    import warnings
    import random
    warnings.warn(
        "kahip module not found; using fallback partitioner. "
        "Install kahip for better partitions (e.g. pip install kahip).",
        UserWarning,
    )
# -------------------------------------------------------
# Try import KaHIP – if it fails, fallback is enabled
# -------------------------------------------------------
try:
    import kahip
    KAHIP_AVAILABLE = True
except Exception:
    KAHIP_AVAILABLE = False
    print("[graph_utils] KaHIP not available — using fallback partitioner.")


# -------------------------------------------------------
# Fallback partitioner
# -------------------------------------------------------
def fallback_partition(csr, m):
    n = len(csr["vwgt"])
    labels = np.zeros(n, dtype=np.int32)
    for i in range(n):
        labels[i] = i % m
    print(f"[graph_utils] Fallback partitioner used. m = {m}")
    return labels


# -------------------------------------------------------
# Wrapper that tries KaHIP and falls back on error
# -------------------------------------------------------
def run_kahip(csr, m=100, imbalance=0.03, mode=2, seed=1):

    if not KAHIP_AVAILABLE:
        return fallback_partition(csr, m)

    print("Running KaHIP partitioner…")

    try:
        edgecut, blocks = kahip.kaffpa(
            np.array(csr["vwgt"], dtype=np.int32),
            np.array(csr["xadj"], dtype=np.int64),
            np.array(csr["adjcwgt"], dtype=np.int32),
            np.array(csr["adjncy"], dtype=np.int32),
            nblocks=m,
            imbalance=imbalance,
            suppress_output=True,
            seed=seed,
            mode=mode,
        )
        print(f"KaHIP edgecut = {edgecut}")
        return np.array(blocks, dtype=np.int32)

    except Exception as e:
        print("⚠ KaHIP crashed, switching to fallback.")
        print(e)
        return fallback_partition(csr, m)

def kaffpa(vwgt, xadj, adjcwgt, adjncy, nblocks=100, imbalance=0.03, suppress_output=True, seed=1, mode=2):
    """
    Lightweight fallback partitioner: deterministic round-robin assignment
    with optional seed-based randomness; returns (edgecut, blocks).
    This is a simple substitute so the module runs without the kahip package.
    """
    random.seed(seed)
    n = len(vwgt)
    # simple round-robin assignment
    blocks = [i % nblocks for i in range(n)]
    # compute edgecut (sum of crossing edge weights, divided by 2 for undirected)
    edgecut = 0
    for i in range(n):
        start = xadj[i]
        end = xadj[i + 1]
        bi = blocks[i]
        for idx in range(start, end):
            j = adjncy[idx]
            if blocks[j] != bi:
                w = adjcwgt[idx] if adjcwgt else 1
                edgecut += w
    edgecut = edgecut // 2
    return int(edgecut), blocks


def build_knn_graph(X, k=10):
    """
    Κατασκευή k-NN γράφου.
    Επιστρέφει indices: shape (n, k)
    όπου indices[i] = τα k κοντινότερα neighbors του X[i].
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    return indices

def build_symmetric_graph(knn_indices):
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
    n = len(adj)
    xadj = [0]
    adjncy = []
    adjcwgt = []

    for i in range(n):
        neighs = adj[i]
        adjncy.extend(neighs.keys())
        adjcwgt.extend(neighs.values())
        xadj.append(len(adjncy))

    vwgt = [1] * n    # <--- ΦΤΙΑΞΕ ΤΟ: πρέπει να είναι λίστα, όχι None

    return {
        "xadj": np.array(xadj, dtype=np.int64),
        "adjncy": np.array(adjncy, dtype=np.int32),
        "adjcwgt": np.array(adjcwgt, dtype=np.int32),
        "vwgt": np.array(vwgt, dtype=np.int32),
    }

# 1. Build a k−NN graph from data (e.g. sklearn NearestNeighbors)
# 2. Convert to (CSR) arrays: xadj, adjncy, adjcwgt, vwgt
# 3. Call KaHIP
# 4. Use 'blocks' as partition labels
# Outputs: edgecut = total cut weight, blocks = list of partition labels.
# Format: CSR (Compressed Sparse Row) convention: undirected, integer weights.
# Larger imbalance allows cheaper cuts but unequal partitions.
# Higher mode gives better accuracy at higher runtime.
def run_kahip(csr, m=100, imbalance=0.03, mode=2, seed=1):
    print("Running KaHIP partitioner…")
    print("CSR xadj len:", len(csr["xadj"]), "dtype:", csr["xadj"].dtype)
    print("CSR adjncy len:", len(csr["adjncy"]), "dtype:", csr["adjncy"].dtype)
    print("CSR adjcwgt len:", len(csr["adjcwgt"]), "dtype:", csr["adjcwgt"].dtype)
    print("CSR vwgt:", csr["vwgt"])
    print("m =", m, "imbalance =", imbalance, "mode =", mode)

    # KaHIP requires standard CSR ordering:
    # xadj, adjncy, vwgt, adjwgt, num_partitions, imbalance, suppress_output, seed, mode
    edgecut, blocks = kahip.kaffpa(
        csr["xadj"],         # index pointer array
        csr["adjncy"],       # adjacency list
        csr.get("vwgt", None),
        csr.get("adjcwgt", None),
        m,                   # number of partitions
        imbalance,           # imbalance %
        True,                # suppress KaHIP output
        seed,                # random seed
        mode                 # fast / eco / strong
    )

    print(f"KaHIP edgecut = {edgecut}")
    return np.array(blocks, dtype=np.int32)
