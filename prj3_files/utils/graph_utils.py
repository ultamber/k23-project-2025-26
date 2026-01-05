import numpy as np
import subprocess
import tempfile
import os
from sklearn.neighbors import NearestNeighbors

def find_kahip():
    for exe in ["kaffpa", "kaffpaE"]:
        try:
            subprocess.run(
                [exe, "--help"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5
            )
            print(f"Found KaHIP binary: {exe}")
            return exe
        except Exception:
            continue
    print("KaHIP binaries not found — falling back to simple partitioner.")
    return None


KAHIP_BIN = find_kahip()

def fallback_partition(csr, m):
    """
    Simple round-robin partitioner when KaHIP is unavailable.
    """
    n = len(csr["vwgt"])
    labels = np.zeros(n, dtype=np.int32)

    # Round-robin assignment
    for i in range(n):
        labels[i] = i % m

    # Compute edgecut
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

    edgecut //= 2  # Each edge counted twice
    print(f"Fallback partitioner used. Edgecut: {edgecut:,}")
    return labels

def write_metis_graph(csr, path):
    n = len(csr["vwgt"])
    xadj = csr["xadj"]
    adjncy = csr["adjncy"]
    adjcwgt = csr["adjcwgt"]

    # Number of undirected edges
    n_edges = xadj[-1] // 2

    with open(path, "w") as f:
        # Header: n_vertices n_edges format_flag
        # format_flag=1 means edge-weighted graph
        f.write(f"{n} {n_edges} 1\n")

        # Each line: adjacency list for vertex i
        for i in range(n):
            start = xadj[i]
            end = xadj[i + 1]

            row = []
            for idx in range(start, end):
                j = adjncy[idx] + 1  # METIS uses 1-indexed vertices
                w = adjcwgt[idx]
                row.append(f"{j} {w}")

            f.write(" ".join(row) + "\n")

def run_kahip(csr, m=100, imbalance=0.03, mode=2, seed=1):
    if KAHIP_BIN is None:
        return fallback_partition(csr, m)

    mode_map = {
        0: "eco",
        1: "fast",
        2: "strong"
    }
    preconfig = mode_map.get(mode, "strong")

    print(f"Partitioning with '{preconfig}' configuration...")
    print(f"Parameters: m={m}, imbalance={imbalance:.2%}, seed={seed}")

    with tempfile.TemporaryDirectory() as tmp:
        graph_filename = "graph.graph"
        graph_path = os.path.join(tmp, graph_filename)
        write_metis_graph(csr, graph_path)

        possible_outputs = [
            os.path.join(tmp, f"tmppartition{m}"),
            os.path.join(tmp, f"{graph_filename}.part.k{m}"),
            os.path.join(tmp, f"graph.part.k{m}"),
        ]

        cmd = [
            KAHIP_BIN,
            graph_filename,
            f"--k={m}",
            f"--imbalance={imbalance}",
            f"--seed={seed}",
            f"--preconfiguration={preconfig}"
        ]

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmp,
                timeout=3600  # 1 hour timeout
            )

            edgecut = None
            balance = None
            time_spent = None

            for line in result.stdout.split('\n'):
                line_lower = line.lower()
                if 'cut' in line_lower and '=' not in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'cut' in part.lower() and i + 1 < len(parts):
                            try:
                                edgecut = int(parts[i + 1])
                            except ValueError:
                                pass
                elif 'balance' in line_lower:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'balance' in part.lower() and i + 1 < len(parts):
                            try:
                                balance = float(parts[i + 1])
                            except ValueError:
                                pass
                elif 'time spent' in line_lower:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.replace('.', '').isdigit():
                            try:
                                time_spent = float(part)
                            except ValueError:
                                pass

            out_path = None
            for possible_path in possible_outputs:
                if os.path.exists(possible_path):
                    out_path = possible_path
                    break

            if out_path is None:
                all_files = os.listdir(tmp)
                part_files = [f for f in all_files if 'partition' in f.lower() or '.part' in f]
                if part_files:
                    out_path = os.path.join(tmp, part_files[0])
                else:
                    raise FileNotFoundError("KaHIP did not produce partition file")

            with open(out_path, "r") as f:
                blocks = [int(line.strip()) for line in f if line.strip()]

            n_expected = len(csr["vwgt"])
            n_actual = len(blocks)

            if n_actual != n_expected:
                raise ValueError(
                    f"Partition count mismatch: expected {n_expected}, got {n_actual}"
                )

            print(f"Partitioning completed successfully")
            if edgecut is not None:
                print(f"  Edgecut: {edgecut:,}")
            if balance is not None:
                print(f"  Balance: {balance:.5f}")
            if time_spent is not None:
                print(f"  Time: {time_spent:.2f}s")

            return np.array(blocks, dtype=np.int32)

        except subprocess.TimeoutExpired:
            print(f"Timeout after 1 hour")
            return fallback_partition(csr, m)

        except subprocess.CalledProcessError as e:
            print(f"Failed with return code {e.returncode}")
            if e.returncode == -11:
                print(f"  Segmentation fault detected")
            return fallback_partition(csr, m)

        except Exception as e:
            print(f"Error: {e}")
            return fallback_partition(csr, m)

def build_knn_graph(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm='auto')
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    return indices


def build_symmetric_graph(knn_indices):
    n, k = knn_indices.shape

    invalid_mask = (knn_indices < 0) | (knn_indices >= n)
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        print(f"ERROR: Found {n_invalid} invalid indices!")
        print(f"Invalid indices: {knn_indices[invalid_mask][:10]}...")  # Show first 10
        raise ValueError("k-NN graph contains invalid indices")

    self_loops = []
    for i in range(min(n, 100)):
        if i in knn_indices[i]:
            self_loops.append(i)

    if self_loops:
        print(f"WARNING: Found {len(self_loops)} self-loops in first 100 nodes")
        print(f"Self-loop nodes: {self_loops[:10]}...")

    print(f"k-NN graph validation passed")

    knn_set = [set(knn_indices[i]) for i in range(n)]

    for i in range(n):
        knn_set[i].discard(i)

    adj = [dict() for _ in range(n)]

    reciprocal_count = 0
    non_reciprocal_count = 0

    # Add edges with weights
    for i in range(n):
        for j in knn_indices[i]:
            j = int(j)

            # Skip self-loops
            if j == i:
                continue

            # Skip if already processed
            if j in adj[i]:
                continue

            # Check if reciprocal
            is_reciprocal = i in knn_set[j]
            weight = 2 if is_reciprocal else 1

            # Add bidirectional edge
            adj[i][j] = weight
            adj[j][i] = weight

            if is_reciprocal:
                reciprocal_count += 1
            else:
                non_reciprocal_count += 1

    print(f"Edge weights assigned:")
    print(f"Reciprocal edges: {reciprocal_count:,} (weight=2)")
    print(f"Non-reciprocal edges: {non_reciprocal_count:,} (weight=1)")
    print(f"Total edges: {reciprocal_count + non_reciprocal_count:,}")

    total_edges = sum(len(neighbors) for neighbors in adj)
    print(f"Total edge count: {total_edges:,}")

    if total_edges == 0:
        raise ValueError("Graph has no edges - Cannot partition empty graph.")

    return adj

def to_csr(adj):
    n = len(adj)
    xadj = [0]
    adjncy = []
    adjcwgt = []

    for i in range(n):
        neighbors = adj[i]
        for j, w in neighbors.items():
            adjncy.append(j)
            adjcwgt.append(w)
        xadj.append(len(adjncy))

    return {
        "xadj": np.array(xadj, dtype=np.int64),
        "adjncy": np.array(adjncy, dtype=np.int32),
        "adjcwgt": np.array(adjcwgt, dtype=np.int32),
        "vwgt": np.ones(n, dtype=np.int32)
    }
