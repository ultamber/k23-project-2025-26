import subprocess
import tempfile
import numpy as np
from pathlib import Path

def compute_knn_from_project1(X, k, calculated_output="", search_path="../bin/search", dtype="mnist", 
                    dataset_path="../datasets/MNIST/train-images.idx3-ubyte",
                    query_path="../datasets/MNIST/train-images.idx3-ubyte",
                    method="ivfflat"):
    """
    Build k-NN graph using Project 1 search methods.
    """
    n, d = X.shape
    
    print(f"\n{'='*60}")
    print(f"Computing k-NN graph using {method.upper()} from Project 1")
    print(f"{'='*60}")
    print(f"Dataset: n={n}, d={d}")
    print(f"k={k}")
    print(f"Type: {dtype}")
    
    # Verify files exist
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not Path(query_path).exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "search_output.txt"
        
        # Build command based on method
        if method == "ivfflat":
            cmd = [
                str(search_path),
                "-d", str(dataset_path),
                "-q", str(query_path),
                "-o", str(output_file),
                "-type", dtype,
                "-ivfflat",
                "-N", str(k),
                "-range", "false",
                "-kclusters", "128",  # Number of clusters
                "-nprobe", "32"      # Number of probes
            ]
        elif method == "lsh":
            cmd = [
                str(search_path),
                "-d", str(dataset_path),
                "-q", str(query_path),
                "-o", str(output_file),
                "-type", dtype,
                "-lsh",
                "-L", "10",
                "-k", "4",
                "-w", "2.0" if dtype == "sift" else "4.0",
                "-N", str(k),
                "-range", "false"
            ]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"\nRunning {method.upper()} search...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Skip subprocess if pre-calculated output exists
            if calculated_output and Path(calculated_output).exists():
                print(f"Using pre-calculated output file: {calculated_output}")
                output_file = Path(calculated_output)
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10000,
                    check=True
                )
            
            print(f"Parsing output...")
            knn_graph = parse_search_output(output_file, n, k)
            
            print(f"\nValidating k-NN graph...")
            knn_graph = validate_and_fix_knn_graph(knn_graph, n, k)
            
            return knn_graph
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{method.upper()} search timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{method.upper()} search failed: {e.stderr}")
        
def validate_and_fix_knn_graph(knn_graph, n, k):
    """
    Validate and fix k-NN graph to prevent KaHIP segfaults.
    
    Fixes:
    1. Invalid indices (-1, out of bounds)
    2. Self-loops
    3. Insufficient neighbors
    """
    print(f"  Original graph shape: {knn_graph.shape}")
    
    # Count issues
    invalid_count = 0
    self_loop_count = 0
    fixed_rows = []
    
    for i in range(n):
        neighbors = knn_graph[i].copy()
        
        # Remove invalid indices
        valid_mask = (neighbors >= 0) & (neighbors < n)
        neighbors = neighbors[valid_mask]
        
        if len(neighbors) < len(knn_graph[i]):
            invalid_count += len(knn_graph[i]) - len(neighbors)
        
        # Remove self-loops
        neighbors = neighbors[neighbors != i]
        if len(neighbors) < valid_mask.sum():
            self_loop_count += 1
        
        # If not enough neighbors, find more using brute force
        if len(neighbors) < k:
            # This shouldn't happen often, but handle it gracefully
            neighbors = find_fallback_neighbors(i, n, k, existing=neighbors)
        
        # Pad to k neighbors if necessary
        if len(neighbors) < k:
            # Duplicate last neighbor if needed (better than invalid index)
            neighbors = np.pad(neighbors, (0, k - len(neighbors)), 
                             mode='edge')
        
        # Take only k neighbors
        neighbors = neighbors[:k]
        
        fixed_rows.append(neighbors)
    
    knn_graph_fixed = np.array(fixed_rows, dtype=np.int32)
    
    print(f"  Fixed graph shape: {knn_graph_fixed.shape}")
    print(f"  Invalid indices removed: {invalid_count}")
    print(f"  Self-loops removed: {self_loop_count}")
    
    # Final validation
    assert knn_graph_fixed.shape == (n, k), f"Shape mismatch: {knn_graph_fixed.shape} != ({n}, {k})"
    assert np.all(knn_graph_fixed >= 0), "Negative indices found!"
    assert np.all(knn_graph_fixed < n), "Out-of-bounds indices found!"
    
    # Check for self-loops
    for i in range(n):
        if i in knn_graph_fixed[i]:
            print(f"  WARNING: Self-loop found at node {i}")
    
    print(f"✓ Graph validation passed")
    
    return knn_graph_fixed


def find_fallback_neighbors(i, n, k, existing):
    """
    Find additional neighbors using simple distance heuristic.
    Just returns closest indices that aren't already neighbors.
    """
    existing_set = set(existing)
    existing_set.add(i)  # Don't include self
    
    # Simple heuristic: use nearby indices
    candidates = []
    for offset in range(1, n):
        if len(candidates) >= k:
            break
        for sign in [1, -1]:
            candidate = (i + sign * offset) % n
            if candidate not in existing_set and candidate != i:
                candidates.append(candidate)
                if len(candidates) >= k:
                    break
    
    # Combine existing and new
    all_neighbors = list(existing) + candidates
    return np.array(all_neighbors[:k], dtype=np.int32)

# def parse_lsh_output(output_file, n, k):
#     """
#     Parse LSH output file and extract k-NN graph.
    
#     Expected output format from LSH binary:
#         Query: <query_id>
#         Nearest neighbor-1: <neighbor_id>
#         distanceApproximate: <distance>
#         Nearest neighbor-2: <neighbor_id>
#         distanceApproximate: <distance>
#         ...
    
#     Args:
#         output_file: Path to LSH output file
#         n: Number of queries (same as dataset size)
#         k: Number of nearest neighbors
    
#     Returns:
#         knn_graph: (n, k) array where knn_graph[i] contains indices of k-nearest neighbors of point i
#     """
#     knn_graph = np.full((n, k), -1, dtype=np.int32)
    
#     with open(output_file, 'r') as f:
#         lines = f.readlines()
    
#     current_query = -1
#     neighbor_count = 0
    
#     for line in lines:
#         line = line.strip()
        
#         # Parse query ID: "Query: 123"
#         if line.startswith("Query:"):
#             try:
#                 current_query = int(line.split(":")[1].strip())
#                 neighbor_count = 0
#             except (IndexError, ValueError):
#                 continue
        
#         # Parse nearest neighbor: "Nearest neighbor-1: 456"
#         elif line.startswith("Nearest neighbor-"):
#             if current_query >= 0 and current_query < n and neighbor_count < k:
#                 try:
#                     # Extract neighbor ID after the colon
#                     neighbor_id = int(line.split(":")[1].strip())
                    
#                     # Skip self-neighbors (when query finds itself)
#                     if neighbor_id == current_query:
#                         continue
                    
#                     knn_graph[current_query, neighbor_count] = neighbor_id
#                     neighbor_count += 1
#                 except (IndexError, ValueError):
#                     continue
    
#     # Check for incomplete results
#     incomplete = np.sum(knn_graph == -1)
#     if incomplete > 0:
#         print(f"  Warning: {incomplete}/{n*k} neighbors not found by LSH")
#         print(f"   This may happen if LSH couldn't find enough neighbors for some queries")
#         print(f"   Consider using --use_exact_knn for more reliable k-NN graph construction")
    
#     return knn_graph
def parse_search_output(output_file, n, k):
    """
    Parse search output from Project 1 (LSH/IVFFlat/Hypercube).
    
    Returns:
        knn_graph: (n, k) array of neighbor indices
    """
    knn_graph = np.full((n, k), -1, dtype=np.int32)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    current_query = -1
    neighbor_count = 0
    
    for line in lines:
        line = line.strip()
        
        # Parse query ID
        if line.startswith("Query:"):
            try:
                current_query = int(line.split(":")[1].strip())
                neighbor_count = 0
            except (IndexError, ValueError):
                continue
        
        # Parse nearest neighbor
        elif line.startswith("Nearest neighbor"):
            if current_query >= 0 and current_query < n and neighbor_count < k:
                try:
                    # Different formats possible:
                    # "Nearest neighbor-1: 456"
                    # "Nearest neighbor-1: id=456, dist=1.23"
                    
                    parts = line.split(":")
                    if len(parts) < 2:
                        continue
                    
                    neighbor_info = parts[1].strip()
                    
                    # Handle "id=456, dist=1.23" format
                    if "id=" in neighbor_info:
                        neighbor_id = int(neighbor_info.split("id=")[1].split(",")[0].strip())
                    else:
                        # Handle simple "456" format
                        neighbor_id = int(neighbor_info.split()[0])
                    
                    # Skip self-neighbors
                    if neighbor_id == current_query:
                        continue
                    
                    # Skip invalid indices
                    if neighbor_id < 0 or neighbor_id >= n:
                        print(f"  WARNING: Invalid neighbor {neighbor_id} for query {current_query}")
                        continue
                    
                    knn_graph[current_query, neighbor_count] = neighbor_id
                    neighbor_count += 1
                    
                except (IndexError, ValueError) as e:
                    print(f"  WARNING: Failed to parse line: {line} ({e})")
                    continue
    
    # Check for incomplete results
    incomplete = np.sum(knn_graph == -1)
    if incomplete > 0:
        print(f"  WARNING: {incomplete}/{n*k} neighbors not found")
    
    return knn_graph
