import subprocess
import tempfile
import numpy as np
from pathlib import Path

def compute_knn_lsh(X, k,calculated_output="", search_path="../bin/search", dtype="mnist", 
                    dataset_path="../datasets/MNIST/train-images.idx3-ubyte",
                    query_path="../datasets/MNIST/t10k-images.idx3-ubyte"):
    """
    Build k-NN graph using LSH from Project 1.
    
    Runs the LSH binary to find k nearest neighbors for each point in X.
    
    Command format:
        ./bin/search -d dataset -q query -o output -type mnist -lsh -L 15 -k 4 -w 2 -N k -range false
    
    Args:
        X: Data matrix (used for determining n, but actual files are read from dataset_path/query_path)
        k: Number of nearest neighbors
        search_path: Path to LSH binary
        dtype: Data type ('mnist' or 'sift')
        dataset_path: Path to existing dataset file (-d parameter)
        query_path: Path to existing query file (-q parameter)
    """
    n, d = X.shape
    
    print(f"\n{'='*60}")
    print(f"Computing k-NN graph using LSH from Project 1")
    print(f"{'='*60}")
    print(f"Dataset: n={n}, d={d}")
    print(f"k={k}")
    print(f"Type: {dtype}")
    print(f"LSH executable: {search_path}")
    print(f"Dataset file: {dataset_path}")
    print(f"Query file: {query_path}")
    
    # Verify files exist
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    if not Path(query_path).exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    
    # Create temporary directory only for output file
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_file = tmpdir / "lsh_output.txt"
        
        # Build command for LSH search
        # ./bin/search -d dataset -q query -o output -type mnist -lsh -L 15 -k 4 -w 2 -N k -range false
        cmd = [
            str(search_path),
            "-d", str(dataset_path),
            "-q", str(query_path),
            "-o", str(output_file),
            "-type", dtype,
            "-lsh",                # Enable LSH mode 
            "-L", "10",            # Number of hash tables
            "-k", "4",             # Hash functions per table
            "-w", "2.0" if dtype == "sift" else "4.0",  # Bucket width (larger for SIFT)
            "-N", str(k),          # Number of nearest neighbors
            "-range", "false"      # Disable range search
        ]
        
        print(f"\nRunning LSH search...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            # Skip subprocess if pre-calculated output file is provided
            if calculated_output and Path(calculated_output).exists():
                print(f"Using pre-calculated output file: {calculated_output}")
                output_file = Path(calculated_output)
            else:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=1800,  # 30 minute timeout
                    check=True
                )
            
            print(f"Parsing LSH output...")
            # Note: n here should be the number of queries (from query_path)
            knn_graph = parse_lsh_output(output_file, n, k)
            
            # Validate graph
            if knn_graph.shape[0] > 0 and knn_graph.shape[1] != k:
                raise RuntimeError(f"Invalid k-NN graph shape: {knn_graph.shape}, expected (?, {k})")
            
            print(f"✓ k-NN graph built successfully")
            return knn_graph
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"LSH search timed out after 30 minutes")
        except subprocess.CalledProcessError as e:
            error_msg = f"LSH search failed.\nCommand: {' '.join(cmd)}\nStderr: {e.stderr}\nStdout: {e.stdout}"
            raise RuntimeError(error_msg)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"LSH executable not found at {search_path}. "
                f"Please build Project 1 first or use --use_exact_knn flag."
            )


def parse_lsh_output(output_file, n, k):
    """
    Parse LSH output file and extract k-NN graph.
    
    Expected output format from LSH binary:
        Query: <query_id>
        Nearest neighbor-1: <neighbor_id>
        distanceApproximate: <distance>
        Nearest neighbor-2: <neighbor_id>
        distanceApproximate: <distance>
        ...
    
    Args:
        output_file: Path to LSH output file
        n: Number of queries (same as dataset size)
        k: Number of nearest neighbors
    
    Returns:
        knn_graph: (n, k) array where knn_graph[i] contains indices of k-nearest neighbors of point i
    """
    knn_graph = np.full((n, k), -1, dtype=np.int32)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    current_query = -1
    neighbor_count = 0
    
    for line in lines:
        line = line.strip()
        
        # Parse query ID: "Query: 123"
        if line.startswith("Query:"):
            try:
                current_query = int(line.split(":")[1].strip())
                neighbor_count = 0
            except (IndexError, ValueError):
                continue
        
        # Parse nearest neighbor: "Nearest neighbor-1: 456"
        elif line.startswith("Nearest neighbor-"):
            if current_query >= 0 and current_query < n and neighbor_count < k:
                try:
                    # Extract neighbor ID after the colon
                    neighbor_id = int(line.split(":")[1].strip())
                    
                    # Skip self-neighbors (when query finds itself)
                    if neighbor_id == current_query:
                        continue
                    
                    knn_graph[current_query, neighbor_count] = neighbor_id
                    neighbor_count += 1
                except (IndexError, ValueError):
                    continue
    
    # Check for incomplete results
    incomplete = np.sum(knn_graph == -1)
    if incomplete > 0:
        print(f"⚠️  Warning: {incomplete}/{n*k} neighbors not found by LSH")
        print(f"   This may happen if LSH couldn't find enough neighbors for some queries")
        print(f"   Consider using --use_exact_knn for more reliable k-NN graph construction")
    
    return knn_graph

def compute_knn_sklearn(X, k):
    """
    Fallback: compute exact k-NN using sklearn (faster and more reliable for graph building).
    
    This is the recommended method for building k-NN graphs in Neural LSH.
    """
    from sklearn.neighbors import NearestNeighbors
    
    print(f"\nComputing exact k-NN graph using sklearn...")
    print(f"Dataset: n={X.shape[0]}, d={X.shape[1]}, k={k}")
    
    nbrs = NearestNeighbors(
        n_neighbors=k+1,  # +1 because point itself is included
        algorithm='auto',
        metric='euclidean',
        n_jobs=-1
    ).fit(X)
    
    distances, indices = nbrs.kneighbors(X)
    
    # Remove self (first neighbor)
    knn_graph = indices[:, 1:k+1]
    
    print(f"✓ Exact k-NN graph computed")
    return knn_graph