# # lsh_knn.py
# """
# k-NN graph construction using LSH from Project 1.

# This module interfaces with the C++ LSH search executable
# to compute approximate k-nearest neighbors for building
# the initial graph for Neural LSH.
# """

# import numpy as np
# import subprocess
# import struct
# import tempfile
# import os
# from pathlib import Path


# def write_mnist_idx(path, X):
#     """
#     Write data in MNIST idx3-ubyte format.
    
#     Args:
#         path: output file path
#         X: numpy array (n, 784) with values in [0, 255]
#     """
#     if X.ndim == 2 and X.shape[1] == 784:
#         n = X.shape[0]
#         magic = 2051  # 0x00000803
        
#         with open(path, "wb") as f:
#             f.write(struct.pack(">I", magic))
#             f.write(struct.pack(">I", n))
#             f.write(struct.pack(">I", 28))  # rows
#             f.write(struct.pack(">I", 28))  # cols
            
#             # Write image data
#             img_data = X.reshape(n, 28, 28).astype(np.uint8)
#             f.write(img_data.tobytes())
#     else:
#         raise ValueError(f"Invalid shape for MNIST: {X.shape}, expected (n, 784)")


# def write_sift_fvecs(path, X):
#     """
#     Write data in SIFT .fvecs format.
    
#     Args:
#         path: output file path
#         X: numpy array (n, 128) with float32 values
#     """
#     if X.ndim == 2 and X.shape[1] == 128:
#         n, d = X.shape
        
#         with open(path, "wb") as f:
#             for i in range(n):
#                 # Write dimension as little-endian int32
#                 f.write(struct.pack("<i", d))
#                 # Write vector as little-endian float32
#                 f.write(X[i].astype("<f4").tobytes())
#     else:
#         raise ValueError(f"Invalid shape for SIFT: {X.shape}, expected (n, 128)")


# def parse_knn_from_output(output_path, k, n_queries):
#     """
#     Parse k-NN results from LSH search output file.
    
#     Expected format:
#         Query: <id>
#         Nearest neighbor-1: <idx>
#         ...
#         Nearest neighbor-k: <idx>
    
#     Args:
#         output_path: path to output file
#         k: number of neighbors
#         n_queries: number of queries
        
#     Returns:
#         knn: numpy array (n_queries, k) with neighbor indices
#     """
#     knn = np.zeros((n_queries, k), dtype=np.int32)
    
#     with open(output_path, "r") as f:
#         current_query = -1
#         neighbor_count = 0
        
#         for line in f:
#             line = line.strip()
            
#             if line.startswith("Query:"):
#                 # New query
#                 current_query = int(line.split(":")[1].strip())
#                 neighbor_count = 0
                
#             elif line.startswith("Nearest neighbor-"):
#                 # Parse neighbor index
#                 parts = line.split(":")
#                 if len(parts) >= 2:
#                     neighbor_idx = int(parts[1].strip())
                    
#                     if current_query >= 0 and neighbor_count < k:
#                         knn[current_query, neighbor_count] = neighbor_idx
#                         neighbor_count += 1
    
#     return knn


# def compute_knn_lsh(X, k, search_path="./bin/search", dtype="mnist"):
#     """
#     Compute approximate k-NN graph using Project 1 LSH executable.
    
#     Strategy:
#     - Write dataset to temporary file
#     - For each point, use it as query to find its k nearest neighbors
#     - Collect all results into k-NN graph
    
#     Args:
#         X: numpy array (n, d) - dataset
#         k: number of nearest neighbors
#         search_path: path to LSH search executable
#         dtype: 'mnist' or 'sift'
        
#     Returns:
#         knn_graph: numpy array (n, k) with neighbor indices
#     """
#     n, d = X.shape
    
#     print(f"\n{'='*60}")
#     print(f"Computing k-NN graph using LSH from Project 1")
#     print(f"{'='*60}")
#     print(f"Dataset: n={n}, d={d}")
#     print(f"k={k}")
#     print(f"Type: {dtype}")
#     print(f"LSH executable: {search_path}")
    
#     # Validate inputs
#     if dtype == "mnist":
#         assert d == 784, f"MNIST requires d=784, got {d}"
#         # Ensure values are in [0, 255]
#         if X.max() <= 1.0:
#             X = (X * 255.0).clip(0, 255)
#         X = X.astype(np.uint8)
        
#     elif dtype == "sift":
#         assert d == 128, f"SIFT requires d=128, got {d}"
#         X = X.astype(np.float32)
#     else:
#         raise ValueError(f"Unsupported dtype: {dtype}")
    
#     # Check if search executable exists
#     search_path = Path(search_path)
#     if not search_path.exists():
#         raise FileNotFoundError(
#             f"LSH search executable not found at {search_path}\n"
#             f"Please ensure Project 1 binary is compiled and path is correct"
#         )
    
#     knn_graph = np.zeros((n, k), dtype=np.int32)
    
#     # Use temporary directory for intermediate files
#     with tempfile.TemporaryDirectory() as tmpdir:
#         tmpdir = Path(tmpdir)
        
#         # Write entire dataset once
#         if dtype == "mnist":
#             dataset_file = tmpdir / "dataset.idx3-ubyte"
#             write_mnist_idx(dataset_file, X)
#         else:  # sift
#             dataset_file = tmpdir / "dataset.fvecs"
#             write_sift_fvecs(dataset_file, X)
        
#         query_file = tmpdir / "query_temp"
#         output_file = tmpdir / "lsh_output.txt"
        
#         print(f"\nProcessing queries:")
#         print_interval = max(1, n // 10)  # Print progress every 10%
        
#         for i in range(n):
#             if i % print_interval == 0:
#                 print(f"  Progress: {i}/{n} ({100*i/n:.1f}%)")
            
#             # Write single query
#             if dtype == "mnist":
#                 write_mnist_idx(query_file, X[i:i+1])
#             else:
#                 write_sift_fvecs(query_file, X[i:i+1])
            
#             # Build command for LSH search
#             cmd = [
#                 str(search_path),
#                 "-d", str(dataset_file),
#                 "-q", str(query_file),
#                 "-o", str(output_file),
#                 "-type", dtype,
#                 "-lsh",  # Use LSH method
#                 "-N", str(k),
#                 "-range", "false"  # Disable range search for speed
#             ]
            
#             try:
#                 # Run LSH search
#                 result = subprocess.run(
#                     cmd,
#                     check=True,
#                     capture_output=True,
#                     text=True,
#                     timeout=30  # 30 second timeout per query
#                 )
                
#                 # Parse output
#                 neighbors = parse_knn_from_output(output_file, k, 1)[0]
                
#                 # Fill in any missing neighbors with self-reference
#                 if len(neighbors) < k:
#                     neighbors = list(neighbors) + [i] * (k - len(neighbors))
                
#                 knn_graph[i] = neighbors[:k]
                
#             except subprocess.TimeoutExpired:
#                 print(f"\n⚠️  Timeout for query {i}, using self as neighbors")
#                 knn_graph[i] = [i] * k
                
#             except subprocess.CalledProcessError as e:
#                 print(f"\n⚠️  LSH search failed for query {i}: {e}")
#                 print(f"stderr: {e.stderr}")
#                 # Fallback: use self as neighbors
#                 knn_graph[i] = [i] * k
                
#             except Exception as e:
#                 print(f"\n⚠️  Unexpected error for query {i}: {e}")
#                 knn_graph[i] = [i] * k
        
#         print(f"  Progress: {n}/{n} (100.0%)")
    
#     print(f"\n✓ k-NN graph construction completed")
#     print(f"  Shape: {knn_graph.shape}")
#     print(f"  Self-loops: {np.sum(knn_graph == np.arange(n)[:, None])}")
#     print(f"{'='*60}\n")
    
#     return knn_graph


# def compute_knn_sklearn(X, k):
#     """
#     Fallback: compute exact k-NN using sklearn (for comparison/debugging).
    
#     Args:
#         X: numpy array (n, d)
#         k: number of neighbors
        
#     Returns:
#         knn_graph: numpy array (n, k)
#     """
#     from sklearn.neighbors import NearestNeighbors
    
#     print(f"Computing exact k-NN with sklearn (k={k})...")
#     nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
#     nn.fit(X)
#     _, indices = nn.kneighbors(X)
    
#     return indices.astype(np.int32)
import subprocess
import tempfile
import numpy as np
from pathlib import Path

def compute_knn_lsh(X, k, search_path="../bin/search", dtype="mnist"):
    """
    Build k-NN graph using LSH from Project 1 (optimized batch mode).
    """
    n, d = X.shape
    
    print(f"\n{'='*60}")
    print(f"Computing k-NN graph using LSH from Project 1")
    print(f"{'='*60}")
    print(f"Dataset: n={n}, d={d}")
    print(f"k={k}")
    print(f"Type: {dtype}")
    print(f"LSH executable: {search_path}")
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Write dataset file (this will be both input and query)
        if dtype == "mnist":
            data_file = tmpdir / "data.idx3-ubyte"
            write_mnist_format(X, data_file)
            query_file = data_file  # Use same file as query
        else:  # sift
            data_file = tmpdir / "data.fvecs"
            write_fvecs(X, data_file)
            query_file = data_file
        
        output_file = tmpdir / "lsh_output.txt"
        
        # Run LSH with optimized parameters for graph building
        # Use fewer hash tables (L) and probes to speed up
        cmd = [
            str(search_path),
            "-d", str(data_file),
            "-q", str(query_file),
            "-o", str(output_file),
            "-type", dtype,
            "-N", str(k),
            "-L", "5",      # Reduced from default 10
            "-k", "4",      # Hash function count
            "-w", "4.0",    # Bucket width
            "-range", "false"
        ]
        
        print(f"\nRunning LSH search...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
                check=True
            )
            
            # Parse output to build k-NN graph
            print(f"Parsing results...")
            knn_graph = parse_lsh_output(output_file, n, k)
            
            print(f"✓ k-NN graph built successfully")
            return knn_graph
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"LSH search timed out after 30 minutes")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"LSH search failed: {e.stderr}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"LSH executable not found at {search_path}. "
                f"Please build it first or use --use_exact_knn flag."
            )


def parse_lsh_output(output_file, n, k):
    """
    Parse LSH output file and extract k-NN graph.
    
    Returns:
        knn_graph: (n, k) array where knn_graph[i] contains indices of k-nearest neighbors of point i
    """
    knn_graph = np.zeros((n, k), dtype=np.int32)
    
    with open(output_file, 'r') as f:
        lines = f.readlines()
    
    current_query = -1
    neighbor_count = 0
    
    for line in lines:
        line = line.strip()
        
        # Parse query ID
        if line.startswith("Query:"):
            current_query = int(line.split(":")[1].strip())
            neighbor_count = 0
            continue
        
        # Parse nearest neighbor
        if line.startswith("Nearest neighbor-"):
            if current_query >= 0 and neighbor_count < k:
                neighbor_id = int(line.split(":")[1].strip())
                knn_graph[current_query, neighbor_count] = neighbor_id
                neighbor_count += 1
    
    return knn_graph


def write_mnist_format(X, filename):
    """Write MNIST idx3-ubyte format."""
    n, d = X.shape
    # Assume 28x28 images
    h, w = 28, 28
    assert d == h * w, f"Expected d={h*w} for MNIST, got {d}"
    
    with open(filename, 'wb') as f:
        # Magic number for images: 2051
        f.write(np.array([0, 0, 8, 3], dtype=np.uint8).tobytes())
        # Number of images
        f.write(np.array([n], dtype=np.int32).byteswap().tobytes())
        # Rows
        f.write(np.array([h], dtype=np.int32).byteswap().tobytes())
        # Columns
        f.write(np.array([w], dtype=np.int32).byteswap().tobytes())
        # Pixel data (scale back to 0-255 if normalized)
        if X.max() <= 1.0:
            data = (X * 255).astype(np.uint8)
        else:
            data = X.astype(np.uint8)
        f.write(data.tobytes())


def write_fvecs(X, filename):
    """Write SIFT .fvecs format."""
    n, d = X.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            # Write dimension
            f.write(np.array([d], dtype=np.int32).tobytes())
            # Write vector
            f.write(X[i].astype(np.float32).tobytes())


def compute_knn_sklearn(X, k):
    """
    Fallback: compute exact k-NN using sklearn (faster for graph building).
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