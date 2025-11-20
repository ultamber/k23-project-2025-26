# lsh_knn.py
import numpy as np
import subprocess
import struct
import tempfile
import os


# ========================================
#   WRITE MNIST IN idx3-ubyte FORMAT
# ========================================
def write_mnist_idx(path, X):
    """
    Write MNIST images in official idx3-ubyte format.
    X must be shape (n, 784), values 0-255 uint8.
    """
    n = X.shape[0]
    magic = 2051  # 0x00000803

    with open(path, "wb") as f:
        f.write(struct.pack(">I", magic))
        f.write(struct.pack(">I", n))
        f.write(struct.pack(">I", 28))
        f.write(struct.pack(">I", 28))
        f.write(X.reshape(n, 28, 28).astype(np.uint8).tobytes())


# ========================================
#   PARSE SEARCH OUTPUT
# ========================================
def parse_knn_from_output(path, k):
    """
    Reads the neighbors from the search program output.
    Returns a list of lists: [[n1, n2, ..., nk]].
    """
    neighbors = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Nearest neighbor-"):
                nn = int(line.split(":")[1].strip())
                neighbors.append(nn)
                if len(neighbors) == k:
                    return [neighbors]
    return [[]]


# ========================================
#   MAIN FUNCTION: compute_knn_lsh
# ========================================
def compute_knn_lsh(X, k, search_path="./bin/search", dtype="mnist"):
    """
    Computes approximate k-NN using the C++ search binary.
    
    X: numpy array (n, 784) scaled 0–1 or 0–255
    k: number of neighbors
    dtype: only 'mnist' supported for now
    """
    assert dtype == "mnist", "Only MNIST supported in this version"

    n = X.shape[0]
    knn = np.zeros((n, k), dtype=np.int32)

    # Normalize to uint8 0–255 as expected by search binary
    X_uint8 = (X * 255.0).clip(0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_file = f"{tmpdir}/input.idx3-ubyte"
        query_file = f"{tmpdir}/query.idx3-ubyte"
        out_file = f"{tmpdir}/out.txt"

        # Write entire dataset once
        write_mnist_idx(dataset_file, X_uint8)

        for i in range(n):
            # Write single query
            write_mnist_idx(query_file, X_uint8[i:i+1])

            cmd = [
                search_path,
                "-d", dataset_file,
                "-q", query_file,
                "-o", out_file,
                "-type", "mnist",
                "-lsh",
                "-N", str(k)
            ]

            subprocess.run(cmd, check=True)

            result = parse_knn_from_output(out_file, k)[0]

            # pad if fewer neighbors
            if len(result) < k:
                result += [i] * (k - len(result))

            knn[i] = result

    return knn
