# dataset_parser.py
import numpy as np

def load_mnist(path):
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Invalid MNIST magic number: {magic}")

        n_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")

        dim = rows * cols

        data = np.frombuffer(f.read(n_images * dim), dtype=np.uint8)

    # Μετατροπή σε float32
    X = data.reshape(n_images, dim).astype(np.float32)

    return X


def load_sift(path):
    vectors = []

    with open(path, "rb") as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break  # EOF

            dim = np.frombuffer(dim_bytes, dtype="<i4")[0]
            if dim != 128:
                raise ValueError(f"Unexpected SIFT dimension: {dim}")

            vec = np.frombuffer(f.read(dim * 4), dtype="<f4")
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32)


def load_dataset(path, dtype):
    if dtype == "mnist":
        return load_mnist(path)
    elif dtype == "sift":
        return load_sift(path)
    else:
        raise ValueError("Unsupported dataset type")
