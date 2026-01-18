import struct
from pathlib import Path
from typing import List
import numpy as np

def save_fvecs(filename: str, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = vectors.shape
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(vectors[i].tobytes())

def load_fvecs(filename: str) -> np.ndarray:
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            d = struct.unpack('<i', dim_bytes)[0]
            vec_bytes = f.read(d * 4)
            if len(vec_bytes) < d * 4:
                break
            vectors.append(np.frombuffer(vec_bytes, dtype=np.float32))
    return np.vstack(vectors) if vectors else np.array([], dtype=np.float32)

def export_for_cpp(
    output_dir_str: str,
    database_embeddings: np.ndarray,
    database_ids: List[str] | None,
    query_embeddings: np.ndarray,
    query_ids: List[str] | None,
    prefix: str = "protein"
) -> dict:
    """Export database and query embeddings to fvecs format for C++ consumption.
    Creates .fvecs files for vectors and .txt files for IDs.
    Returns dict with paths to created files."""
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if needed

    if database_ids is None:
        database_ids = []

    if query_ids is None:
        query_ids = []

    paths = {}

    # Database
    db_fvecs = output_dir / f"{prefix}_database.fvecs"
    db_ids_file = output_dir / f"{prefix}_database_ids.txt"
    save_fvecs(str(db_fvecs), database_embeddings)
    with open(db_ids_file, 'w') as f:
        for pid in database_ids:
            f.write(f"{pid}\n")
    paths['database'] = str(db_fvecs)
    paths['database_ids'] = str(db_ids_file)

    # Queries export
    query_fvecs = output_dir / f"{prefix}_queries.fvecs"
    query_ids_file = output_dir / f"{prefix}_query_ids.txt"
    save_fvecs(str(query_fvecs), query_embeddings)
    with open(query_ids_file, 'w') as f:
        for pid in query_ids:
            f.write(f"{pid}\n")
    paths['queries'] = str(query_fvecs)
    paths['query_ids'] = str(query_ids_file)

    print(f"\nExported for C++:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    return paths

