"""
Embedding I/O utilities for Python <-> C++ interoperability.

This module provides functions to save and load embeddings in formats
compatible with the C++ ANN search implementations.

Formats supported:
- .fvecs: Standard format for float vectors (used by SIFT, compatible with your C++)
- .npy: NumPy format (for Python-only workflows)
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, List
import struct


# =============================================================================
# FVECS Format (C++ Compatible)
# =============================================================================

def save_fvecs(filename: Union[str, Path], vectors: np.ndarray) -> None:
    """
    Save vectors in .fvecs format for C++ consumption.
    
    Format: For each vector, write [dim (int32), values (float32 * dim)]
    This is the same format used by SIFT and read by your SIFT_Dataset::load()
    
    Args:
        filename: Output file path
        vectors: NumPy array of shape (n_vectors, dimension)
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, d = vectors.shape
    
    with open(filename, 'wb') as f:
        for i in range(n):
            # Write dimension as int32
            f.write(struct.pack('<i', d))
            # Write vector values as float32
            f.write(vectors[i].tobytes())
    
    print(f"Saved {n} vectors of dimension {d} to {filename}")


def load_fvecs(filename: Union[str, Path]) -> np.ndarray:
    """
    Load vectors from .fvecs format.
    
    Args:
        filename: Input file path
        
    Returns:
        NumPy array of shape (n_vectors, dimension)
    """
    vectors = []
    
    with open(filename, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            
            d = struct.unpack('<i', dim_bytes)[0]
            
            # Read vector values
            vec_bytes = f.read(d * 4)
            if len(vec_bytes) < d * 4:
                break
                
            vec = np.frombuffer(vec_bytes, dtype=np.float32)
            vectors.append(vec)
    
    result = np.vstack(vectors) if vectors else np.array([], dtype=np.float32)
    print(f"Loaded {len(vectors)} vectors of dimension {result.shape[1] if len(vectors) > 0 else 0}")
    return result


# =============================================================================
# IVECS Format (for integer data like IDs or ground truth indices)
# =============================================================================

def save_ivecs(filename: Union[str, Path], vectors: np.ndarray) -> None:
    """
    Save integer vectors in .ivecs format (e.g., for ground truth indices).
    
    Args:
        filename: Output file path
        vectors: NumPy array of shape (n_vectors, dimension), integer type
    """
    vectors = np.ascontiguousarray(vectors, dtype=np.int32)
    n, d = vectors.shape
    
    with open(filename, 'wb') as f:
        for i in range(n):
            f.write(struct.pack('<i', d))
            f.write(vectors[i].tobytes())
    
    print(f"Saved {n} integer vectors of dimension {d} to {filename}")


def load_ivecs(filename: Union[str, Path]) -> np.ndarray:
    """
    Load integer vectors from .ivecs format.
    
    Args:
        filename: Input file path
        
    Returns:
        NumPy array of shape (n_vectors, dimension)
    """
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
                
            vec = np.frombuffer(vec_bytes, dtype=np.int32)
            vectors.append(vec)
    
    return np.vstack(vectors) if vectors else np.array([], dtype=np.int32)


# =============================================================================
# ID Mapping (to preserve protein IDs)
# =============================================================================

def save_id_mapping(filename: Union[str, Path], ids: List[str]) -> None:
    """
    Save protein/sequence IDs to a text file.
    Line number corresponds to vector index in .fvecs file.
    
    Args:
        filename: Output file path
        ids: List of string IDs (e.g., UniProt accessions)
    """
    with open(filename, 'w') as f:
        for id_ in ids:
            f.write(f"{id_}\n")
    
    print(f"Saved {len(ids)} IDs to {filename}")


def load_id_mapping(filename: Union[str, Path]) -> List[str]:
    """
    Load protein/sequence IDs from a text file.
    
    Args:
        filename: Input file path
        
    Returns:
        List of string IDs
    """
    with open(filename, 'r') as f:
        ids = [line.strip() for line in f]
    
    print(f"Loaded {len(ids)} IDs")
    return ids


# =============================================================================
# Complete Pipeline Helpers
# =============================================================================

def save_embeddings_for_cpp(
    output_dir: Union[str, Path],
    database_embeddings: np.ndarray,
    database_ids: List[str],
    query_embeddings: Optional[np.ndarray] = None,
    query_ids: Optional[List[str]] = None,
    prefix: str = "protein"
) -> dict:
    """
    Save all embeddings and metadata for C++ consumption.
    
    Args:
        output_dir: Directory to save files
        database_embeddings: Database vectors (N, D)
        database_ids: Database sequence IDs
        query_embeddings: Query vectors (Q, D), optional
        query_ids: Query sequence IDs, optional
        prefix: Filename prefix
        
    Returns:
        Dictionary with paths to all created files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Save database
    db_vectors_path = output_dir / f"{prefix}_database.fvecs"
    db_ids_path = output_dir / f"{prefix}_database_ids.txt"
    
    save_fvecs(db_vectors_path, database_embeddings)
    save_id_mapping(db_ids_path, database_ids)
    
    paths['database_vectors'] = str(db_vectors_path)
    paths['database_ids'] = str(db_ids_path)
    
    # Save queries if provided
    if query_embeddings is not None:
        query_vectors_path = output_dir / f"{prefix}_queries.fvecs"
        save_fvecs(query_vectors_path, query_embeddings)
        paths['query_vectors'] = str(query_vectors_path)
        
        if query_ids is not None:
            query_ids_path = output_dir / f"{prefix}_query_ids.txt"
            save_id_mapping(query_ids_path, query_ids)
            paths['query_ids'] = str(query_ids_path)
    
    # Save metadata
    meta_path = output_dir / f"{prefix}_meta.txt"
    with open(meta_path, 'w') as f:
        f.write(f"n_database={len(database_embeddings)}\n")
        f.write(f"dimension={database_embeddings.shape[1]}\n")
        if query_embeddings is not None:
            f.write(f"n_queries={len(query_embeddings)}\n")
    
    paths['metadata'] = str(meta_path)
    
    print(f"\nAll files saved to {output_dir}/")
    return paths


def load_cpp_results(
    results_file: Union[str, Path],
    id_mapping: Optional[List[str]] = None
) -> dict:
    """
    Parse C++ search results output file.
    
    Args:
        results_file: Path to C++ output file
        id_mapping: Optional list to convert indices back to IDs
        
    Returns:
        Dictionary with parsed results
    """
    results = {
        'method': None,
        'queries': [],
        'summary': {}
    }
    
    current_query = None
    
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Method name (first non-empty line)
            if results['method'] is None and line and not line.startswith('Query'):
                results['method'] = line
                continue
            
            # Query start
            if line.startswith('Query:'):
                if current_query is not None:
                    results['queries'].append(current_query)
                current_query = {
                    'query_idx': int(line.split(':')[1].strip()),
                    'neighbors': [],
                    'distances_approx': [],
                    'distances_true': [],
                    'r_neighbors': []
                }
                continue
            
            # Neighbor info
            if line.startswith('Nearest neighbor-'):
                idx = int(line.split(':')[1].strip())
                if id_mapping and idx < len(id_mapping):
                    current_query['neighbors'].append(id_mapping[idx])
                else:
                    current_query['neighbors'].append(idx)
                continue
            
            if line.startswith('distanceApproximate:'):
                current_query['distances_approx'].append(float(line.split(':')[1].strip()))
                continue
                
            if line.startswith('distanceTrue:'):
                current_query['distances_true'].append(float(line.split(':')[1].strip()))
                continue
            
            # Summary stats
            if line.startswith('Average AF:'):
                results['summary']['avg_af'] = float(line.split(':')[1].strip())
            elif line.startswith('Recall@N:'):
                results['summary']['recall'] = float(line.split(':')[1].strip())
            elif line.startswith('QPS:'):
                results['summary']['qps'] = float(line.split(':')[1].strip())
            elif line.startswith('tApproximateAverage:'):
                results['summary']['avg_time_approx'] = float(line.split(':')[1].strip())
            elif line.startswith('tTrueAverage:'):
                results['summary']['avg_time_true'] = float(line.split(':')[1].strip())
    
    # Don't forget last query
    if current_query is not None:
        results['queries'].append(current_query)
    
    return results


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Create dummy data and save
    print("=" * 60)
    print("Example: Saving and Loading Embeddings")
    print("=" * 60)
    
    # Simulate ESM-2 embeddings (320-dimensional)
    n_database = 1000
    n_queries = 10
    dim = 320
    
    np.random.seed(42)
    database_embeddings = np.random.randn(n_database, dim).astype(np.float32)
    query_embeddings = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Normalize (like ESM-2 embeddings often are)
    database_embeddings /= np.linalg.norm(database_embeddings, axis=1, keepdims=True)
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    
    # Create fake protein IDs
    database_ids = [f"P{i:05d}" for i in range(n_database)]
    query_ids = [f"Q{i:03d}" for i in range(n_queries)]
    
    # Save for C++
    paths = save_embeddings_for_cpp(
        output_dir="./cpp_data",
        database_embeddings=database_embeddings,
        database_ids=database_ids,
        query_embeddings=query_embeddings,
        query_ids=query_ids,
        prefix="example"
    )
    
    print("\nCreated files:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Verify by loading back
    print("\n" + "=" * 60)
    print("Verification: Loading back")
    print("=" * 60)
    
    loaded_db = load_fvecs(paths['database_vectors'])
    loaded_ids = load_id_mapping(paths['database_ids'])
    
    print(f"Database shape: {loaded_db.shape}")
    print(f"First 5 IDs: {loaded_ids[:5]}")
    print(f"Vectors match: {np.allclose(database_embeddings, loaded_db)}")