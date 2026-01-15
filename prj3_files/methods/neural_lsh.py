import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Dict, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict
import pickle
import json
from pathlib import Path
import time


class MLPClassifier(nn.Module):

    def __init__(self, d_in: int, n_out: int, layers: int = 3, 
                 nodes: int = 128, dropout: float = 0.2):
        super().__init__()
        
        if layers < 2:
            raise ValueError("layers must be >= 2")
        
        blocks = []
        in_dim = d_in
        
        # Hidden layers
        for _ in range(layers - 1):
            blocks.append(nn.Linear(in_dim, nodes))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(p=dropout))
            in_dim = nodes
        
        # Output layer
        blocks.append(nn.Linear(in_dim, n_out))
        
        self.net = nn.Sequential(*blocks)
    
    def forward(self, x):
        return self.net(x)


def train_mlp(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    val_split: float = 0.1,
    patience: int = 10,
    verbose: bool = True
) -> nn.Module:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Train/val split
    n = len(X)
    n_val = int(n * val_split)
    n_train = n - n_val
    
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Data loader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer & loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # Early stopping
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
        
        train_loss /= n_train
        train_acc = correct / n_train
        
        # Validation
        model.eval()
        with torch.no_grad():
            X_val_gpu = X_val.to(device)
            y_val_gpu = y_val.to(device)
            
            val_logits = model(X_val_gpu)
            val_loss = loss_fn(val_logits, y_val_gpu).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val_gpu).float().mean().item()
        
        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:03d}/{epochs} - "
                  f"train_loss: {train_loss:.4f} train_acc: {train_acc*100:.2f}% - "
                  f"val_loss: {val_loss:.4f} val_acc: {val_acc*100:.2f}%")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(f"Loaded best model with val_acc: {best_val_acc*100:.2f}%")
    
    model.to("cpu")
    return model
# TODO : change metric to support cosine similarity

class NeuralLSH:

    def __init__(
        self,
        m: int = 100,
        k_neighbors: int = 25,
        mlp_layers: int = 3,
        mlp_nodes: int = 128,
        mlp_dropout: float = 0.2,
        seed: int = 42,
        hidden_dims: List[int] = [128, 128],
        epochs: int = 50,
        metric: str = 'L2'
    ):
        self.m = m
        self.k_neighbors = k_neighbors
        self.mlp_layers = mlp_layers
        self.mlp_nodes = mlp_nodes
        self.mlp_dropout = mlp_dropout
        self.seed = seed
        self.epochs = epochs
        self.hidden_dims = hidden_dims
        self.metric = metric
        
        self.model = None
        self.inverted_index = None
        self.embeddings = None
        self.partitions = None
        self.dim = None
        self.n = None
        
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def build_index(
        self,
        embeddings: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        use_kahip: bool = True,
        verbose: bool = True
    ):

        self.n, self.dim = embeddings.shape
        self.embeddings = embeddings.astype(np.float32)
        
        if verbose:
            print(f"Building Neural LSH index:")
            print(f"N = {self.n} proteins")
            print(f"D = {self.dim} dimensions")
            print(f"m = {self.m} partitions")
            print(f"k = {self.k_neighbors} neighbors")
        
        # Step 1: Build k-NN graph
        if verbose:
            print(f"\nBuilding k-NN graph...")
        knn_graph = self._build_knn_graph()
        
        # Step 2: Partition graph
        if verbose:
            print(f"\nPartitioning graph...")
        
        if use_kahip:
            # Try KaHIP (requires external binary)
            try:
                print(f"Using KaHIP for partitioning...")
                partitions = self._partition_kahip(knn_graph)
            except Exception as e:
                if verbose:
                    print(f"KaHIP failed: {e}")
                    print(f"Falling back to k-means...")
                partitions = self._partition_kmeans()
        else:
            # Use k-means
            print(f"Using k-means for partitioning...")
            partitions = self._partition_kmeans()
        
        self.partitions = partitions
        
        # Build inverted index
        self.inverted_index = defaultdict(list)
        for i, p in enumerate(partitions):
            self.inverted_index[int(p)].append(i)
        
        # Print partition stats
        partition_sizes = np.bincount(partitions)
        if verbose:
            print(f"Partitions: {len(partition_sizes)} clusters")
            print(f"  Size: min={partition_sizes.min()}, "
                  f"max={partition_sizes.max()}, "
                  f"mean={partition_sizes.mean():.1f}")
        
        # Step 3: Train MLP
        if verbose:
            print(f"\nTraining MLP classifier...")
            print(f"Layers: {self.mlp_layers}")
            print(f"Nodes: {self.mlp_nodes}")
            print(f"Dropout: {self.mlp_dropout}")
            print(f"Epochs: {epochs}")
        
        self.model = MLPClassifier(
            d_in=self.dim,
            n_out=self.m,
            layers=self.mlp_layers,
            nodes=self.mlp_nodes,
            dropout=self.mlp_dropout
        )
        
        self.model = train_mlp(
            self.model,
            self.embeddings,
            partitions,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            verbose=verbose
        )
        
        if verbose:
            print(f"\nNeural LSH index built!")
    
    def _build_knn_graph(self) -> np.ndarray:
        nbrs = NearestNeighbors(
            n_neighbors=self.k_neighbors + 1,  # +1 for self
            metric='euclidean',
            algorithm='auto'
        )
        nbrs.fit(self.embeddings)
        _, indices = nbrs.kneighbors(self.embeddings)
        
        # Remove self from neighbors
        knn_graph = indices[:, 1:]  # Skip first column (self)
        
        return knn_graph
    
    def _partition_kmeans(self) -> np.ndarray:
        """Partition using k-means (fallback)."""
        kmeans = KMeans(
            n_clusters=self.m,
            random_state=self.seed,
            n_init=10
        )
        labels = kmeans.fit_predict(self.embeddings)
        return labels.astype(np.int32)
    
    def _partition_kahip(self, knn_graph: np.ndarray) -> np.ndarray:
        # Import graph utilities from original implementation
        from utils.graph_utils import build_symmetric_graph, to_csr, run_kahip
        
        # Build symmetric graph
        adj = build_symmetric_graph(knn_graph)
        
        # Convert to CSR
        csr = to_csr(adj)
        
        # Run KaHIP
        labels = run_kahip(
            csr,
            m=self.m,
            imbalance=0.03,
            mode=2,  # strong
            seed=self.seed
        )
        
        return labels
    
    def search(
        self,
        query: np.ndarray,
        N: int = 10,
        T: int = 5
    ) -> List[Tuple[int, float]]:

        if self.model is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Predict partition probabilities
        self.model.eval()
        with torch.no_grad():
            q_tensor = torch.tensor(query, dtype=torch.float32).unsqueeze(0)
            logits = self.model(q_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Select top-T partitions
        top_partitions = np.argsort(-probs)[:T]
        
        # Collect candidates
        candidates = []
        for p in top_partitions:
            p = int(p)
            if p in self.inverted_index:
                candidates.extend(self.inverted_index[p])
        
        if len(candidates) == 0:
            # Fallback: random sample
            candidates = np.random.choice(
                self.n,
                min(N * 10, self.n),
                replace=False
            ).tolist()
        
        candidates = list(set(candidates))  # Remove duplicates
        
        # Compute distances
        candidate_vecs = self.embeddings[candidates]
        distances = np.linalg.norm(candidate_vecs - query, axis=1)
        
        # Find top-N
        if len(distances) >= N:
            top_idx = np.argpartition(distances, N-1)[:N]
            top_idx = top_idx[np.argsort(distances[top_idx])]
        else:
            top_idx = np.argsort(distances)
        
        results = [
            (candidates[i], distances[i])
            for i in top_idx[:N]
        ]
        
        return results
    
    def batch_search(
        self,
        queries: np.ndarray,
        N: int = 10,
        T: int = 5,
        verbose: bool = True
    ) -> List[List[Tuple[int, float]]]:
        results = []
        num_queries = len(queries)
        
        if verbose:
            from tqdm import tqdm
            iterator = tqdm(range(num_queries), desc="Neural LSH search")
        else:
            iterator = range(num_queries)
        
        for qi in iterator:
            result = self.search(queries[qi], N=N, T=T)
            results.append(result)
        
        return results
    
    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path / "model.pth")
        
        # Save inverted index
        with open(path / "inverted_index.pkl", "wb") as f:
            pickle.dump(dict(self.inverted_index), f)
        
        # Save embeddings
        np.save(path / "embeddings.npy", self.embeddings)
        
        # Save partitions
        np.save(path / "partitions.npy", self.partitions)
        
        # Save metadata
        metadata = {
            "n": self.n,
            "dim": self.dim,
            "m": self.m,
            "k_neighbors": self.k_neighbors,
            "mlp_layers": self.mlp_layers,
            "mlp_nodes": self.mlp_nodes,
            "mlp_dropout": self.mlp_dropout,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Neural LSH index saved to: {path}")
    
    @staticmethod
    def load(path: str) -> 'NeuralLSH':
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            meta = json.load(f)
        
        # Create instance
        nlsh = NeuralLSH(
            m=meta["m"],
            k_neighbors=meta["k_neighbors"],
            mlp_layers=meta["mlp_layers"],
            mlp_nodes=meta["mlp_nodes"],
            mlp_dropout=meta["mlp_dropout"]
        )
        
        # Load model
        nlsh.model = MLPClassifier(
            d_in=meta["dim"],
            n_out=meta["m"],
            layers=meta["mlp_layers"],
            nodes=meta["mlp_nodes"],
            dropout=meta["mlp_dropout"]
        )
        state = torch.load(path / "model.pth", map_location="cpu")
        nlsh.model.load_state_dict(state)
        nlsh.model.eval()
        
        # Load other data
        with open(path / "inverted_index.pkl", "rb") as f:
            nlsh.inverted_index = pickle.load(f)
        
        nlsh.embeddings = np.load(path / "embeddings.npy")
        nlsh.partitions = np.load(path / "partitions.npy")
        nlsh.n = meta["n"]
        nlsh.dim = meta["dim"]
        
        print(f"Neural LSH index loaded from: {path}")
        return nlsh
    
    def get_stats(self) -> Dict:
        partition_sizes = [
            len(self.inverted_index[p])
            for p in range(self.m)
            if p in self.inverted_index
        ]
        
        return {
            'n_points': self.n,
            'n_partitions': self.m,
            'k_neighbors': self.k_neighbors,
            'occupied_partitions': len(partition_sizes),
            'avg_partition_size': np.mean(partition_sizes) if partition_sizes else 0,
            'max_partition_size': max(partition_sizes) if partition_sizes else 0,
            'min_partition_size': min(partition_sizes) if partition_sizes else 0,
        }


if __name__ == '__main__':
    import argparse
    import pickle
    from pathlib import Path
    import time
    
    parser = argparse.ArgumentParser(
        description="Neural LSH for Protein Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required (database OR load-index)
    parser.add_argument('-d', '--database', help='Database embeddings (.npy)')
    parser.add_argument('-q', '--queries', required=True, help='Query embeddings (.npy)')
    parser.add_argument('-o', '--output', required=True, help='Output results file (.pkl)')
    
    # Search parameters
    parser.add_argument('--N', type=int, default=50, help='Number of neighbors (default: 50)')
    parser.add_argument('--T', type=int, default=10, help='Partitions to probe (default: 10)')
    
    # Neural LSH parameters
    parser.add_argument('--m', type=int, default=100, help='Number of partitions (default: 100)')
    parser.add_argument('--k', type=int, default=25, help='k for k-NN graph (default: 25)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--mlp-layers', type=int, default=3, help='MLP layers (default: 3)')
    parser.add_argument('--mlp-nodes', type=int, default=128, help='MLP hidden nodes (default: 128)')
    
    # Options
    parser.add_argument('--max-queries', type=int, help='Limit number of queries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save-index', help='Save built index to directory')
    parser.add_argument('--load-index', help='Load pre-built index from directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate: need database OR load-index
    if not args.load_index and not args.database:
        parser.error("Either --database or --load-index is required")
    
    print("="*70)
    print("Neural LSH for Protein Embeddings")
    print("="*70)
    
    # Load queries
    print(f"\nLoading query embeddings...")
    query_embeddings = np.load(args.queries)
    print(f"Queries: {query_embeddings.shape}")
    
    if args.max_queries:
        query_embeddings = query_embeddings[:args.max_queries]
        print(f"Limited to {len(query_embeddings)} queries")
    
    # Build or load index
    if args.load_index and Path(args.load_index).exists():
        print(f"\nLoading pre-trained index from {args.load_index}...")
        nlsh = NeuralLSH.load(args.load_index)
        print(f"Index loaded")
    else:
        # Load database
        print(f"\nLoading database embeddings...")
        db_embeddings = np.load(args.database)
        print(f"Database: {db_embeddings.shape}")
        
        # Build index
        print(f"\nBuilding Neural LSH index...")
        print(f"m={args.m} partitions, k={args.k} neighbors")
        print(f"MLP: {args.mlp_layers} layers x {args.mlp_nodes} nodes")
        print(f"Training: {args.epochs} epochs")
        
        nlsh = NeuralLSH(
            m=args.m,
            k_neighbors=args.k,
            mlp_layers=args.mlp_layers,
            mlp_nodes=args.mlp_nodes,
            seed=args.seed
        )
        
        t0 = time.time()
        nlsh.build_index(
            db_embeddings,
            epochs=args.epochs,
            verbose=args.verbose
        )
        build_time = time.time() - t0
        
        print(f"Index built in {build_time:.2f}s ({build_time/60:.1f} min)")
        
        # Save index if requested
        if args.save_index:
            print(f"Saving index to {args.save_index}...")
            nlsh.save(args.save_index)
            print(f"Index saved")
    
    # Search
    print(f"\n[Step {4 if not args.load_index else 3}] Searching {len(query_embeddings)} queries...")
    print(f"T={args.T} partitions to probe per query")
    
    t0 = time.time()
    results = nlsh.batch_search(query_embeddings, N=args.N, T=args.T, verbose=args.verbose)
    search_time = time.time() - t0
    
    qps = len(query_embeddings) / search_time if search_time > 0 else 0
    
    print(f"Search completed in {search_time:.2f}s")
    print(f"QPS: {qps:.2f}")
    
    # Save results
    print(f"\n[Step {5 if not args.load_index else 4}] Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'results': results,
        'params': {
            'method': 'neural-lsh',
            'm': nlsh.m,
            'k_neighbors': nlsh.k_neighbors,
            'T': args.T,
            'N': args.N,
            'num_queries': len(query_embeddings),
            'num_database': nlsh.n,
        },
        'metrics': {
            'search_time': search_time,
            'qps': qps,
            'avg_query_time': search_time / len(query_embeddings),
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"Results saved to {output_path}")
    
    
    print("Summary:")
    print(f"{'='*70}")
    print(f"Database: {nlsh.n} proteins")
    print(f"Queries: {len(query_embeddings)}")
    print(f"Top-N: {args.N}")
    print(f"Partitions: {nlsh.m}")
    print(f"Probes per query: {args.T}")
    print(f"QPS: {qps:.2f}")
    print(f"Avg query time: {search_time/len(query_embeddings)*1000:.2f} ms")
    
    print(f"\nSample results (first query):")
    for i, (idx, dist) in enumerate(results[0][:5], 1):
        print(f"{i}. Index {idx}: distance = {dist:.4f}")
    
    
    print("Neural LSH Search Completed")
    