import torch


def save_index(path, model, X, labels, args=None):
    """
    Αποθηκεύει όλα τα components του Neural LSH index.
    """
    import json
    import pickle
    from pathlib import Path
    import torch
    import numpy as np

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # 1. Αποθήκευση μοντέλου
    torch.save(model.state_dict(), path/"model.pth")

    # 2. Inverted index
    inverted = {}
    for i, lab in enumerate(labels):
        lab = int(lab)
        inverted.setdefault(lab, []).append(i)

    with open(path/"inverted_index.pkl", "wb") as f:
        pickle.dump(inverted, f)

    # 3. Dataset
    np.save(path/"dataset.npy", X)

    # 4. Metadata
    if args is not None:
        metadata = {
            "dataset_type": args.type,
            "dimension": X.shape[1],
            "n_vectors": X.shape[0],
            "knn_graph_k": args.knn,
            "m": args.m,
            "layers": args.layers,
            "nodes": args.nodes,
            "dropout": getattr(args, "dropout", 0.0),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        }

        with open(path/"metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Index saved under directory: {path}")


def set_seed(seed=1):
    import random
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)