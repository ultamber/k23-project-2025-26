# models.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------
# MLP ταξινομητής για τα partitions του Neural LSH
# -------------------------------------------------------------
class MLPClassifier(nn.Module):
    """
    Απλό πλήρως συνδεδεμένο MLP για ταξινόμηση σε m bins.
    d_in  : διάσταση εισόδου (128 για SIFT, 784 για MNIST)
    n_out : αριθμός partitions (m)
    layers: συνολικός αριθμός Linear layers (>= 2)
    nodes : # hidden units ανά κρυφό layer
    """

    def __init__(self, d_in, n_out, layers=3, nodes=64, dropout=0.0):
        super().__init__()

        if layers < 2:
            raise ValueError("layers must be >= 2 (τουλάχιστον 1 κρυφό + 1 output layer)")

        blocks = []
        in_dim = d_in

        # κρυφά layers: (Linear -> ReLU -> Dropout)
        for _ in range(layers - 1):
            blocks.append(nn.Linear(in_dim, nodes))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(p=dropout))
            in_dim = nodes

        # output layer (logits για m κλάσεις)
        blocks.append(nn.Linear(in_dim, n_out))

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------
# Training loop
# -------------------------------------------------------------
def _to_tensor(X, y):
    """Βοηθητικό: μετατροπή numpy -> torch tensors."""
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)
    return X, y


def train_model(
    model,
    X,
    y,
    epochs=10,
    batch_size=128,
    lr=1e-3,
    weight_decay=0.0,
    verbose=True,
):
    """
    Εκπαίδευση MLP με CrossEntropyLoss.

    model  : instance of MLPClassifier
    X      : numpy array ή tensor shape (n, d)
    y      : labels (partition ids) shape (n,)
    epochs : #περίοδοι εκπαίδευσης
    batch_size : μέγεθος batch
    lr     : learning rate (Adam)
    weight_decay : L2 regularization (0 αν δεν θέλεις)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # μετατροπή σε tensors
    X, y = _to_tensor(X, y)
    dataset = TensorDataset(X, y)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    model.train()
    n_samples = len(dataset)

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            # accuracy
            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()

        avg_loss = running_loss / n_samples
        acc = correct / n_samples

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"- loss: {avg_loss:.4f} "
                f"- acc: {acc*100:.2f}%"
            )

    model.to("cpu")
    return model
