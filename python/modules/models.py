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
    epochs=50,
    batch_size=128,
    lr=1e-3,
    weight_decay=0.0,
    val_split=0.1,  # Add validation
    patience=25,      # Early stopping
    verbose=True,
):
    """Train MLP with validation and early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split train/val
    X, y = _to_tensor(X, y)
    n = len(X)
    n_val = int(n * val_split)
    n_train = n - n_val
    
    indices = torch.randperm(n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()

        train_loss = running_loss / n_train
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

        if verbose:
            print(
                f"Epoch {epoch:03d}/{epochs} "
                f"- train_loss: {train_loss:.4f} train_acc: {train_acc*100:.2f}% "
                f"- val_loss: {val_loss:.4f} val_acc: {val_acc*100:.2f}%"
            )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_acc: {best_val_acc*100:.2f}%")

    model.to("cpu")
    return model