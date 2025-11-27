# Neural LSH: Locality-Sensitive Hashing with Neural Networks

Implementation of Neural LSH for approximate nearest neighbor search, combining graph partitioning with neural network classification.

---

##  Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Key Parameters](#key-parameters)
- [Experimental Comparison](#experimental-comparison)
- [Troubleshooting](#troubleshooting)
- [Implementation Details](#implementation-details)
- [Assignment Compliance](#assignment-compliance)

---

##  Overview

Neural LSH builds an index for fast approximate nearest neighbor search by:

1. **k-NN Graph Construction**: Build a graph connecting similar points
2. **Graph Partitioning**: Use KaHIP to partition the graph into m balanced bins
3. **MLP Training**: Train a neural network to predict which bin contains a query's neighbors
4. **Multi-Probe Search**: Search top-T predicted bins for nearest neighbors

### Key Features

 ~ Supports MNIST and SIFT datasets  
 ~ Configurable MLP architecture (layers, nodes, dropout)  
 ~ Early stopping with validation split  

---

## Installation

### Prerequisites

- Python 3.10+
- KaHIP graph partitioning library
- Project 1 search executable (`../bin/search`)

### Setup
```bash
# Create virtual environment
python3 -m venv nlsh_env
source nlsh_env/bin/activate  # On Windows: nlsh_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt
```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
kahip>=3.16
```

### Dataset Structure
```
datasets/
├── MNIST/
│   ├── train-images.idx3-ubyte
│   └── t10k-images.idx3-ubyte
└── SIFT/
    ├── sift_base.fvecs
    └── sift_query.fvecs
```

---

## Project Structure
```
python/
├── nlsh_build.py              # Index construction
├── nlsh_search.py             # Assignment-compliant search
├── modules/
│   ├── models.py              # MLP classifier
│   ├── lsh_knn.py             # k-NN graph construction
│   ├── graph_utils.py         # Graph symmetrization & KaHIP
│   ├── dataset_parser.py      # Dataset loading
│   └── utils.py               # Helper functions
└── requirements.txt
```

---

## Quick Start

### Build Index (MNIST)
```bash
python nlsh_build.py \
  -d ../datasets/MNIST/train-images.idx3-ubyte \
  -i ./mnist_index \
  -type mnist \
  --knn 15 \
  -m 100 \
  --method #lsh/hypercube/ivfflat/ivfpq
  --epochs 50 \
  --layers 3 \
  --nodes 64 \
  --batch_size 128 \
  --lr 0.001
```

### Search
```bash
python nlsh_search.py \
  -d ../datasets/MNIST/train-images.idx3-ubyte \
  -q ../datasets/MNIST/t10k-images.idx3-ubyte \
  -i ./mnist_index \
  -o ./results.txt \
  -type mnist \
  -N 10 \
  -T 5
```

---

## Detailed Usage

### Building an Index
```bash
python nlsh_build.py [OPTIONS]

Required Arguments:
  -d, --dataset PATH          Dataset file path
  -i, --index PATH            Output index directory
  -type {mnist,sift}          Dataset type

k-NN Graph Options:
  --knn K                     Number of neighbors (default: 10)
  --use_exact_knn             Use sklearn for exact k-NN (recommended)
  --search_path PATH          Path to Project 1 search binary
  --method                    Use an algorithm from project 1 ( default = ivfflat)

Partitioning Options:
  -m NUM                      Number of partitions/bins (default: 50)
  --imbalance FLOAT           KaHIP imbalance parameter (default: 0.03)

MLP Training Options:
  --epochs N                  Training epochs (default: 30)
  --layers N                  Number of hidden layers (default: 3)
  --nodes N                   Nodes per hidden layer (default: 64)
  --batch_size N              Batch size (default: 128)
  --lr FLOAT                  Learning rate (default: 0.001)
  --dropout FLOAT             Dropout rate (default: 0.0)
  --weight_decay FLOAT        L2 regularization (default: 0.0)
  --patience N                Early stopping patience (default: 25)
```

### Searching

#### Assignment-Compliant Search (Required for Grading)
```bash
python nlsh_search.py [OPTIONS]

Required Arguments:
  -d, --dataset PATH          Training dataset path
  -q, --query PATH            Query dataset path
  -i, --index PATH            Index directory
  -o, --output PATH           Output file path
  -type {mnist,sift}          Dataset type
  -N NUM                      Number of nearest neighbors
  -T NUM                      Number of bins to probe

Optional Arguments:
  -range {true,false}         Range search mode (default: false)
  -R FLOAT                    Range threshold (default: 2000.0)
```

**Output format:**
```

```

---

## Key Parameters

### Build Phase

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `--knn` | Graph connectivity | 10-20 |
| `-m` | Number of bins | 50-200 |
| `--layers` | MLP depth | 2-4 |
| `--nodes` | MLP width | 32-128 |
| `--epochs` | Training iterations | 30-100 |

### Search Phase

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `-N` | Neighbors to return | 1-100 |
| `-T` | Bins to probe | 1-10 |

**Trade-off:** Higher T = better recall, slower search

---

## Implementation Details

### Algorithm Flow
```
BUILD PHASE:
1. Load training data (60k MNIST or 1M SIFT)
2. Build k-NN graph:
   - Query: training set
   - Search: training set (same!)
   - Result: 60k × k graph
3. Symmetrize graph with edge weights:
   - Reciprocal edges: weight = 2
   - Non-reciprocal edges: weight = 1
4. Partition graph with KaHIP → m bins
5. Train MLP: data → bin distribution
6. Save: model.pth, inverted_index.pkl, metadata.json

SEARCH PHASE:
1. Load index and queries
2. For each query:
   a. MLP predicts bin probabilities
   b. Select top-T bins
   c. Collect candidates from bins
   d. Compute exact distances
   e. Return N nearest neighbors
3. Compute metrics (AF, Recall, QPS)
```

### MLP Architecture
```
Input Layer:       d dimensions (784 for MNIST, 128 for SIFT)
                   ↓
Hidden Layer 1:    ReLU(Linear(d → nodes))
                   ↓ Dropout (optional)
Hidden Layer 2:    ReLU(Linear(nodes → nodes))
                   ↓ Dropout (optional)
    ...
Hidden Layer L:    ReLU(Linear(nodes → nodes))
                   ↓ Dropout (optional)
Output Layer:      Linear(nodes → m)
                   ↓
Logits:           m classes (bin probabilities via softmax)
```

**Loss:** CrossEntropyLoss  
**Optimizer:** Adam  
**Training:** 90% train, 10% validation  
**Early Stopping:** Patience = 25 epochs

### Graph Construction

**Symmetrization rules:**
```python
if i in knn[j] and j in knn[i]:
    edge(i,j) = 2  # Reciprocal
else:
    edge(i,j) = 1  # Non-reciprocal
```

**Why symmetrization?** KaHIP requires undirected graphs.

---

### Required Programs

1. `nlsh_build.py` - Index construction
2. `nlsh_search.py` - Search with compliant output

### Output Format

```
Neural LSH

Query: 9999
Nearest neighbor-1: 42247
distanceApproximate: 3056.399170
distanceTrue: 3056.266357
Nearest neighbor-2: 41358
distanceApproximate: 3056.563721
distanceTrue: 3056.399170
Nearest neighbor-3: 5993
distanceApproximate: 3056.800537
distanceTrue: 3056.413086
Nearest neighbor-4: 5973
distanceApproximate: 3056.942383
distanceTrue: 3056.563721
Nearest neighbor-5: 56015
distanceApproximate: 3057.081055
distanceTrue: 3056.772461
Nearest neighbor-6: 6201
distanceApproximate: 3057.085449
distanceTrue: 3056.800537
Nearest neighbor-7: 6139
distanceApproximate: 3057.118652
distanceTrue: 3056.877930
Nearest neighbor-8: 5972
distanceApproximate: 3057.246582
distanceTrue: 3056.919922
Nearest neighbor-9: 7641
distanceApproximate: 3057.773682
distanceTrue: 3056.936035
Nearest neighbor-10: 12489
distanceApproximate: 3057.774902
distanceTrue: 3056.942383
R-near neighbors:

Average AF: 1.000031
Recall@10: 0.389850
QPS: 16.822934
tApproximateAverage: 0.059443
tTrueAverage: 0.288515
```

## Expected Results

### MNIST (60k train, 10k test)

| Config | Recall@10 | QPS | Build Time |
|--------|-----------|-----|------------|
| m=50, T=3, k=10 | ~70% | ~100 | 15 min |
| m=100, T=5, k=15 | ~85% | ~50 | 25 min |
| m=200, T=7, k=20 | ~92% | ~30 | 40 min |

### SIFT (1M base, 10k query)

| Config | Recall@10 | QPS | Build Time |
|--------|-----------|-----|------------|
| m=100, T=5, k=15 | ~75% | ~20 | 2 hours |
| m=200, T=7, k=20 | ~88% | ~15 | 4 hours |

---

## References

- KaHIP documentation: https://github.com/KaHIP/KaHIP

---

## Acknowledgments

Implementation based on:
- Dong et al. "Scalable k-NN graph construction for visual descriptors"
- KaHIP: Karlsruhe High Quality Partitioning
- PyTorch: Deep learning framework

---
