#  **README –  Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα**
Χειμερινό εξάμηνο 2025-26

1η Προγραμματιστική Εργασία

Αναζήτηση διανυσμάτων στη C/C++

**Συγγραφείς:**
**Ονοματεπώνυμο:** Μαρκοπούλου Χριστίνα ,Βασταρδής Νικόλαος

**Αριθμός Μητρώου:** 1115201800109  ,1115201900020 

**Neural LSH: Locality-Sensitive Hashing with Neural Networks**

Υλοποίηση του Neural LSH για προσεγγιστική αναζήτηση κοντινότερων γειτόνων, συνδυάζοντας διαμέριση γραφήματος με ταξινόμηση μέσω νευρωνικών δικτύων.

---

## Περιεχόμενα

- [Επισκόπηση](#overview)
- [Εγκατάσταση](#installation)
- [Δομή Έργου](#project-structure)
- [Γρήγορη Εκκίνηση](#quick-start)
- [Λεπτομερής Χρήση](#detailed-usage)
- [Κύριες Παράμετροι](#key-parameters)
- [Πειραματική Σύγκριση](#experimental-comparison)
- [Αντιμετώπιση Προβλημάτων](#troubleshooting)
- [Λεπτομέρειες Υλοποίησης](#implementation-details)
- [Συμμόρφωση με Ανάθεση](#assignment-compliance)

---

## Επισκόπηση

Το Neural LSH κατασκευάζει ένα ευρετήριο για γρήγορη προσεγγιστική αναζήτηση κοντινότερων γειτόνων μέσω:

1. **Κατασκευή Γραφήματος k-NN**: Δημιουργία γραφήματος που συνδέει όμοια σημεία  
2. **Διαμέριση Γραφήματος**: Χρήση KaHIP για διαμέριση του γραφήματος σε m ισορροπημένα bins  
3. **Εκπαίδευση MLP**: Εκπαίδευση νευρωνικού δικτύου για πρόβλεψη σε ποιο bin βρίσκονται οι γείτονες ενός ερωτήματος  
4. **Αναζήτηση Multi-Probe**: Αναζήτηση στα top-T προβλεπόμενα bins για κοντινότερους γείτονες

### Κύρια Χαρακτηριστικά

- Υποστηρίζει τα σύνολα δεδομένων MNIST και SIFT  
- Διαμορφώσιμη αρχιτεκτονική MLP (στρώματα, κόμβοι, dropout)  
- Προωθημένη διακοπή εκπαίδευσης με validation split
- Βελτίωση δομής και αλγορίθμων πρώτης εργασίας.

---

## Επισκόπηση Λειτουργίας

### Κύρια Σημεία

- Το Neural LSH δημιουργεί ένα δείκτη που επιτρέπει γρήγορη και αποτελεσματική αναζήτηση κοντινότερων γειτόνων.
- Συνδυάζει γραφήματα k-NN με διαμέριση γραφήματος και ταξινόμηση μέσω νευρωνικών δικτύων.
- Η αναζήτηση πραγματοποιείται σε bins με υψηλή πιθανότητα να περιέχουν τους γείτονες του ερωτήματος.

---
## Installation

### Prerequisites

- Python 3.10+
- KaHIP graph partitioning library
- Project 1 search executable (`../bin/search`)

### Setup

```bash
# Move into neural_lsh folder
cd ./neural_lsh

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
neural_lsh/
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

--

## Quick Start

### Build Index (MNIST)

```bash
python nlsh_build.py \
  -d ../datasets/MNIST/input.dat \
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
  -d ../datasets/MNIST/input.dat \
  -q ../datasets/MNIST/query.dat \
  -i ./mnist_index \
  -o ./results.txt \
  -type mnist \
  -N 10 \
  -T 5
```


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

---

## Key Parameters

### Build Phase

| Parameter  | Effect              | Recommended |
| ---------- | ------------------- | ----------- |
| `--knn`    | Graph connectivity  | 10-20       |
| `-m`       | Number of bins      | 50-200      |
| `--layers` | MLP depth           | 2-4         |
| `--nodes`  | MLP width           | 32-128      |
| `--epochs` | Training iterations | 30-100      |

### Search Phase

| Parameter | Effect              | Recommended |
| --------- | ------------------- | ----------- |
| `-N`      | Neighbors to return | 1-100       |
| `-T`      | Bins to probe       | 1-10        |

**Trade-off:** Higher T = better recall, slower search

---

## Λεπτομέρειες Υλοποίησης

### Φάση Κατασκευής Δείκτη

- Φορτώνονται τα δεδομένα εκπαίδευσης.  
- Δημιουργείται το γράφημα k-NN και συμμετρικοποιείται με κανόνες βάρους ακμών.  
- Το γράφημα διαμερίζεται σε bins μέσω KaHIP.  
- Εκπαιδεύεται MLP για να προβλέπει την κατανομή των γειτόνων στα bins.  
- Αποθηκεύονται το μοντέλο και τα απαραίτητα αρχεία ευρετηρίου.

### Φάση Αναζήτησης

- Φορτώνονται το ευρετήριο και τα ερωτήματα.  
- Το MLP προβλέπει πιθανότητες για κάθε bin.  
- Επιλέγονται τα κορυφαία T bins και συλλέγονται οι υποψήφιοι γείτονες.  
- Υπολογίζονται οι ακριβείς αποστάσεις και επιστρέφονται οι N κοντινότεροι γείτονες.  
- Υπολογίζονται μετρικές όπως AF, Recall και QPS.

---

## Αρχιτεκτονική MLP

- Το δίκτυο αποτελείται από στρώμα εισόδου, πολλαπλά κρυφά στρώματα με ReLU και dropout, και στρώμα εξόδου που παρέχει πιθανότητες bins μέσω softmax.  
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Εκπαίδευση: 90% εκπαίδευση, 10% validation  
- Early stopping: υπομονή 25 εποχές

---

## Κατασκευή Γραφήματος

- Οι ακμές συμμετρικοποιούνται: αμοιβαίες ακμές = βάρος 2, μη αμοιβαίες = βάρος 1  
- Η συμμετρικοποίηση απαιτείται για τη σωστή λειτουργία της KaHIP.

---

### Required Programs

1. `nlsh_build.py` - Index construction
2. `nlsh_search.py` - Search with compliant output

## Κύριες Παράμετροι

### Φάση Κατασκευής

- `--knn`: Συνδεσιμότητα γραφήματος  
- `-m`: Αριθμός bins  
- `--layers`: Βάθος MLP  
- `--nodes`: Πλάτος MLP  
- `--epochs`: Αριθμός επαναλήψεων εκπαίδευσης

### Φάση Αναζήτησης

- `-N`: Αριθμός γειτόνων προς επιστροφή  
- `-T`: Αριθμός bins προς έρευνα  

**Συμβιβασμός:** Υψηλότερο T = καλύτερο recall, πιο αργή αναζήτηση

---

## Expected Results

### MNIST (60k train, 10k test)

| Config           | Recall@10 | QPS  | Build Time |
| ---------------- | --------- | ---- | ---------- |
| m=50, T=3, k=10  | ~70%      | ~100 | 15 min     |
| m=100, T=5, k=15 | ~85%      | ~50  | 25 min     |
| m=200, T=7, k=20 | ~92%      | ~30  | 40 min     |

### SIFT (1M base, 10k query)

| Config           | Recall@10 | QPS | Build Time |
| ---------------- | --------- | --- | ---------- |
| m=100, T=5, k=15 | ~75%      | ~20 | 2 hours    |
| m=200, T=7, k=20 | ~88%      | ~15 | 4 hours    |

---

## Αναφορές

- Τεκμηρίωση KaHIP: https://github.com/KaHIP/KaHIP

---

## Ευχαριστίες

Η υλοποίηση βασίζεται σε:

- Dong et al., "Scalable k-NN graph construction for visual descriptors"  
- KaHIP: Karlsruhe High Quality Partitioning  
- PyTorch: Framework για βαθιά μάθηση