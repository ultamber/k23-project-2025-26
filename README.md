# **README – Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα**

Χειμερινό εξάμηνο 2025-26

2η Προγραμματιστική Εργασία

Αναζήτηση διανυσμάτων στην Python

**Συγγραφείς:**
**Ονοματεπώνυμο:** Μαρκοπούλου Χριστίνα, Βασταρδής Νικόλαος

**Αριθμός Μητρώου:** 1115201800109, 1115201900020

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

## Εγκατάσταση

### Προαπαιτούμενα

- Python 3.10+
- Βιβλιοθήκη διαμέρισης γραφήματος KaHIP
- Εκτελέσιμο αναζήτησης από την Εργασία 1 (`../bin/search`)

### Setup

```bash
# Move into neural_lsh folder
cd ./neural_lsh

# Create virtual environment
python3 -m venv nlsh_env
source nlsh_env/bin/activate  # Σε Windows: nlsh_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
kahip>=3.21
```

### Δομή Συνόλων Δεδομένων

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

## Δομή Έργου

```
neural_lsh/
├── nlsh_build.py              # Κατασκευή ευρετηρίου
├── nlsh_search.py             # Αναζήτηση συμβατή με την ανάθεση
├── modules/
│   ├── models.py              # Ταξινομητής MLP
│   ├── lsh_knn.py             # Κατασκευή γραφήματος k-NN
│   ├── graph_utils.py         # Συμμετρικοποίηση γραφήματος & KaHIP
│   ├── dataset_parser.py      # Φόρτωση συνόλων δεδομένων
│   └── utils.py               # Βοηθητικές συναρτήσεις
└── requirements.txt
```

---

## Quick Start

### Κατασκευή Ευρετηρίου (MNIST)

```bash
python nlsh_build.py \
  -d ../datasets/MNIST/input.dat \
  -i ./database_index \
  -type sift \
  --knn 15 \
  -m 100 \
  --method ivfflat \
  --epochs 50 \
  --layers 3 \
  --nodes 64 \
  --batch_size 128 \
  --lr 0.001 \
  --calculated_output ../out_ivfflat.txt
```

### Αναζήτηση

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

---

## Λεπτομερής Χρήση

### Κατασκευή Ευρετηρίου

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

### Αναζήτηση

#### Αναζήτηση

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

## Κύριες Παράμετροι

### Φάση Κατασκευής

| Παράμετρος | Επίδραση                 | Συνιστώμενη |
| ---------- | ------------------------ | ----------- |
| `--knn`    | Συνδεσιμότητα γραφήματος | 10-20       |
| `-m`       | Αριθμός bins             | 50-200      |
| `--layers` | Βάθος MLP                | 2-4         |
| `--nodes`  | Πλάτος MLP               | 32-128      |
| `--epochs` | Επαναλήψεις εκπαίδευσης  | 30-100      |

### Φάση Αναζήτησης

| Παράμετρος | Επίδραση                | Συνιστώμενη |
| ---------- | ----------------------- | ----------- |
| `-N`       | Γείτονες προς επιστροφή | 1-100       |
| `-T`       | Bins προς έρευνα        | 1-10        |

**Συμβιβασμός:** Υψηλότερο T = καλύτερο recall, πιο αργή αναζήτηση

---

## Λεπτομέρειες Υλοποίησης

### Φάση Κατασκευής Ευρετηρίου

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
- Συνάρτηση Απώλειας: CrossEntropyLoss
- Βελτιστοποιητής: Adam
- Εκπαίδευση: 90% εκπαίδευση, 10% validation
- Πρόωρη διακοπή: υπομονή 25 epochs

---

## Κατασκευή Γραφήματος

- Οι ακμές συμμετρικοποιούνται: αμοιβαίες ακμές = βάρος 2, μη αμοιβαίες = βάρος 1
- Η συμμετρικοποίηση απαιτείται για τη σωστή λειτουργία της KaHIP.

---

## Απαιτούμενα Προγράμματα

1. `nlsh_build.py` - Κατασκευή ευρετηρίου
2. `nlsh_search.py` - Αναζήτηση

---

## Αναμενόμενα Αποτελέσματα (σύνοψη από πειράματα)

Οι παρακάτω πίνακες συνοψίζουν τυπικά αποτελέσματα που παρατηρήθηκαν στα πειράματά μας (δείτε `Report_MNIST.md` και `Report_SIFT.md` για πλήρεις λεπτομέρειες). Οι τιμές είναι σε εύρη που προέκυψαν από σάρωση των παραμέτρων `m`, `T` και γραφήματος k-NN. Προορίζονται ως καθοδήγηση και όχι ως αυστηρές εγγυήσεις.

### MNIST (60k training set, 10k queries)

| Μετρική / Τυπική διαμόρφωση                    | Αναμενόμενο Recall@10 | Αναμενόμενο QPS (query/Sec.) | Τυπικός χρόνος κατασκευής |
| ---------------------------------------------- | --------------------: | ---------------------------: | ------------------------: |
| Neural-LSH (υψηλή ακρίβεια: T >= 75, m=50–100) |                 ~100% |                        3 – 6 |                15–45 λεπ. |
| Neural-LSH (ισορροπημένο: T ≈ 50, m=100)       |                50–90% |                       7 – 25 |                20–40 λεπ. |
| Neural-LSH (υψηλή απόδοση: T ≈ 30, m=200)      |                10–30% |                     20 – 35+ |                30–60 λεπ. |

Σημειώσεις: Το Neural-LSH τυπικά επιτυγχάνει πολύ υψηλό recall (συχνά 1.0 με επαρκώς μεγάλο `T`) αλλά με κόστος χαμηλότερου QPS από μεθόδους κατακερματισμού/κβαντοποίησης. Ο αριθμός ερευνών `T` είναι ο κύριος έλεγχος για τον συμβιβασμό recall έναντι ταχύτητας.

### SIFT (1M base, 10k queries)

| Μετρική / Τυπική διαμόρφωση | Αναμενόμενο Recall@10 |       Αναμενόμενο QPS (query/sec.) |     Τυπικός χρόνος κατασκευής |
| --------------------------- | --------------------: | ---------------------------------: | ----------------------------: |
| Neural-LSH (υψηλή ακρίβεια) |                 ~100% |                            30 – 90 | ώρες (εξαρτάται από δεδομένα) |
| Κλασσικό (IVFPQ / IVFFlat)  |             0.97–1.00 | 300 – 8.500 (εξαρτάται από μέθοδο) |                          ώρες |

Σημειώσεις: Για πειράματα SIFT μεγάλης κλίμακας το απόλυτο QPS μπορεί να είναι υψηλότερο από το MNIST στο περιβάλλον μας (παρατηρήθηκε 30–90 για ορισμένες εκτελέσεις Neural-LSH), αλλά οι χρόνοι κατασκευής και οι απαιτήσεις πόρων αυξάνονται σημαντικά· το IVFFlat και οι βελτιστοποιημένες δομές ευρετηρίου παραμένουν η προτιμώμενη επιλογή όταν η απόδοση είναι προτεραιότητα.

---

## Αναφορές

- KaHIP: https://github.com/KaHIP/KaHIP

---
