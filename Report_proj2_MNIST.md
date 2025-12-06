# MNIST Approximate Nearest Neighbor Search: Comprehensive Experimental Report

**Dataset:** MNIST

---

## Executive Summary

This report analyzes ~200 experiments comparing two graph structures (KNN=25 vs KNN=50) for approximate nearest neighbor search on MNIST. The KNN=25 experiments include an extensive hyperparameter sweep across learning rate, weight decay, dropout, batch size, and network depth, while KNN=50 experiments focus on candidate pool size (m) and search iterations (T) variations.

**Key Findings:**

1. **Optimal hyperparameters:** lr=0.01, weight_decay=0.001, dropout=0.5, batch_size=256, layers=4
2. **Graph structure:** KNN=25 and KNN=50 achieve similar peak performance. KNN=25 shows marginally better recall at intermediate T values
3. **Critical trade-off:** Parameter m dominates the recall-QPS trade-off. T acts as a fine-tuning mechanism
4. **Best configuration:** m=50, T≥50 achieves perfect recall (1.0) at ~3.5 QPS regardless of graph structure

---

## 1. Experimental Design

### 1.1 Graph Structure Comparison

| Graph  | Experiments | Hyperparameter Sweep                              |
| ------ | ----------- | ------------------------------------------------- |
| KNN=25 | 185         | Full sweep (lr, wd, dropout, batch, layers, m, T) |
| KNN=50 | 24          | Fixed optimal hyperparams, vary m and T only      |

### 1.2 Hyperparameter Ranges (KNN=25 Sweep)

| Parameter          | Values              | Description                |
| ------------------ | ------------------- | -------------------------- |
| Learning Rate (lr) | 0.0001, 0.001, 0.01 | Optimizer step size        |
| Weight Decay (wd)  | 0.0, 0.0001, 0.001  | L2 regularization          |
| Dropout (dr)       | 0.0, 0.2, 0.5       | Regularization via dropout |
| Batch Size         | 128, 256            | Training batch size        |
| Layers             | 4, 5, 8             | Network depth              |
| m                  | 50, 100, 200        | Candidate pool size        |
| T                  | 30, 50, 75, 100     | Search iterations          |

### 1.3 Fixed Parameters

| Parameter   | Value |
| ----------- | ----- |
| N           | 5     |
| Epochs      | 50    |
| Imbalance   | 0.03  |
| Nodes       | 512   |
| Max Queries | 1000  |

---

## 2. Hyperparameter Analysis (KNN=25)

### 2.1 Learning Rate Impact

| lr       | Mean Recall@5 | Max Recall | Mean QPS |
| -------- | ------------- | ---------- | -------- |
| 0.0001   | 0.540         | 0.641      | 8.93     |
| 0.001    | 0.530         | 0.609      | 9.21     |
| **0.01** | **0.548**     | **1.000**  | 8.77     |

**Finding:** lr=0.01 is the only learning rate achieving perfect recall. Lower learning rates plateau around 0.64 recall.

### 2.2 Weight Decay Impact

| Weight Decay | Mean Recall@5 | Max Recall | Mean QPS |
| ------------ | ------------- | ---------- | -------- |
| 0.0          | 0.513         | 0.599      | 9.03     |
| 0.0001       | 0.514         | 0.641      | 8.94     |
| **0.001**    | **0.578**     | **1.000**  | 8.88     |

**Finding:** Light regularization (wd=0.001) significantly improves both mean and maximum recall.

### 2.3 Dropout Impact

| Dropout | Mean Recall@5 | Max Recall | Mean QPS |
| ------- | ------------- | ---------- | -------- |
| 0.0     | 0.522         | 0.641      | 8.89     |
| 0.2     | 0.514         | 0.638      | 9.01     |
| **0.5** | **0.571**     | **1.000**  | 8.93     |

**Finding:** Higher dropout (0.5) provides the best regularization, enabling perfect recall configurations.

### 2.4 Batch Size Impact

| Batch Size | Mean Recall@5 | Max Recall | Mean QPS |
| ---------- | ------------- | ---------- | -------- |
| 128        | 0.521         | 0.641      | 9.18     |
| **256**    | **0.556**     | **1.000**  | 8.76     |

**Finding:** Larger batch size (256) enables better recall while maintaining competitive throughput.

### 2.5 Network Depth Impact

| Layers | Mean Recall@5 | Max Recall | Mean QPS | Mean Query Time (s) |
| ------ | ------------- | ---------- | -------- | ------------------- |
| **4**  | **0.573**     | **1.000**  | 8.85     | 0.131               |
| 5      | 0.511         | 0.618      | 8.78     | 0.115               |
| 8      | 0.524         | 0.641      | 9.24     | 0.109               |

**Finding:** Shallow networks (4 layers) achieve best recall. Deeper networks (5, 8 layers) fail to reach perfect recall despite faster individual queries.

---

## 3. Search Parameter Analysis

### 3.1 Impact of m (Candidate Pool Size)

| m      | Mean Recall@5 | Max Recall | Mean QPS | Mean Speedup |
| ------ | ------------- | ---------- | -------- | ------------ |
| **50** | **0.904**     | **1.000**  | 4.33     | 0.76×        |
| 100    | 0.532         | 1.000      | 8.86     | 1.38×        |
| 200    | 0.352         | 0.532      | 15.46    | 2.69×        |

**Critical Insight:** m is the dominant factor controlling the recall-throughput trade-off:

- m=50 achieves highest recall but provides no speedup over exact search (0.76×)
- m=100 offers a balance with 1.38× speedup and up to perfect recall with T=100
- m=200 provides 2.69× speedup but recall never exceeds 53.2%

### 3.2 Impact of T (Search Iterations)

| T       | Mean Recall@5 | Max Recall | Mean QPS |
| ------- | ------------- | ---------- | -------- |
| 30      | 0.388         | 0.615      | 14.87    |
| **50**  | 0.527         | **1.000**  | 8.96     |
| **75**  | 0.755         | **1.000**  | 6.23     |
| **100** | 0.844         | **1.000**  | 5.46     |

**Finding:** T≥50 is sufficient for perfect recall with appropriate m values. Higher T improves average recall but with diminishing returns on throughput.

---

## 4. KNN=25 vs KNN=50 Comparison

### 4.1 Recall@5 Comparison (Matched Hyperparameters)

Using identical settings (lr=0.01, wd=0.001, dr=0.5, batch=256, layers=4):

**KNN=25 Graph:**

| m \ T | 30    | 50    | 75    | 100   |
| ----- | ----- | ----- | ----- | ----- |
| 50    | 0.615 | 1.000 | 1.000 | 1.000 |
| 100   | 0.381 | 0.667 | 0.855 | 1.000 |
| 200   | 0.169 | 0.297 | 0.409 | 0.532 |

**KNN=50 Graph:**

| m \ T | 30    | 50    | 75    | 100   |
| ----- | ----- | ----- | ----- | ----- |
| 50    | 0.631 | 1.000 | 1.000 | 1.000 |
| 100   | 0.299 | 0.544 | 0.794 | 1.000 |
| 200   | 0.188 | 0.263 | 0.402 | 0.496 |

**Analysis:**

- Both graphs achieve identical perfect recall (1.0) for m=50 with T≥50
- KNN=25 shows **better intermediate recall** at m=100 (0.667 vs 0.544 at T=50; 0.855 vs 0.794 at T=75)
- KNN=50 slightly better for m=50, T=30 (0.631 vs 0.615)
- At m=200, KNN=25 achieves marginally better recall across all T values

### 4.2 Throughput (QPS) Comparison

**KNN=25 Graph:**

| m \ T | 30    | 50    | 75    | 100  |
| ----- | ----- | ----- | ----- | ---- |
| 50    | 6.80  | 3.49  | 3.59  | 3.47 |
| 100   | 11.63 | 7.42  | 4.55  | 3.67 |
| 200   | 26.18 | 15.86 | 10.55 | 9.24 |

**KNN=50 Graph:**

| m \ T | 30    | 50    | 75    | 100  |
| ----- | ----- | ----- | ----- | ---- |
| 50    | 5.84  | 3.33  | 3.21  | 3.24 |
| 100   | 12.04 | 8.37  | 5.45  | 3.69 |
| 200   | 30.73 | 21.80 | 13.12 | 9.09 |

**Analysis:**

- KNN=50 achieves **higher throughput** at high-m configurations (up to 30.73 vs 26.18 QPS)
- At m=50 (optimal recall), both graphs achieve similar QPS (~3.2-3.7)
- The KNN=50 graph's larger neighborhood provides faster convergence but slightly lower recall at intermediate settings

---

## 5. Optimal Configurations

### Configurations Achieving Perfect Recall (1.0)

All 8 perfect-recall configurations share these hyperparameters:

- **lr:** 0.01
- **weight_decay:** 0.001
- **dropout:** 0.5
- **batch_size:** 256
- **layers:** 4

**With fastest QPS:**

| Rank | Graph  | m   | T   | Recall@5 | QPS  |
| ---- | ------ | --- | --- | -------- | ---- |
| 1    | KNN=50 | 100 | 100 | 1.000    | 3.72 |
| 2    | KNN=25 | 100 | 100 | 1.000    | 3.68 |
| 3    | KNN=50 | 100 | 100 | 1.000    | 3.66 |
| 4    | KNN=50 | 50  | 50  | 1.000    | 3.63 |
| 5    | KNN=25 | 50  | 75  | 1.000    | 3.61 |

---

## 6. Conclusions

### Optimal Hyperparameters

For MNIST approximate nearest neighbor search, use:

```
lr = 0.01
weight_decay = 0.001
dropout = 0.5
batch_size = 256
layers = 4
```

These settings consistently achieve the best results across both graph structures.

---
