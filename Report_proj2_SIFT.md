# SIFT Approximate Nearest Neighbor Search: Experimental Report

---

## Executive Summary

This report analyzes 24 experiments on the SIFT dataset, comparing two graph structures (KNN=25 vs KNN=50) for approximate nearest neighbor search. Using optimal hyperparameters identified from MNIST experiments, we evaluate the recall-throughput trade-off on this more challenging, higher-dimensional dataset.

**Key Findings:**

1. **Higher throughput:** SIFT achieves ~4× higher QPS than MNIST across all configurations
2. **Similar recall patterns:** Both datasets show identical recall behavior — perfect recall with m=50, T≥50
3. **Lower speedup:** The approximate method provides less speedup over exact search on SIFT
4. **Harder dataset:** SIFT shows marginally lower recall at intermediate T values compared to MNIST

---

## 1. Experimental Setup

### 1.1 Configuration

All experiments use the optimal hyperparameters identified from MNIST:

| Parameter     | Value |
| ------------- | ----- |
| Learning Rate | 0.01  |
| Weight Decay  | 0.001 |
| Dropout       | 0.5   |
| Batch Size    | 256   |
| Layers        | 4     |
| Nodes         | 512   |
| Epochs        | 50    |

### 1.2 Variable Parameters

| Parameter             | Values          |
| --------------------- | --------------- |
| Graph Structure (KNN) | 25, 50          |
| Candidate Pool (m)    | 50, 100, 200    |
| Search Iterations (T) | 30, 50, 75, 100 |

---

## 2. Results: KNN=25 Graph Structure

### 2.1 Recall@5 Matrix

| m \ T   | 30    | 50        | 75        | 100       |
| ------- | ----- | --------- | --------- | --------- |
| **50**  | 0.551 | **1.000** | **1.000** | **1.000** |
| **100** | 0.298 | 0.556     | 0.773     | **1.000** |
| **200** | 0.177 | 0.260     | 0.409     | 0.507     |

### 2.2 QPS (Queries Per Second) Matrix

| m \ T   | 30    | 50    | 75    | 100   |
| ------- | ----- | ----- | ----- | ----- |
| **50**  | 24.65 | 13.80 | 13.02 | 14.52 |
| **100** | 43.97 | 33.19 | 22.01 | 16.13 |
| **200** | 79.09 | 57.34 | 44.23 | 31.81 |

### 2.3 Speedup vs Exact Search

| m \ T   | 30    | 50    | 75    | 100   |
| ------- | ----- | ----- | ----- | ----- |
| **50**  | 0.99× | 0.57× | 0.56× | 0.54× |
| **100** | 1.71× | 1.21× | 0.76× | 0.57× |
| **200** | 2.77× | 2.07× | 1.58× | 1.17× |

---

## 3. Results: KNN=50 Graph Structure

### 3.1 Recall@5 Matrix

| m \ T   | 30    | 50        | 75        | 100       |
| ------- | ----- | --------- | --------- | --------- |
| **50**  | 0.636 | **1.000** | **1.000** | **1.000** |
| **100** | 0.332 | 0.517     | 0.772     | **1.000** |
| **200** | 0.145 | 0.247     | 0.355     | 0.475     |

### 3.2 QPS Matrix

| m \ T   | 30    | 50    | 75    | 100   |
| ------- | ----- | ----- | ----- | ----- |
| **50**  | 20.88 | 12.07 | 12.25 | 12.62 |
| **100** | 41.80 | 29.60 | 20.36 | 14.39 |
| **200** | 65.25 | 48.28 | 40.31 | 32.50 |

### 3.3 Speedup vs Exact Search

| m \ T   | 30    | 50    | 75    | 100   |
| ------- | ----- | ----- | ----- | ----- |
| **50**  | 1.00× | 0.56× | 0.57× | 0.59× |
| **100** | 1.80× | 1.18× | 0.77× | 0.55× |
| **200** | 2.61× | 2.06× | 1.48× | 1.17× |

---

## 4. KNN=25 vs KNN=50 Comparison

### 4.1 Recall Comparison

| Configuration | KNN=25    | KNN=50    |
| ------------- | --------- | --------- |
| m=50, T=30    | 0.551     | **0.636** |
| m=50, T≥50    | 1.000     | 1.000     |
| m=100, T=30   | 0.298     | **0.332** |
| m=100, T=50   | **0.556** | 0.517     |
| m=100, T=75   | 0.773     | 0.772     |
| m=100, T=100  | 1.000     | 1.000     |
| m=200, T=30   | **0.177** | 0.145     |
| m=200, T=50   | **0.260** | 0.247     |
| m=200, T=75   | **0.409** | 0.355     |
| m=200, T=100  | **0.507** | 0.475     |

**Summary:**

- KNN=50 performs better at low T values (T=30)
- KNN=25 performs better at higher m values (m=200)
- Both achieve identical perfect recall at optimal settings

### 4.2 Throughput Comparison

| m   | KNN=25 Avg QPS | KNN=50 Avg QPS | Difference  |
| --- | -------------- | -------------- | ----------- |
| 50  | 16.50          | 14.45          | KNN=25 +14% |
| 100 | 28.82          | 26.54          | KNN=25 +9%  |
| 200 | 53.12          | 46.58          | KNN=25 +14% |

**Finding:** KNN=25 consistently achieves ~10-14% higher throughput on SIFT.

---

## 5. Perfect Recall Configurations

Only **4 configurations** achieve perfect recall (1.0) on SIFT for each graph structure:

### KNN=25 — Perfect Recall

| m   | T   | Recall@5 | QPS       | Speedup |
| --- | --- | -------- | --------- | ------- |
| 100 | 100 | 1.000    | **16.13** | 0.57×   |
| 50  | 100 | 1.000    | 14.52     | 0.54×   |
| 50  | 50  | 1.000    | 13.80     | 0.57×   |
| 50  | 75  | 1.000    | 13.02     | 0.56×   |

### KNN=50 — Perfect Recall

| m   | T   | Recall@5 | QPS       | Speedup |
| --- | --- | -------- | --------- | ------- |
| 100 | 100 | 1.000    | **14.39** | 0.55×   |
| 50  | 100 | 1.000    | 12.62     | 0.59×   |
| 50  | 75  | 1.000    | 12.25     | 0.57×   |
| 50  | 50  | 1.000    | 12.07     | 0.56×   |

**Optimal Configuration:** m=100, T=100, KNN=25 — achieves 1.0 recall at 16.1 QPS

---

## 6. SIFT vs MNIST Comparison

### 6.1 Recall Comparison (KNN=25)

| Configuration | MNIST | SIFT  | Difference |
| ------------- | ----- | ----- | ---------- |
| m=50, T=30    | 0.615 | 0.551 | -0.064     |
| m=50, T=50    | 1.000 | 1.000 | 0          |
| m=100, T=30   | 0.381 | 0.298 | -0.083     |
| m=100, T=50   | 0.667 | 0.556 | -0.110     |
| m=100, T=75   | 0.855 | 0.773 | -0.082     |
| m=200, T=100  | 0.532 | 0.507 | -0.025     |

**Finding:** SIFT shows 5-11% lower recall than MNIST at intermediate configurations, indicating it's a harder dataset for approximate search.

### 6.2 Throughput Comparison

| Configuration | MNIST QPS | SIFT QPS | Ratio    |
| ------------- | --------- | -------- | -------- |
| m=50, T=50    | 3.49      | 13.80    | **4.0×** |
| m=100, T=100  | 3.67      | 16.13    | **4.4×** |
| m=200, T=75   | 10.55     | 44.23    | **4.2×** |
| **Average**   | —         | —        | **~4×**  |

**Finding:** SIFT achieves approximately **4× higher QPS** than MNIST across all configurations.

### 6.3 Speedup Comparison

| m   | MNIST Speedup | SIFT Speedup |
| --- | ------------- | ------------ |
| 50  | 0.76×         | 0.67×        |
| 100 | 1.29×         | 1.06×        |
| 200 | 2.69×         | 1.90×        |

**Finding:** SIFT shows **lower speedup ratios** — the approximate method provides less benefit relative to exact search on SIFT.

---
