# Sweep Experiments — Summary Report

Generated: 2025-12-06

Scanned folder: `neural_lsh/sweep_experiments-pt2`

This report summarizes the experiments recorded in `board.md` (each row corresponds to one experiment folder). For each experiment we collected the flattened configuration parameters and five final metrics from the corresponding `search_results.txt`:

- `Average AF`
- `Recall@5`
- `QPS` (queries per second)
- `tApproximateAverage` (average approximate search time)
- `tTrueAverage` (average brute-force time)

Summary (from `board.md`):

- Total experiments scanned: 24

Overview of trends

- There is a clear trade-off between Recall@5 and throughput (QPS): experiments with very high QPS (tens of queries/sec) tend to have low Recall@5, while experiments with Recall@5 ≈ 1.0 have low QPS (single-digit).
- The `T` parameter (number of bins probed / search probes) is the primary driver of this trade-off in these runs: small `T` values (e.g. 30) give high QPS but low recall; larger `T` (50–100) increase recall while reducing QPS.
- The `m` parameter (number of partitions) and `knn` appear to have modest impact compared with `T` in this subset.

Top results by metric

Top 10 experiments by QPS (highest throughput):

| Rank | Folder                                                               |       QPS | Recall@5 |   T |   m | knn |
| ---: | -------------------------------------------------------------------- | --------: | -------: | --: | --: | --: |
|    1 | `knn50_m200_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T30`  | 35.873570 | 0.187800 |  30 | 200 |  50 |
|    2 | `knn25_m200_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T30`  | 25.581522 | 0.187800 |  30 | 200 |  25 |
|    3 | `knn50_m200_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T50`  | 24.741245 | 0.262600 |  50 | 200 |  50 |
|    4 | `knn25_m200_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T50`  | 18.853476 | 0.262600 |  50 | 200 |  25 |
|    5 | `knn50_m200_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T75`  | 15.452061 | 0.401600 |  75 | 200 |  50 |
|    6 | `knn25_m200_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T100` |  6.839020 | 0.496000 | 100 | 200 |  25 |
|    7 | `knn25_m100_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T30`  | 11.307445 | 0.298600 |  30 | 100 |  25 |
|    8 | `knn50_m100_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T30`  | 12.768504 | 0.298600 |  30 | 100 |  50 |
|    9 | `knn25_m100_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T50`  |  7.424169 | 0.544000 |  50 | 100 |  25 |
|   10 | `knn50_m100_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T50`  |  9.307233 | 0.544000 |  50 | 100 |  50 |

Top experiments by Recall@5 (highest accuracy):

| Rank | Folder                          | Recall@5 |     QPS |      T |      m |   knn |
| ---: | ------------------------------- | -------: | ------: | -----: | -----: | ----: |
|    1 | several experiments (many ties) | 1.000000 | 3.0–4.0 | 50–100 | 50–100 | 25–50 |

|
Selected high-recall runs (Recall@5 = 1.0) with highest QPS among them:

| Folder                                                               | Recall@5 |      QPS |   T |   m | knn |
| -------------------------------------------------------------------- | -------: | -------: | --: | --: | --: |
| `knn25_m100_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T100` | 1.000000 | 3.723044 | 100 | 100 |  25 |
| `knn50_m100_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T100` | 1.000000 | 3.664125 | 100 | 100 |  50 |
| `knn25_m50_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T50`   | 1.000000 | 3.626982 |  50 |  50 |  25 |
| `knn25_m50_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T75`   | 1.000000 | 3.161932 |  75 |  50 |  25 |
| `knn50_m50_ep50_L4_N512_imb0.03_lr0.01_wd0.001_dr0.5_batch256_T50`   | 1.000000 | 3.041342 |  50 |  50 |  50 |

Observations and interpretation

- Tuning `T` (probe count) controls recall vs speed:

  - `T = 30` often produced the highest QPS (fastest), but Recall@5 dropped (≈0.18–0.30).
  - `T = 50` improved Recall substantially (often 0.26–0.54) while keeping QPS in the mid range (≈7–25 depending on `m` and `knn`).
  - `T >= 75` and `T = 100` tend to reach Recall@5 ≈ 1.0 in many configs, but QPS falls to the low single digits (~3–6 qps) in those runs.

- Increasing `m` (number of partitions) from 50 → 200 had mixed effects: for fixed `T` and `knn`, larger `m` sometimes reduced QPS but could slightly improve recall, depending on how inverted lists size changes.

- `knn` values (25 vs 50) show little systematic difference in Recall when other params are equal, but they do affect inverted-list sizes and therefore QPS in some runs.

Recommendations

- If your goal is maximum accuracy (Recall@5 ≈ 1.0): use `T` large (75–100) and `m` moderate (50–100). Expect QPS ≈ 3–6.
- If you want a balanced trade-off (good recall with reasonable speed): try `T=50` with `m=100` and `knn=25` or `50` (several experiments show Recall@5≈0.26–0.54 with QPS in the 7–25 range). For example, `knn25_m100_T50` yields Recall≈0.54 and QPS≈7.4.
- If throughput is the top priority and lower recall is acceptable: use `T=30` and larger `m` (e.g., `m=200`) to obtain QPS > 20, but expect Recall@5 < 0.3.

Suggested next experiments

1. Run a fine-grained sweep around `T=40..60` with both `m=100` and `m=200` to identify the Pareto frontier between Recall@5 and QPS.
2. Try `knn` values other than 25/50 (e.g., 10, 100) to see whether k-NN graph density changes the inverted-list size distribution and shifts the trade-off.
3. Collect per-experiment memory usage and candidate set sizes; these will help explain QPS differences beyond `T` and `m`.

Appendix — top entries (raw)

The full `board.md` contains the flattened configs and raw metric strings. See `neural_lsh/sweep_experiments-pt2/board.md` for the complete table. This report highlights the top rows by QPS and the highest-recall runs with their QPS for quick comparison.

---

If you want, I can:

- Produce a sortable CSV/Excel that ranks experiments by multiple columns, or
- Generate plots (Recall@5 vs QPS) and per-`T` boxplots to visualize variance, or
- Filter experiments by thresholds (e.g., Recall@5 ≥ 0.5) and export only the matching folders for deeper analysis.

Tell me which output you prefer and I'll generate it next.
