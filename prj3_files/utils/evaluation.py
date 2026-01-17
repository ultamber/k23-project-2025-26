import numpy as np
from typing import List, Tuple, Dict, Set
import time


def compute_recall(
    ann_results: List[int],
    ground_truth: List[int],
    N: int
) -> float:
    """
    Compute recall@N for a single query.

    Recall@N = |ANN_top_N ∩ BLAST_top_N| / |BLAST_top_N|
    """
    ann_top_n = set(ann_results[:N])  # Top N results from ANN method
    blast_top_n = set(ground_truth[:N])  # Top N results from ground truth (BLAST)

    intersection = ann_top_n & blast_top_n  # Common results in both sets

    if len(blast_top_n) == 0:
        return 0.0  # Avoid division by zero

    recall = len(intersection) / len(blast_top_n)  # Fraction of BLAST results found by ANN
    return recall


def compute_average_recall(
    all_ann_results: List[List[int]],
    all_ground_truth: List[List[int]],
    N: int
) -> float:
    """
    Compute average recall@N across multiple queries.
    """
    recalls = []

    for ann_res, blast_res in zip(all_ann_results, all_ground_truth):
        recall = compute_recall(ann_res, blast_res, N)
        recalls.append(recall)

    return np.mean(recalls)  # Average recall across all queries


def compute_qps(
    num_queries: int,
    total_time: float
) -> float:
    if total_time <= 0:
        return 0.0

    return num_queries / total_time


def compute_precision_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute precision@k for a single query.

    Precision@k = |retrieved_top_k ∩ relevant| / k
    """
    retrieved_at_k = set(retrieved[:k])  # Top k retrieved items
    relevant_retrieved = retrieved_at_k & relevant  # Relevant items in top k

    return len(relevant_retrieved) / k if k > 0 else 0.0  # Fraction of top k that are relevant


def compute_map(
    retrieved_lists: List[List[int]],
    relevant_sets: List[Set[int]]
) -> float:
    """
    Compute Mean Average Precision (MAP) across multiple queries.

    For each query:
    - AP = average of precision@i for each relevant item at position i
    - MAP = mean of AP across all queries
    """
    average_precisions = []

    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if len(relevant) == 0:
            continue  # Skip queries with no relevant items

        precisions = []
        num_relevant = 0

        for k, item in enumerate(retrieved, 1):
            if item in relevant:
                num_relevant += 1
                precision_at_k = num_relevant / k
                precisions.append(precision_at_k)

        if precisions:
            avg_precision = np.mean(precisions)
            average_precisions.append(avg_precision)

    return np.mean(average_precisions) if average_precisions else 0.0


class PerformanceTracker:
    """
    Tracks performance metrics for ANN search methods, including build time,
    search time, QPS, and per-query timing.
    """

    def __init__(self):
        self.metrics = {}  # Stores aggregated metrics per method
        self.current_method = None
        self.build_start = None
        self.search_start = None
        self.build_time = 0.0
        self.search_time = 0.0

        # Per-query tracking
        self.per_query_times = {}  # method -> list of query times
        self.query_start = None
        self.query_times = []  # Legacy, not used in current implementation
        self.ann_results = []   # Legacy

    def start_build(self, method_name: str = None):
        self.current_method = method_name
        self.build_start = time.time()

        if method_name and method_name not in self.metrics:
            self.metrics[method_name] = {}
            self

    def end_build(self, method_name: str = None):
        if self.build_start is None:
            raise RuntimeError("start_build() not called")

        method = method_name or self.current_method
        self.build_time = time.time() - self.build_start
        self.build_start = None

        if method:
            if method not in self.metrics:
                self.metrics[method] = {}
            self.metrics[method]['build_time'] = self.build_time

    def start_search(self, method_name: str = None):
        self.current_method = method_name or self.current_method
        self.search_start = time.time()

        if self.current_method not in self.per_query_times:
            self.per_query_times[self.current_method] = []

    def end_search(self, method_name: str = None, num_queries: int = 0):
        if self.search_start is None:
            raise RuntimeError("start_search() not called")

        method = method_name or self.current_method
        self.search_time = time.time() - self.search_start
        self.search_start = None

        qps = num_queries / self.search_time if self.search_time > 0 else 0.0
        avg_query_time = self.search_time / num_queries if num_queries > 0 else 0.0

        if method:
            if method not in self.metrics:
                self.metrics[method] = {}
            self.metrics[method]['search_time'] = self.search_time
            self.metrics[method]['num_queries'] = num_queries
            self.metrics[method]['qps'] = qps
            self.metrics[method]['avg_query_time'] = avg_query_time

    def start_query(self):
        self.query_start = time.time()

    def end_query(self, method_name: str = None):
        if self.query_start is None:
            return 0.0

        method = method_name or self.current_method
        query_time = time.time() - self.query_start
        self.query_start = None

        if method:
            if method not in self.per_query_times:
                self.per_query_times[method] = []
            self.per_query_times[method].append(query_time)

        return query_time

    def get_per_query_times(self, method_name: str) -> List[float]:
        return self.per_query_times.get(method_name, [])

    def get_per_query_qps(self, method_name: str) -> List[float]:
        """
        Get per-query QPS values (1/time for each query).
        """
        times = self.per_query_times.get(method_name, [])
        return [1.0 / t if t > 0 else 0.0 for t in times]

    def get_metrics(self, N: int = 50) -> Dict:
        """
        Compute comprehensive metrics including recall, QPS, and timing statistics.
        Note: This method uses legacy attributes (query_times, ann_results, ground_truth)
        that may not be populated in the current implementation.
        """
        num_queries = len(self.query_times)  # Note: uses legacy query_times

        if num_queries == 0:
            return {
                'num_queries': 0,
                'recall_at_n': 0.0,
                'qps': 0.0,
                'avg_query_time': 0.0,
                'build_time': self.build_time,
            }

        recall = compute_average_recall(
            self.ann_results,  # Legacy: list of ANN result lists
            self.ground_truth,  # Legacy: list of ground truth lists
            N
        )

        total_time = sum(self.query_times)
        qps = compute_qps(num_queries, total_time)

        avg_time = np.mean(self.query_times)

        return {
            'num_queries': num_queries,
            'recall_at_n': recall,
            'qps': qps,
            'avg_query_time': avg_time,
            'total_time': total_time,
            'min_query_time': min(self.query_times),
            'max_query_time': max(self.query_times),
            'build_time': self.build_time,
        }

    def reset(self):
        self.query_times = []
        self.ann_results = []
        self.ground_truth = []
        self.current_start = None
        self.build_time = 0.0
        self.build_start = None
        self.search_start = None
        self.search_time = 0.0
        self.metrics = {}
        self.current_method = None


def compare_methods(
    results_dict: Dict[str, Dict],
    N: int = 50
) -> None:
    print(f"\n{'='*80}")
    print(f"Performance comparisons (Recall@{N})")
    print(f"{'='*80}")

    print(f"{'Method':<20} {'Recall@N':<12} {'QPS':<12} {'Avg Time (ms)':<15}")
    print(f"{'-'*80}")

    # Sort by recall (descending) to show best methods first
    sorted_methods = sorted(
        results_dict.items(),
        key=lambda x: x[1].get('recall_at_n', 0),
        reverse=True
    )

    for method_name, metrics in sorted_methods:
        recall = metrics.get('recall_at_n', 0.0)
        qps = metrics.get('qps', 0.0)
        avg_time = metrics.get('avg_query_time', 0.0) * 1000  # Convert to milliseconds

        print(f"{method_name:<20} {recall:<12.4f} {qps:<12.2f} {avg_time:<15.2f}")

    print(f"{'='*80}\n")


def save_results(
    results_dict: Dict[str, Dict],
    output_file: str,
    N: int = 50
):
    import json
    from pathlib import Path

    output_data = {
        'N': N,
        'methods': results_dict,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_file}")

def compute_recall_at_n(
    ann_results: List[List[Tuple[int, float]]],
    blast_results: Dict[int, List[Tuple]],
    N: int
) -> float:
    """
    Compute average recall@N across queries using BLAST ground truth.

    This is a more sophisticated recall calculation that handles:
    - Variable number of ground truth results per query
    - ANN results may have fewer than N neighbors
    - Ensures denominator is |BLAST_top_N ∩ unique_indices|

    For each query q:
    - Get BLAST_top_N = first N BLAST results (may have duplicates if BLAST returns multiples)
    - Take unique indices: S_BLAST = set of unique neighbor indices in BLAST_top_N
    - n_eff = |S_BLAST| (effective ground truth size)
    - Get ANN_top_k where k = min(n_eff, len(ANN_results[q]))
    - S_ANN = set of indices in ANN_top_k
    - recall_q = |S_BLAST ∩ S_ANN| / n_eff

    Returns average recall across all queries with ground truth.
    """
    total = 0.0  # Sum of recall values
    counted = 0  # Number of queries with valid ground truth
    missing = 0  # Number of queries missing from blast_results

    for q_idx, ann_neighbors in enumerate(ann_results):
        if q_idx not in blast_results:
            missing += 1
            continue  # Skip queries without ground truth

        # Cap BLAST list to N, then use its actual size
        blast_hits = blast_results[q_idx][:N]  # First N BLAST results (may include scores)
        if not blast_hits:
            continue  # No BLAST results for this query

        blast_set = {hit[0] for hit in blast_hits}  # Unique neighbor indices from BLAST
        n_eff = len(blast_set)  # Effective ground truth size (|S_BLAST|)

        k = min(n_eff, len(ann_neighbors))  # Don't take more ANN results than ground truth
        ann_set = {idx for idx, _ in ann_neighbors[:k]}  # ANN results up to k
        total += len(blast_set & ann_set) / n_eff   # |intersection| / n_eff
        counted += 1

    if missing:
        print(f"WARNING: {missing} queries missing from blast_results")
    print(f"Computed recall for {counted} queries")

    return total / counted if counted else 0.0  # Average recall
