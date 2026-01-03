import numpy as np
from typing import List, Tuple, Dict, Set
import time
from Bio import Align


def compute_recall_at_n(
    ann_results: List[int],
    ground_truth: List[int],
    N: int
) -> float:
    ann_top_n = set(ann_results[:N])
    blast_top_n = set(ground_truth[:N])
    
    intersection = ann_top_n & blast_top_n
    
    if len(blast_top_n) == 0:
        return 0.0
    
    recall = len(intersection) / len(blast_top_n)
    return recall


def compute_average_recall(
    all_ann_results: List[List[int]],
    all_ground_truth: List[List[int]],
    N: int
) -> float:
    recalls = []
    
    for ann_res, blast_res in zip(all_ann_results, all_ground_truth):
        recall = compute_recall_at_n(ann_res, blast_res, N)
        recalls.append(recall)
    
    return np.mean(recalls)


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
    retrieved_at_k = set(retrieved[:k])
    relevant_retrieved = retrieved_at_k & relevant
    
    return len(relevant_retrieved) / k if k > 0 else 0.0


def compute_map(
    retrieved_lists: List[List[int]],
    relevant_sets: List[Set[int]]
) -> float:
    average_precisions = []
    
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        if len(relevant) == 0:
            continue
        
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

    def __init__(self):
        self.query_times = []
        self.ann_results = []
        self.ground_truth = []
        self.current_start = None
        
        self.metrics = {}
        
        self.current_method = None
        self.build_start = None
        self.search_start = None
        self.build_time = 0.0
        self.search_time = 0.0
    
    def start_build(self, method_name: str = None):
        self.current_method = method_name
        self.build_start = time.time()
        
        if method_name and method_name not in self.metrics:
            self.metrics[method_name] = {}
    
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
    
    def end_search(self, method_name: str = None, num_queries: int = 0):
        if self.search_start is None:
            raise RuntimeError("start_search() not called")
        
        method = method_name or self.current_method
        self.search_time = time.time() - self.search_start
        self.search_start = None
        
        qps = compute_qps(num_queries, self.search_time) if num_queries > 0 else 0.0
        
        if method:
            if method not in self.metrics:
                self.metrics[method] = {}
            self.metrics[method]['search_time'] = self.search_time
            self.metrics[method]['num_queries'] = num_queries
            self.metrics[method]['qps'] = qps
            self.metrics[method]['avg_query_time'] = self.search_time / num_queries if num_queries > 0 else 0.0
    
    def start_query(self):
        self.current_start = time.time()
    
    def end_query(self, ann_result: List[int], gt_result: List[int]):
        if self.current_start is None:
            raise RuntimeError("start_query() not called")
        
        elapsed = time.time() - self.current_start
        self.query_times.append(elapsed)
        self.ann_results.append(ann_result)
        self.ground_truth.append(gt_result)
        self.current_start = None
    
    def get_metrics(self, N: int = 50) -> Dict:
        num_queries = len(self.query_times)
        
        if num_queries == 0:
            return {
                'num_queries': 0,
                'recall_at_n': 0.0,
                'qps': 0.0,
                'avg_query_time': 0.0,
                'build_time': self.build_time,
            }
        
        recall = compute_average_recall(
            self.ann_results,
            self.ground_truth,
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
    
    # Sort by recall (descending)
    sorted_methods = sorted(
        results_dict.items(),
        key=lambda x: x[1].get('recall_at_n', 0),
        reverse=True
    )
    
    for method_name, metrics in sorted_methods:
        recall = metrics.get('recall_at_n', 0.0)
        qps = metrics.get('qps', 0.0)
        avg_time = metrics.get('avg_query_time', 0.0) * 1000  # Convert to ms
        
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

def compute_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Compute sequence identity percentage using pairwise alignment.
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Identity percentage (0-100)
    """
    if not seq1 or not seq2:
        return 0.0
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    # Use Biopython's pairwise alignment
    alignments = aligner.align(seq1, seq2)
    
    if not alignments:
        return 0.0
    
    alignment = alignments[0]
    aligned_seq1 = alignment.seqA
    aligned_seq2 = alignment.seqB
    
    # Count matches
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    
    # Identity = matches / min(len(seq1), len(seq2))
    identity = 100.0 * matches / min(len(seq1), len(seq2))
    
    return identity