import numpy as np
from typing import Dict, List
import torch


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    k_values: List[int] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: [num_samples, num_candidates]
        labels: [num_samples]
        k_values: List of k values for Recall@K
    
    Returns:
        metrics: Dictionary of metric values
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]
    
    metrics = {}
    
    for k in k_values:
        top_k_preds = predictions[:, :k]
        labels_expanded = labels.reshape(-1, 1)
        correct = (top_k_preds == labels_expanded).any(axis=1)
        recall = correct.mean()
        metrics[f'recall@{k}'] = float(recall)
    
    mrr = compute_mrr(predictions, labels)
    metrics['mrr'] = float(mrr)
    
    ndcg = compute_ndcg(predictions, labels)
    metrics['ndcg'] = float(ndcg)
    
    return metrics


def compute_mrr(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute Mean Reciprocal Rank"""
    reciprocal_ranks = []
    
    for pred, label in zip(predictions, labels):
        rank = np.where(pred == label)[0]
        if len(rank) > 0:
            reciprocal_ranks.append(1.0 / (rank[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks)


def compute_ndcg(predictions: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """Compute Normalized Discounted Cumulative Gain"""
    ndcg_scores = []
    
    for pred, label in zip(predictions, labels):
        relevance = np.zeros(min(len(pred), k))
        
        label_positions = np.where(pred[:k] == label)[0]
        if len(label_positions) > 0:
            relevance[label_positions[0]] = 1.0
        
        dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
        
        ideal_relevance = np.array([1.0])
        idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)