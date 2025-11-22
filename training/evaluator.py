import torch
import torch.nn.functional as F
from typing import Dict, List
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for code search model"""
    def __init__(
        self,
        model,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        recall_k: List[int] = None
    ):
        self.model = model
        self.device = device
        self.recall_k = recall_k if recall_k else [1, 5, 10, 20]
    
    def compute_recall(
        self,
        query_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Recall@K metrics
        
        Args:
            query_embeddings: [num_queries, embedding_dim]
            code_embeddings: [num_codes, embedding_dim]
        
        Returns:
            metrics: Dictionary of recall scores
        """
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=-1)
        
        similarity_matrix = torch.matmul(query_embeddings, code_embeddings.T)
        
        _, indices = torch.topk(similarity_matrix, k=max(self.recall_k), dim=-1)
        
        num_queries = query_embeddings.size(0)
        correct_indices = torch.arange(num_queries).unsqueeze(-1)
        
        metrics = {}
        for k in self.recall_k:
            top_k_indices = indices[:, :k]
            correct_in_top_k = (top_k_indices == correct_indices).any(dim=-1)
            recall = correct_in_top_k.float().mean().item()
            metrics[f'recall@{k}'] = recall
        
        mrr = self.compute_mrr(similarity_matrix, correct_indices)
        metrics['mrr'] = mrr
        
        ndcg = self.compute_ndcg(similarity_matrix, correct_indices)
        metrics['ndcg'] = ndcg
        
        return metrics
    
    def compute_mrr(
        self,
        similarity_matrix: torch.Tensor,
        correct_indices: torch.Tensor
    ) -> float:
        """Compute Mean Reciprocal Rank"""
        _, sorted_indices = torch.sort(similarity_matrix, dim=-1, descending=True)
        
        ranks = []
        for i, correct_idx in enumerate(correct_indices.squeeze()):
            rank = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(1.0 / rank)
        
        mrr = np.mean(ranks)
        return mrr
    
    def compute_ndcg(
        self,
        similarity_matrix: torch.Tensor,
        correct_indices: torch.Tensor,
        k: int = 10
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain"""
        _, sorted_indices = torch.sort(similarity_matrix, dim=-1, descending=True)
        
        ndcg_scores = []
        for i, correct_idx in enumerate(correct_indices.squeeze()):
            relevance = torch.zeros(sorted_indices.size(1))
            correct_position = (sorted_indices[i] == correct_idx).nonzero(as_tuple=True)[0]
            
            if len(correct_position) > 0 and correct_position[0] < k:
                relevance[correct_position[0]] = 1.0
            
            dcg = self._compute_dcg(relevance[:k])
            idcg = self._compute_dcg(torch.ones(1))
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return np.mean(ndcg_scores)
    
    def _compute_dcg(self, relevance: torch.Tensor) -> float:
        """Compute Discounted Cumulative Gain"""
        gains = relevance / torch.log2(torch.arange(2, len(relevance) + 2).float())
        return gains.sum().item()
    
    def evaluate_batch(
        self,
        batch: Dict,
        return_embeddings: bool = False
    ) -> Dict:
        """Evaluate a single batch"""
        self.model.eval()
        
        with torch.no_grad():
            code_input_ids = batch['code_input_ids'].to(self.device)
            code_attention_mask = batch['code_attention_mask'].to(self.device)
            query_input_ids = batch['query_input_ids'].to(self.device)
            query_attention_mask = batch['query_attention_mask'].to(self.device)
            graph = batch['graph'].to(self.device)
            
            code_embeddings = self.model.encode_code(
                code_input_ids,
                code_attention_mask,
                graph
            )
            
            query_embeddings = self.model.encode_query(
                query_input_ids,
                query_attention_mask
            )
            
            metrics = self.compute_recall(query_embeddings, code_embeddings)
            
            if return_embeddings:
                metrics['code_embeddings'] = code_embeddings.cpu()
                metrics['query_embeddings'] = query_embeddings.cpu()
        
        return metrics