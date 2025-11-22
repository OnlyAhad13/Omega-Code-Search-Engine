import torch
import numpy as np
from typing import List, Tuple, Dict
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletSampler:
    """Sample triplets for contrastive learning"""
    
    def __init__(
        self,
        dataset,
        hard_negative_ratio: float = 0.5,
        num_negatives: int = 4
    ):
        self.dataset = dataset
        self.hard_negative_ratio = hard_negative_ratio
        self.num_negatives = num_negatives
        
        self.embeddings_cache = None
        self.indices_cache = None
    
    def update_embeddings(
        self,
        embeddings: torch.Tensor,
        indices: List[int]
    ):
        """Update cached embeddings for hard negative mining"""
        self.embeddings_cache = embeddings.cpu().detach()
        self.indices_cache = indices
    
    def sample_triplets(
        self,
        batch_indices: List[int]
    ) -> Tuple[List[int], List[int], List[List[int]]]:
        """
        Sample triplets for a batch
        
        Args:
            batch_indices: Indices of anchor samples
        
        Returns:
            anchors: List of anchor indices
            positives: List of positive indices (same as anchors)
            negatives: List of lists of negative indices
        """
        anchors = batch_indices
        positives = batch_indices
        
        negatives = []
        for anchor_idx in batch_indices:
            neg_samples = self._sample_negatives(anchor_idx)
            negatives.append(neg_samples)
        
        return anchors, positives, negatives
    
    def _sample_negatives(self, anchor_idx: int) -> List[int]:
        """Sample negative samples for an anchor"""
        num_hard = int(self.num_negatives * self.hard_negative_ratio)
        num_random = self.num_negatives - num_hard
        
        negatives = []
        
        if (self.embeddings_cache is not None and
            self.indices_cache is not None and
            num_hard > 0):
            hard_negs = self._sample_hard_negatives(anchor_idx, num_hard)
            negatives.extend(hard_negs)
        else:
            num_random = self.num_negatives
        
        random_negs = self._sample_random_negatives(anchor_idx, num_random)
        negatives.extend(random_negs)
        
        return negatives
    
    def _sample_hard_negatives(
        self,
        anchor_idx: int,
        num_samples: int
    ) -> List[int]:
        """Sample hard negatives using cached embeddings"""
        if anchor_idx not in self.indices_cache:
            return self._sample_random_negatives(anchor_idx, num_samples)
        
        cache_idx = self.indices_cache.index(anchor_idx)
        anchor_emb = self.embeddings_cache[cache_idx]
        
        distances = torch.norm(
            self.embeddings_cache - anchor_emb.unsqueeze(0),
            p=2,
            dim=-1
        )
        
        distances[cache_idx] = float('inf')
        
        _, sorted_indices = torch.sort(distances)
        
        hard_neg_indices = sorted_indices[:num_samples * 2].tolist()
        
        hard_negatives = [
            self.indices_cache[idx]
            for idx in hard_neg_indices
            if self.indices_cache[idx] != anchor_idx
        ][:num_samples]
        
        if len(hard_negatives) < num_samples:
            additional = self._sample_random_negatives(
                anchor_idx,
                num_samples - len(hard_negatives)
            )
            hard_negatives.extend(additional)
        
        return hard_negatives
    
    def _sample_random_negatives(
        self,
        anchor_idx: int,
        num_samples: int
    ) -> List[int]:
        """Sample random negative samples"""
        all_indices = list(range(len(self.dataset)))
        all_indices.remove(anchor_idx)
        
        if len(all_indices) < num_samples:
            return all_indices
        
        return random.sample(all_indices, num_samples)