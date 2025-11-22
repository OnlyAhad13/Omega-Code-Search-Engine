import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletLoss(nn.Module):
    """Triplet loss for contrastive learning"""
    
    def __init__(self, margin: float = 0.5, hard_negative_weight: float = 0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss
        
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negative: [batch_size, embedding_dim]
        
        Returns:
            loss: scalar
        """
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        neg_distance = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()
    
    def forward_with_hard_mining(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss with hard negative mining
        
        Args:
            anchor: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negatives: [batch_size, num_negatives, embedding_dim]
        
        Returns:
            loss: scalar
        """
        batch_size = anchor.size(0)
        
        pos_distance = F.pairwise_distance(anchor, positive, p=2)
        
        anchor_expanded = anchor.unsqueeze(1).expand_as(negatives)
        neg_distances = torch.norm(anchor_expanded - negatives, p=2, dim=-1)
        
        hardest_neg_distance, _ = torch.min(neg_distances, dim=1)
        
        hard_loss = F.relu(pos_distance - hardest_neg_distance + self.margin)
        
        mean_neg_distance = neg_distances.mean(dim=1)
        mean_loss = F.relu(pos_distance - mean_neg_distance + self.margin)
        
        total_loss = (
            self.hard_negative_weight * hard_loss +
            (1 - self.hard_negative_weight) * mean_loss
        )
        
        return total_loss.mean()


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature: float = 0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss
        
        Args:
            query: [batch_size, embedding_dim]
            positive: [batch_size, embedding_dim]
            negatives: [batch_size, num_negatives, embedding_dim] or None
        
        Returns:
            loss: scalar
        """
        query = F.normalize(query, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        pos_similarity = torch.sum(query * positive, dim=-1) / self.temperature
        
        if negatives is not None:
            negatives = F.normalize(negatives, p=2, dim=-1)
            query_expanded = query.unsqueeze(1)
            neg_similarity = torch.sum(
                query_expanded * negatives, dim=-1
            ) / self.temperature
            
            logits = torch.cat([pos_similarity.unsqueeze(1), neg_similarity], dim=1)
        else:
            similarity_matrix = torch.matmul(query, positive.T) / self.temperature
            logits = similarity_matrix
        
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined triplet and InfoNCE loss"""
    
    def __init__(
        self,
        triplet_margin: float = 0.5,
        temperature: float = 0.07,
        triplet_weight: float = 0.5,
        infonce_weight: float = 0.5
    ):
        super(CombinedLoss, self).__init__()
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.infonce_loss = InfoNCELoss(temperature=temperature)
        self.triplet_weight = triplet_weight
        self.infonce_weight = infonce_weight
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss"""
        triplet = self.triplet_loss(anchor, positive, negative)
        infonce = self.infonce_loss(anchor, positive)
        
        total_loss = (
            self.triplet_weight * triplet +
            self.infonce_weight * infonce
        )
        
        return {
            'total': total_loss,
            'triplet': triplet,
            'infonce': infonce
        }