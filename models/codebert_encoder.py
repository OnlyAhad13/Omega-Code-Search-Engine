import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeBERTEncoder(nn.Module):
    """CodeBERT encoder for semantic code representation"""
    
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        hidden_dim: int = 768,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super(CodeBERTEncoder, self).__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        
        logger.info(f"Loading CodeBERT model: {model_name}")
        self.bert = AutoModel.from_pretrained(model_name)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            logger.info("CodeBERT parameters frozen")
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(f"CodeBERT encoder initialized with hidden_dim={hidden_dim}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through CodeBERT
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            embeddings: [batch_size, hidden_dim]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        cls_embedding = self.dropout(cls_embedding)
        cls_embedding = self.layer_norm(cls_embedding)
        
        return cls_embedding
    
    def get_embeddings_with_pooling(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = 'cls'
    ) -> torch.Tensor:
        """Get embeddings with different pooling strategies"""
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        hidden_states = outputs.last_hidden_state       # (batch, seq_len, hidden)
        
        if pooling == 'cls':
            embeddings = hidden_states[:, 0, :]      
        elif pooling == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif pooling == 'max':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.clone()
            hidden_states[mask_expanded == 0] = -1e9  
            embeddings = torch.max(hidden_states, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        embeddings = self.dropout(embeddings)
        embeddings = self.layer_norm(embeddings)
        
        return embeddings
