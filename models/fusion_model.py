import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, Tuple, Optional
import logging

from .codebert_encoder import CodeBERTEncoder
from .gnn_encoder import GNNEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FusionModel(nn.Module):
    """Fusion model combining CodeBERT and GNN encoders"""
    
    def __init__(
        self,
        codebert_model: str = "microsoft/codebert-base",
        gnn_hidden_dim: int = 256,
        gnn_num_layers: int = 3,
        gnn_type: str = 'GAT',
        gnn_heads: int = 4,
        dropout: float = 0.1,
        fusion_method: str = 'concat',
        final_embedding_dim: int = 768,
        node_vocab_size: int = 1000
    ):
        super(FusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        self.final_embedding_dim = final_embedding_dim
        
        self.codebert_encoder = CodeBERTEncoder(
            model_name=codebert_model,
            hidden_dim=768,
            dropout=dropout
        )
        
        self.gnn_encoder = GNNEncoder(
            input_dim=4,
            hidden_dim=gnn_hidden_dim,
            output_dim=768,
            num_layers=gnn_num_layers,
            gnn_type=gnn_type,
            heads=gnn_heads,
            dropout=dropout,
            node_vocab_size=node_vocab_size
        )
        
        if fusion_method == 'concat':
            self.fusion_layer = nn.Sequential(
                nn.Linear(768 + 768, final_embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(final_embedding_dim)
            )
        elif fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=768,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(768, final_embedding_dim),
                nn.LayerNorm(final_embedding_dim)
            )
        elif fusion_method == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(768 + 768, 1),
                nn.Sigmoid()
            )
            self.fusion_layer = nn.Sequential(
                nn.Linear(768, final_embedding_dim),
                nn.LayerNorm(final_embedding_dim)
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        logger.info(f"Fusion model initialized with method: {fusion_method}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph: Batch
    ) -> torch.Tensor:
        """
        Forward pass through fusion model
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            graph: PyTorch Geometric Batch
        
        Returns:
            fused_embedding: [batch_size, final_embedding_dim]
        """
        codebert_embedding = self.codebert_encoder(input_ids, attention_mask)
        
        gnn_embedding = self.gnn_encoder(graph)
        
        fused_embedding = self._fuse_embeddings(codebert_embedding, gnn_embedding)
        
        return fused_embedding
    
    def _fuse_embeddings(
        self,
        codebert_emb: torch.Tensor,
        gnn_emb: torch.Tensor
    ) -> torch.Tensor:
        """Fuse embeddings using specified method"""
        if self.fusion_method == 'concat':
            combined = torch.cat([codebert_emb, gnn_emb], dim=-1)
            fused = self.fusion_layer(combined)
        
        elif self.fusion_method == 'attention':
            combined = torch.stack([codebert_emb, gnn_emb], dim=1)
            attn_out, _ = self.attention(combined, combined, combined)
            fused = self.fusion_layer(attn_out.mean(dim=1))
        
        elif self.fusion_method == 'gated':
            combined = torch.cat([codebert_emb, gnn_emb], dim=-1)
            gate = self.gate(combined)
            weighted = gate * codebert_emb + (1 - gate) * gnn_emb
            fused = self.fusion_layer(weighted)
        
        return fused
    
    def encode_code(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        graph: Batch
    ) -> torch.Tensor:
        """Encode code snippets"""
        return self.forward(input_ids, attention_mask, graph)
    
    def encode_query(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode natural language queries (CodeBERT only)"""
        return self.codebert_encoder(input_ids, attention_mask)