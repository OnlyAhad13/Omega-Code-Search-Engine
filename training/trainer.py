import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
# Kept the fix: Import this from transformers, not torch
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, Optional
import json

from models.fusion_model import FusionModel
from models.losses import TripletLoss, InfoNCELoss, CombinedLoss
from data.dataset import collate_fn
from .evaluator import Evaluator
from .triplet_sampler import TripletSampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for code search model"""
    
    def __init__(
        self,
        model: FusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        learning_rate = float(config['training']['learning_rate'])
        weight_decay = float(config['training']['weight_decay'])
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        num_epochs = int(config['training']['num_epochs'])
        total_steps = len(train_loader) * num_epochs
        
        warmup_steps = int(config['training']['warmup_steps'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        triplet_margin = float(config['training']['triplet_margin'])
        hard_negative_weight = float(config['training']['hard_negative_weight'])
        
        self.criterion = TripletLoss(
            margin=triplet_margin,
            hard_negative_weight=hard_negative_weight
        )
        
        self.evaluator = Evaluator(
            model=self.model,
            device=device,
            recall_k=config['evaluation']['recall_k']
        )
        
        self.triplet_sampler = TripletSampler(
            dataset=train_loader.dataset
        )
        
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        self.save_dir = Path(config['paths']['model_save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def train(self, num_epochs: int):
        """Train the model"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            val_metrics = self.validate()
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val Metrics: {val_metrics}")
            
            self.save_checkpoint(epoch, val_metrics)
            
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info("Saved best model")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        grad_accum_steps = int(self.config['training']['gradient_accumulation_steps'])
        
        for batch_idx, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            
            total_loss += loss
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
            
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_step(self, batch: Dict) -> float:
        """Single training step"""
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
        
        batch_size = code_embeddings.size(0)
        negative_indices = torch.roll(torch.arange(batch_size), shifts=1)
        negative_embeddings = code_embeddings[negative_indices]
        
        loss = self.criterion(
            query_embeddings,
            code_embeddings,
            negative_embeddings
        )
        
        grad_accum_steps = int(self.config['training']['gradient_accumulation_steps'])
        max_grad_norm = float(self.config['training']['max_grad_norm'])
        
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (self.global_step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item() * grad_accum_steps
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_query_embeddings = []
        all_code_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
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
                
                batch_size = code_embeddings.size(0)
                negative_indices = torch.roll(torch.arange(batch_size), shifts=1)
                negative_embeddings = code_embeddings[negative_indices]
                
                loss = self.criterion(
                    query_embeddings,
                    code_embeddings,
                    negative_embeddings
                )
                
                total_loss += loss.item()
                num_batches += 1
                
                all_query_embeddings.append(query_embeddings.cpu())
                all_code_embeddings.append(code_embeddings.cpu())
        
        avg_loss = total_loss / num_batches
        
        if all_query_embeddings:
            all_query_embeddings = torch.cat(all_query_embeddings, dim=0)
            all_code_embeddings = torch.cat(all_code_embeddings, dim=0)
            
            recall_metrics = self.evaluator.compute_recall(
                all_query_embeddings,
                all_code_embeddings
            )
        else:
            recall_metrics = {'recall@1': 0.0, 'mrr': 0.0, 'ndcg': 0.0}
        
        metrics = {
            'loss': avg_loss,
            **recall_metrics
        }
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        
        metrics_path = self.save_dir / f'metrics_epoch_{epoch}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['metrics']