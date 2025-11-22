import argparse
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from data.dataset import CodeSearchDataset, collate_fn
from models.fusion_model import FusionModel
from training.trainer import Trainer
from utils.logger import setup_logger

logger = setup_logger('train')


def main():
    parser = argparse.ArgumentParser(description='Train code search model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--train_data', type=str, default='./data/datasets/train.json',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='./data/datasets/val.json',
                        help='Path to validation data')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.config:
        from config import Config
        cfg = Config(args.config)
    else:
        cfg = config
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    logger.info("Loading datasets...")
    train_dataset = CodeSearchDataset(
        data_path=args.train_data,
        tokenizer_name=cfg.get('model', 'codebert_model'),
        max_length=cfg.get('data', 'max_code_length'),
        max_nodes=cfg.get('data', 'max_ast_nodes'),
        languages=cfg.get('data', 'languages'),
        mode='train'
    )
    
    val_dataset = CodeSearchDataset(
        data_path=args.val_data,
        tokenizer_name=cfg.get('model', 'codebert_model'),
        max_length=cfg.get('data', 'max_code_length'),
        max_nodes=cfg.get('data', 'max_ast_nodes'),
        languages=cfg.get('data', 'languages'),
        mode='val'
    )
    
    batch_size = args.batch_size or cfg.get('training', 'batch_size')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get('data', 'num_workers'),
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get('evaluation', 'batch_size'),
        shuffle=False,
        num_workers=cfg.get('data', 'num_workers'),
        collate_fn=collate_fn
    )
    
    logger.info("Initializing model...")
    model = FusionModel(
        codebert_model=cfg.get('model', 'codebert_model'),
        gnn_hidden_dim=cfg.get('model', 'gnn_hidden_dim'),
        gnn_num_layers=cfg.get('model', 'gnn_num_layers'),
        gnn_type=cfg.get('model', 'gnn_type'),
        gnn_heads=cfg.get('model', 'gnn_heads'),
        dropout=cfg.get('model', 'dropout'),
        fusion_method=cfg.get('model', 'fusion_method'),
        final_embedding_dim=cfg.get('model', 'final_embedding_dim')
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg.to_dict(),
        device=device
    )
    
    num_epochs = args.epochs or cfg.get('training', 'num_epochs')
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    trainer.train(num_epochs)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()