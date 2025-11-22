import argparse
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from data.dataset import CodeSearchDataset, collate_fn
from models.fusion_model import FusionModel
from inference.vector_store import VectorStore
from inference.search_engine import SearchEngine
from utils.logger import setup_logger

logger = setup_logger('build_index')


def main():
    parser = argparse.ArgumentParser(description='Build FAISS index')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data/datasets/train.json',
                        help='Path to data for indexing')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for index')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    logger.info("Loading model...")
    model = FusionModel(
        codebert_model=config.get('model', 'codebert_model'),
        gnn_hidden_dim=config.get('model', 'gnn_hidden_dim'),
        gnn_num_layers=config.get('model', 'gnn_num_layers'),
        gnn_type=config.get('model', 'gnn_type'),
        gnn_heads=config.get('model', 'gnn_heads'),
        dropout=config.get('model', 'dropout'),
        fusion_method=config.get('model', 'fusion_method'),
        final_embedding_dim=config.get('model', 'final_embedding_dim')
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from {args.checkpoint}")
    
    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        embedding_dim=config.get('model', 'final_embedding_dim'),
        index_type=config.get('inference', 'faiss_index_type'),
        nlist=config.get('inference', 'faiss_nlist'),
        nprobe=config.get('inference', 'faiss_nprobe'),
        use_gpu=device == 'cuda'
    )
    
    logger.info("Initializing search engine...")
    search_engine = SearchEngine(
        model=model,
        vector_store=vector_store,
        device=device
    )
    
    logger.info("Loading dataset...")
    dataset = CodeSearchDataset(
        data_path=args.data,
        tokenizer_name=config.get('model', 'codebert_model'),
        max_length=config.get('data', 'max_code_length'),
        max_nodes=config.get('data', 'max_ast_nodes'),
        languages=config.get('data', 'languages'),
        mode='train'
    )
    
    code_snippets = []
    for i in range(len(dataset)):
        item = dataset.data[i]
        code_snippets.append(item)
    
    logger.info(f"Indexing {len(code_snippets)} code snippets...")
    search_engine.index_code(code_snippets, batch_size=args.batch_size)
    
    output_dir = args.output or config.get('paths', 'index_save_dir')
    output_path = Path(output_dir)
    
    logger.info(f"Saving index to {output_path}...")
    search_engine.save_index(str(output_path))
    
    logger.info("Index building complete!")
    logger.info(f"Total vectors in index: {vector_store.index.ntotal}")


if __name__ == "__main__":
    main()