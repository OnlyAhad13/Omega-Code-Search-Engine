import argparse
import sys
import torch
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import config
from models.fusion_model import FusionModel
from inference.vector_store import VectorStore
from inference.search_engine import SearchEngine
from utils.logger import setup_logger

logger = setup_logger('test_search')


def main():
    parser = argparse.ArgumentParser(description='Test search engine')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--index', type=str, required=True,
                        help='Path to FAISS index')
    parser.add_argument('--query', type=str, default='sort an array',
                        help='Search query')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of results')
    
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
    
    logger.info("Loading vector store...")
    vector_store = VectorStore(
        embedding_dim=config.get('model', 'final_embedding_dim'),
        index_type=config.get('inference', 'faiss_index_type'),
        use_gpu=device == 'cuda'
    )
    vector_store.load(args.index)
    
    logger.info("Initializing search engine...")
    search_engine = SearchEngine(
        model=model,
        vector_store=vector_store,
        device=device
    )
    
    logger.info(f"\nSearching for: '{args.query}'")
    results = search_engine.search(args.query, top_k=args.top_k)
    
    print("\n" + "="*80)
    print(f"Search Results for: '{args.query}'")
    print("="*80)
    
    for result in results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f})")
        print(f"Language: {result['language']}")
        print(f"Function: {result['func_name']}")
        print(f"Description: {result['docstring']}")
        print(f"Code:\n{result['code']}")
        print("-"*80)


if __name__ == "__main__":
    main()