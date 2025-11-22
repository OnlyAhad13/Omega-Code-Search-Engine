import torch
from typing import List, Dict, Tuple
import logging
from pathlib import Path

from models.fusion_model import FusionModel
from data.ast_parser import ASTParser
from data.graph_builder import GraphBuilder
from data.preprocessing import CodePreprocessor
from .vector_store import VectorStore
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchEngine:
    """End-to-end code search engine"""
    
    def __init__(
        self,
        model: FusionModel,
        vector_store: VectorStore,
        tokenizer_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        max_nodes: int = 500,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vector_store = vector_store
        self.device = device
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.ast_parser = ASTParser()
        self.graph_builder = GraphBuilder(max_nodes=max_nodes)
        self.preprocessor = CodePreprocessor()
        
        logger.info("Search engine initialized")
    
    def index_code(
        self,
        code_snippets: List[Dict],
        batch_size: int = 32
    ):
        """
        Index code snippets into vector store
        
        Args:
            code_snippets: List of dicts with 'code', 'language', and other metadata
            batch_size: Batch size for encoding
        """
        logger.info(f"Indexing {len(code_snippets)} code snippets...")
        
        all_embeddings = []
        all_metadata = []
        
        for i in range(0, len(code_snippets), batch_size):
            batch = code_snippets[i:i + batch_size]
            
            embeddings = self._encode_code_batch(batch)
            all_embeddings.append(embeddings.cpu().numpy())
            all_metadata.extend(batch)
        
        import numpy as np
        all_embeddings = np.vstack(all_embeddings)
        
        self.vector_store.add_embeddings(all_embeddings, all_metadata)
        
        logger.info(f"Indexed {len(code_snippets)} code snippets")
    
    def _encode_code_batch(self, batch: List[Dict]) -> torch.Tensor:
        """Encode a batch of code snippets"""
        input_ids_list = []
        attention_masks_list = []
        graphs_list = []
        
        for item in batch:
            code = item['code']
            language = item.get('language', 'python')
            
            preprocessed_code = self.preprocessor.preprocess(
                code, language, remove_comments=True, normalize_whitespace=True
            )
            
            tokens = self.tokenizer(
                preprocessed_code,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids_list.append(tokens['input_ids'])
            attention_masks_list.append(tokens['attention_mask'])
            
            tree = self.ast_parser.parse(code, language)
            nodes = self.ast_parser.extract_nodes(tree, code) if tree else []
            graph = self.graph_builder.build_graph(nodes)
            graphs_list.append(graph)
        
        input_ids = torch.cat(input_ids_list, dim=0).to(self.device)
        attention_masks = torch.cat(attention_masks_list, dim=0).to(self.device)
        
        from torch_geometric.data import Batch
        batched_graphs = Batch.from_data_list(graphs_list).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_code(
                input_ids,
                attention_masks,
                batched_graphs
            )
        
        return embeddings
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict]:
        """
        Search for code snippets matching the query
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            return_scores: Whether to include similarity scores
        
        Returns:
            results: List of matching code snippets with metadata
        """
        query_embedding = self._encode_query(query)
        
        distances, indices, metadata = self.vector_store.search(
            query_embedding.cpu().numpy(),
            top_k=top_k
        )
        
        results = []
        for i, (distance, metadata_dict) in enumerate(zip(distances[0], metadata[0])):
            result = {
                'rank': i + 1,
                'code': metadata_dict.get('code', ''),
                'language': metadata_dict.get('language', 'unknown'),
                'func_name': metadata_dict.get('func_name', ''),
                'docstring': metadata_dict.get('docstring', '')
            }
            
            if return_scores:
                result['score'] = float(distance)
            
            results.append(result)
        
        return results
    
    def _encode_query(self, query: str) -> torch.Tensor:
        """Encode a natural language query"""
        tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_query(input_ids, attention_mask)
        
        return embedding
    
    def save_index(self, save_path: str):
        """Save vector store index"""
        self.vector_store.save(save_path)
    
    def load_index(self, load_path: str):
        """Load vector store index"""
        self.vector_store.load(load_path)