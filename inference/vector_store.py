import faiss
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for efficient similarity search"""
    
    def __init__(
        self,
        embedding_dim: int = 768,
        index_type: str = 'Flat',
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        
        self.index = None
        self.metadata = []
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        elif self.index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
        
        elif self.index_type == 'HNSW':
            self.index = faiss.IndexHNSWFlat(
                self.embedding_dim,
                32,
                faiss.METRIC_INNER_PRODUCT
            )
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
            logger.info("Using GPU for FAISS index")
        
        logger.info(f"Initialized {self.index_type} index with dim={self.embedding_dim}")
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict]
    ):
        """
        Add embeddings to the index
        
        Args:
            embeddings: [num_vectors, embedding_dim]
            metadata: List of metadata dictionaries
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        embeddings = embeddings.astype('float32')
        
        faiss.normalize_L2(embeddings)
        
        if self.index_type == 'IVF' and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("IVF index trained")
        
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(embeddings)} embeddings to index. Total: {self.index.ntotal}")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Search for similar vectors
        
        Args:
            query_embeddings: [num_queries, embedding_dim]
            top_k: Number of results to return
        
        Returns:
            distances: [num_queries, top_k]
            indices: [num_queries, top_k]
            metadata: List of lists of metadata dictionaries
        """
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy()
        
        query_embeddings = query_embeddings.astype('float32')
        
        faiss.normalize_L2(query_embeddings)
        
        if self.index_type == 'IVF':
            self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(query_embeddings, top_k)
        
        results_metadata = []
        for query_indices in indices:
            query_metadata = [
                self.metadata[idx] if idx < len(self.metadata) else {}
                for idx in query_indices
            ]
            results_metadata.append(query_metadata)
        
        return distances, indices, results_metadata
    
    def save(self, save_path: str):
        """Save index and metadata to disk"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        index_path = save_path / 'faiss_index.bin'
        
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        metadata_path = save_path / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        config_path = save_path / 'config.pkl'
        config = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'nprobe': self.nprobe
        }
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved index to {save_path}")
    
    def load(self, load_path: str):
        """Load index and metadata from disk"""
        load_path = Path(load_path)
        
        config_path = load_path / 'config.pkl'
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        self.embedding_dim = config['embedding_dim']
        self.index_type = config['index_type']
        self.nlist = config['nlist']
        self.nprobe = config['nprobe']
        
        index_path = load_path / 'faiss_index.bin'
        self.index = faiss.read_index(str(index_path))
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        metadata_path = load_path / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded index from {load_path}. Total vectors: {self.index.ntotal}")
    
    def clear(self):
        """Clear the index and metadata"""
        self._initialize_index()
        self.metadata = []
        logger.info("Cleared index and metadata")