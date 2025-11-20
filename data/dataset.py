import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import json
import random
import logging
from pathlib import Path
import numpy as np

from .ast_parser import ASTParser
from .graph_builder import GraphBuilder
from .preprocessing import CodePreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeSearchDataset(Dataset):
    """Dataset for code search training with CodeBERT + GNN"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        max_nodes: int = 500,
        languages: List[str] = None,
        mode: str = 'train'
    ):
        self.data_path = Path(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.mode = mode
        
        if languages is None:
            languages = ['python', 'javascript', 'java']
        
        self.ast_parser = ASTParser(languages=languages)
        self.graph_builder = GraphBuilder(max_nodes=max_nodes)
        self.preprocessor = CodePreprocessor()
        
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} code samples for {mode} mode")
    
    def _load_data(self) -> List[Dict]:
        """Load dataset from JSON file or directory"""
        data = []
        
        if self.data_path.is_file():
            with open(self.data_path, 'r') as f:
                data = json.load(f)
        elif self.data_path.is_dir():
            for json_file in self.data_path.glob('*.json'):
                with open(json_file, 'r') as f:
                    data.extend(json.load(f))
        else:
            logger.warning(f"Data path {self.data_path} not found. Creating sample data.")
            data = self._create_sample_data()
        
        return data
    
    def _create_sample_data(self) -> List[Dict]:
        """Create sample data for demonstration"""
        sample_data = [
            {
                'code': 'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
                'docstring': 'Binary search algorithm to find target in sorted array',
                'language': 'python',
                'func_name': 'binary_search'
            },
            {
                'code': 'function quickSort(arr) {\n    if (arr.length <= 1) return arr;\n    const pivot = arr[0];\n    const left = arr.slice(1).filter(x => x < pivot);\n    const right = arr.slice(1).filter(x => x >= pivot);\n    return [...quickSort(left), pivot, ...quickSort(right)];\n}',
                'docstring': 'Quick sort algorithm implementation',
                'language': 'javascript',
                'func_name': 'quickSort'
            },
            {
                'code': 'public class BubbleSort {\n    public static void sort(int[] arr) {\n        int n = arr.length;\n        for (int i = 0; i < n-1; i++) {\n            for (int j = 0; j < n-i-1; j++) {\n                if (arr[j] > arr[j+1]) {\n                    int temp = arr[j];\n                    arr[j] = arr[j+1];\n                    arr[j+1] = temp;\n                }\n            }\n        }\n    }\n}',
                'docstring': 'Bubble sort algorithm for sorting arrays',
                'language': 'java',
                'func_name': 'sort'
            }
        ]
        
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = self.data_path if self.data_path.is_file() else self.data_path / 'sample_data.json'
        
        with open(output_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        return sample_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single item with code embeddings and graph"""
        item = self.data[idx]
        
        code = item['code']
        language = item.get('language', 'python')
        docstring = item.get('docstring', '')
        
        preprocessed_code = self.preprocessor.preprocess(
            code, language, remove_comments=True, normalize_whitespace=True
        )
        
        code_tokens = self.tokenizer(
            preprocessed_code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        query_tokens = self.tokenizer(
            docstring,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tree = self.ast_parser.parse(code, language)
        nodes = self.ast_parser.extract_nodes(tree, code) if tree else []
        graph = self.graph_builder.build_graph(nodes)
        
        return {
            'code_input_ids': code_tokens['input_ids'].squeeze(0),
            'code_attention_mask': code_tokens['attention_mask'].squeeze(0),
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'graph': graph,
            'language': language,
            'idx': idx
        }
    
    def get_triplet(self, idx: int) -> Tuple[Dict, Dict, Dict]:
        """Get triplet: anchor, positive, negative"""
        anchor = self.__getitem__(idx)
        positive = anchor
        
        neg_idx = random.choice([i for i in range(len(self.data)) if i != idx])
        negative = self.__getitem__(neg_idx)
        
        return anchor, positive, negative


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching graphs"""
    code_input_ids = torch.stack([item['code_input_ids'] for item in batch])
    code_attention_mask = torch.stack([item['code_attention_mask'] for item in batch])
    query_input_ids = torch.stack([item['query_input_ids'] for item in batch])
    query_attention_mask = torch.stack([item['query_attention_mask'] for item in batch])
    
    graphs = [item['graph'] for item in batch]
    batched_graphs = Batch.from_data_list(graphs)
    
    return {
        'code_input_ids': code_input_ids,
        'code_attention_mask': code_attention_mask,
        'query_input_ids': query_input_ids,
        'query_attention_mask': query_attention_mask,
        'graph': batched_graphs,
        'indices': [item['idx'] for item in batch]
    }