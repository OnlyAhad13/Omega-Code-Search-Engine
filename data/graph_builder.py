import torch
from torch_geometric.data import Data
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    """Convert AST nodes to PyTorch Geometric graph structures"""
    
    def __init__(self, max_nodes: int = 500):
        self.max_nodes = max_nodes
        self.node_type_vocab = {}
        self.node_type_counter = 0
    
    def _get_node_type_id(self, node_type: str) -> int:
        """Get or create node type ID"""
        if node_type not in self.node_type_vocab:
            self.node_type_vocab[node_type] = self.node_type_counter
            self.node_type_counter += 1
        return self.node_type_vocab[node_type]
    
    def build_graph(self, nodes: List[Dict]) -> Data:
        """Build PyTorch Geometric graph from AST nodes"""
        if not nodes:
            return self._empty_graph()
        
        if len(nodes) > self.max_nodes:
            nodes = nodes[:self.max_nodes]
        
        num_nodes = len(nodes)
        
        node_features = []
        for node in nodes:
            node_type_id = self._get_node_type_id(node['type'])
            depth = node['depth']
            child_count = node['child_count']
            is_named = 1.0 if node['is_named'] else 0.0
            
            features = [
                float(node_type_id),
                float(depth),
                float(child_count),
                is_named
            ]
            node_features.append(features)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        edge_index = self._build_edge_index(nodes, num_nodes)
        
        graph = Data(x=x, edge_index=edge_index)
        graph.num_nodes = num_nodes
        
        return graph
    
    def _build_edge_index(self, nodes: List[Dict], num_nodes: int) -> torch.Tensor:
        """Build edge index tensor with multiple edge types"""
        parent_child_edges = []
        
        for i, node in enumerate(nodes):
            if node['parent_id'] is not None and node['parent_id'] < num_nodes:
                parent_child_edges.append([node['parent_id'], i])
                parent_child_edges.append([i, node['parent_id']])
        
        sibling_edges = self._build_sibling_edges(nodes, num_nodes)
        control_flow_edges = self._build_control_flow_edges(nodes, num_nodes)
        
        all_edges = parent_child_edges + sibling_edges + control_flow_edges
        
        if not all_edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def _build_sibling_edges(self, nodes: List[Dict], num_nodes: int) -> List[List[int]]:
        """Build edges between sibling nodes"""
        parent_to_children = defaultdict(list)
        for i, node in enumerate(nodes):
            if node['parent_id'] is not None and node['parent_id'] < num_nodes:
                parent_to_children[node['parent_id']].append(i)
        
        sibling_edges = []
        for children in parent_to_children.values():
            for i in range(len(children) - 1):
                sibling_edges.append([children[i], children[i + 1]])
                sibling_edges.append([children[i + 1], children[i]])
        
        return sibling_edges
    
    def _build_control_flow_edges(self, nodes: List[Dict], num_nodes: int) -> List[List[int]]:
        """Build control flow edges for control structures"""
        control_flow_types = {
            'if_statement', 'while_statement', 'for_statement',
            'try_statement', 'switch_statement', 'function_definition',
            'method_declaration', 'class_declaration'
        }
        
        edges = []
        control_nodes = [
            i for i, node in enumerate(nodes)
            if node['type'] in control_flow_types
        ]
        
        for i in range(len(control_nodes) - 1):
            edges.append([control_nodes[i], control_nodes[i + 1]])
        
        return edges
    
    def _empty_graph(self) -> Data:
        """Create an empty graph for error cases"""
        x = torch.zeros((1, 4), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, num_nodes=1)
    
    def get_vocab_size(self) -> int:
        """Get the size of node type vocabulary"""
        return self.node_type_counter