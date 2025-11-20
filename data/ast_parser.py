import tree_sitter
from tree_sitter import Language, Parser
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASTParser:
    """Parse source code into Abstract Syntax Trees using tree-sitter"""
    LANGUAGE_REPOS = {
        'python': 'https://github.com/tree-sitter/tree-sitter-python',
        'javascript': 'https://github.com/tree-sitter/tree-sitter-javascript',
        'java': 'https://github.com/tree-sitter/tree-sitter-java',
    }

    def __init__(self, languages: List[str] = None):
        if languages is None:
            languages = ['python', 'javascript', 'java']
        
        self.languages = languages
        self.parsers = {}
        self.ts_languages = {}

        self._setup__tree_sitter()
        self._initialize_parsers()

    def _setup_tree_sitter(self):
        """Download and build tree-sitter languages"""
        ts_dir = Path.home() / '.tree-sitter'
        ts_dir.mkdir(exist_ok=True)

        lang_dir = ts_dir / 'languages'
        lang_dir.mkdir(exist_ok=True)

        so_file = ts_dir / 'languages.so'

        if so_file.exists():
            try:
                for lang in self.languages:
                    self.ts_languages[lang] = Language(str(so_file), lang)
                logger.info("Loaded existing tree-sitter languages")
                return
            except Exception as e:
                logger.warning(f"Failed to load existing tree-sitter languages: {e}")
            
        logger.info("Building tree-sitter languages")

        lang_paths = []
        for lang in self.languages:
            repo_path = lang_dir / f'tree-sitter-{lang}'
            if not repo_path.exists():
                logger.info(f"Cloning tree-sitter-{lang}...")
                subprocess.run([
                    'git', 'clone', '--depth', '1',
                    self.LANGUAGE_REPOS[lang],
                    str(repo_path)
                ], check=True, capture_output=True)
            lang_paths.append(str(repo_path))

        Language.build_library(str(so_file), lang_paths)
        
        for lang in self.languages:
            self.ts_languages[lang] = Language(str(so_file), lang)
        
        logger.info("Tree-sitter languages built successfully")

    def _initialize_parsers(self):
        """Initialize parsers for each language"""
        for lang in self.languages:
            parser = Parser()
            parser.set_language(self.ts_languages[lang])
            self.parsers[lang] = parser
    
    def parse(self, code: str, language: str) -> Optional[tree_sitter.Tree]:
        """Parse code and return AST"""
        if language not in self.parsers:
            raise ValueError(f"Unsupported language: {language}")
        

        try:
            code_bytes = bytes(code, 'utf-8')
            tree = self.parsers[language].parse(code_bytes)
            return tree
        except Exception as e:
            logger.error(f"Failed to parse code: {e}")
            return None
    
    def extract_nodes(self, tree: tree_sitter.Tree, code: str) -> List[Dict]:
        """Extract nodes from AST with their properties"""
        if tree is None or tree.root_node is None:
            return []
        
        nodes = []
        code_bytes = bytes(code, 'utf-8')

        def traverse(node, parent_id=None, depth=0):
            node_id = len(nodes)
            node_type = node.type

            start_byte = node.start_byte
            end_byte = node.end_byte
            node_text = code_bytes[start_byte:end_byte].decode('utf-8', errors='ignore')
            
            node_info = {
                'id': node_id,
                'type': node_type,
                'text': node_text[:100],
                'start_point': node.start_point,
                'end_point': node.end_point,
                'parent_id': parent_id,
                'depth': depth,
                'is_named': node.is_named,
                'child_count': node.child_count
            }

            nodes.append(node_info)
            
            for child in node.children:
                traverse(child, node_id, depth + 1)
        
        traverse(tree.root_node)
        return nodes
    
    def get_node_types(self, nodes: List[Dict]) -> List[str]:
        """Extract unique node types from parsed nodes"""
        return list(set(node['type'] for node in nodes))
    
    def build_parent_child_edges(self, nodes: List[Dict]) -> List[Tuple[int, int]]:
        """Build parent-child edges from nodes"""
        edges = []
        for node in nodes:
            if node['parent_id'] is not None:
                edges.append((node['parent_id'], node['id']))
        return edges
    
    


