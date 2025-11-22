import tree_sitter
from tree_sitter import Language, Parser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASTParser:
    """Parse source code into Abstract Syntax Trees using tree-sitter"""
    
    def __init__(self, languages: list[str] = None):
        if languages is None:
            languages = ['python', 'javascript', 'java']
        
        self.languages = languages
        self.parsers = {}
        self.ts_languages = {}

        self._setup_tree_sitter()
        self._initialize_parsers()

    def _setup_tree_sitter(self):
        """Setup tree-sitter languages using the modern Python bindings"""
        logger.info("Loading tree-sitter languages")
        
        try:
            import tree_sitter_python
            import tree_sitter_javascript
            import tree_sitter_java
            
            language_modules = {
                'python': tree_sitter_python,
                'javascript': tree_sitter_javascript,
                'java': tree_sitter_java
            }
            
            for lang in self.languages:
                if lang in language_modules:
                    lang_obj = language_modules[lang].language()
                    try:
                        # Try wrapping in Language class (standard for 0.22+)
                        self.ts_languages[lang] = Language(lang_obj)
                    except Exception:
                        # Fallback: use the object directly if it's already a Language instance
                        self.ts_languages[lang] = lang_obj
                        
                    logger.info(f"✓ Loaded {lang} grammar")
            
            return
            
        except ImportError as e:
            logger.error(f"Missing required packages: {e}")
            raise

    def _initialize_parsers(self):
        """Initialize parsers for each language"""
        for lang in self.languages:
            if lang in self.ts_languages:
                # NEW API CHANGE: Pass language directly to constructor
                # parser.set_language() was removed in v0.22.0
                self.parsers[lang] = Parser(self.ts_languages[lang])
                logger.debug(f"Initialized parser for {lang}")
    
    def parse(self, code: str, language: str):
        """Parse code and return AST"""
        if language not in self.parsers:
            logger.error(f"Unsupported language: {language}")
            return None
        
        try:
            code_bytes = bytes(code, 'utf-8')
            tree = self.parsers[language].parse(code_bytes)
            return tree
        except Exception as e:
            logger.error(f"Failed to parse code: {e}")
            return None
    
    def extract_nodes(self, tree, code: str) -> list[dict]:
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
    
    def get_node_types(self, nodes: list[dict]) -> list[str]:
        """Extract unique node types from parsed nodes"""
        return list(set(node['type'] for node in nodes))
    
    def build_parent_child_edges(self, nodes: list[dict]) -> list[tuple[int, int]]:
        """Build parent-child edges from nodes"""
        edges = []
        for node in nodes:
            if node['parent_id'] is not None:
                edges.append((node['parent_id'], node['id']))
        return edges