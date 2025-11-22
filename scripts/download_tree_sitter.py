import subprocess
import sys
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data.ast_parser import ASTParser

def main():
    """Download and setup tree-sitter grammars"""
    logger.info("Setting up tree-sitter grammars...")
    
    try:
        parser = ASTParser(languages=['python', 'javascript', 'java'])
        logger.info("Tree-sitter grammars successfully set up!")
        
        for lang in ['python', 'javascript', 'java']:
            test_code = "def test(): pass" if lang == 'python' else "function test() {}"
            tree = parser.parse(test_code, lang)
            if tree:
                logger.info(f"✓ {lang} grammar working")
            else:
                logger.error(f"✗ {lang} grammar failed")
        
        logger.info("All grammars ready!")
        
    except Exception as e:
        logger.error(f"Error setting up tree-sitter: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()