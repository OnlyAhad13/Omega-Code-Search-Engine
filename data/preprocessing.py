import re
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodePreprocessor:
    """Preprocess source code for better embedding quality"""
    
    def __init__(self):
        self.comment_patterns = {
            'python': [
                (r'#.*?$', ''),
                (r'\"\"\".*?\"\"\"', '', re.DOTALL),
                (r"'''.*?'''", '', re.DOTALL),
            ],
            'javascript': [
                (r'//.*?$', ''),
                (r'/\*.*?\*/', '', re.DOTALL),
            ],
            'java': [
                (r'//.*?$', ''),
                (r'/\*.*?\*/', '', re.DOTALL),
            ]
        }
    
    def remove_comments(self, code: str, language: str) -> str:
        """Remove comments from code"""
        if language not in self.comment_patterns:
            return code
        
        for pattern, replacement, *flags in self.comment_patterns[language]:
            flag = flags[0] if flags else 0
            code = re.sub(pattern, replacement, code, flags=re.MULTILINE | flag)
        
        return code
    
    def normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code"""
        code = re.sub(r'[ \t]+', ' ', code)
        code = re.sub(r'\n\s*\n', '\n\n', code)
        code = code.strip()
        return code
    
    def remove_empty_lines(self, code: str) -> str:
        """Remove empty lines"""
        lines = [line for line in code.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def normalize_strings(self, code: str) -> str:
        """Normalize string literals"""
        code = re.sub(r'"[^"]*"', '"STRING"', code)
        code = re.sub(r"'[^']*'", "'STRING'", code)
        return code
    
    def normalize_numbers(self, code: str) -> str:
        """Normalize numeric literals"""
        code = re.sub(r'\b\d+\.?\d*\b', 'NUM', code)
        return code
    
    def preprocess(self, code: str, language: str,
                   remove_comments: bool = True,
                   normalize_whitespace: bool = True,
                   normalize_strings: bool = False,
                   normalize_numbers: bool = False) -> str:
        """Apply all preprocessing steps"""
        if remove_comments:
            code = self.remove_comments(code, language)
        
        if normalize_whitespace:
            code = self.normalize_whitespace(code)
        
        if normalize_strings:
            code = self.normalize_strings(code)
        
        if normalize_numbers:
            code = self.normalize_numbers(code)
        
        return code
    
    def extract_function_signature(self, code: str, language: str) -> str:
        """Extract function signature from code"""
        patterns = {
            'python': r'def\s+(\w+)\s*\([^)]*\)',
            'javascript': r'function\s+(\w+)\s*\([^)]*\)',
            'java': r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\([^)]*\)'
        }
        
        if language in patterns:
            match = re.search(patterns[language], code)
            if match:
                return match.group(0)
        
        return ''