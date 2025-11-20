import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for key in ['data_dir', 'model_save_dir', 'index_save_dir', 'logs_dir']:
            path = self.config['paths'][key]
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, *keys):
        """Get nested config value"""
        value = self.config
        for key in keys:
            value = value[key]
        return value
    
    def __getitem__(self, key):
        return self.config[key]
    
    def to_dict(self) -> Dict[str, Any]:
        return self.config

config = Config()