import yaml
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        return self.config.get('model', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        return self.config.get('api', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        return self.config.get('logging', {})