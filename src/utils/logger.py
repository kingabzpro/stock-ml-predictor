import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    def __init__(self, name: str, config: Optional[dict] = None):
        self.name = name
        self.config = config or {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'logs/app.log'
        }
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.name)
        logger.setLevel(getattr(logging, self.config['level']))
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config['level']))
        
        # File handler
        log_file = self.config.get('file', 'logs/app.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, self.config['level']))
        
        # Formatter
        formatter = logging.Formatter(self.config['format'])
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def exception(self, message: str):
        self.logger.exception(message)

def get_logger(name: str, config: Optional[dict] = None) -> Logger:
    return Logger(name, config)