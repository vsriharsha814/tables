"""Model manager for handling ML models."""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages ML models and caching."""
    
    _instance = None
    _sentence_transformer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_sentence_transformer(self, model_name='all-MiniLM-L6-v2'):
        """Get or load the sentence transformer model."""
        if self._sentence_transformer is None:
            logger.info(f"Loading sentence transformer: {model_name}")
            cache_dir = Path("models/cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            self._sentence_transformer = SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir)
            )
            logger.info("Model loaded successfully")
        
        return self._sentence_transformer
