"""Embedding utilities and caching."""

import numpy as np
from typing import List, Dict, Tuple
import hashlib
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Cache for sentence embeddings to improve performance."""
    
    def __init__(self, cache_dir: str = "models/embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}  # In-memory cache for current session
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, texts: List[str]) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """Get cached embeddings and indices of missing texts."""
        cached = {}
        missing_indices = []
        
        for i, text in enumerate(texts):
            key = self._get_cache_key(text)
            
            # Check memory cache first
            if key in self.memory_cache:
                cached[i] = self.memory_cache[key]
                continue
            
            # Check disk cache
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    embedding = np.load(cache_file)
                    self.memory_cache[key] = embedding
                    cached[i] = embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {e}")
                    missing_indices.append(i)
            else:
                missing_indices.append(i)
        
        return cached, missing_indices
    
    def save(self, texts: List[str], embeddings: np.ndarray):
        """Save embeddings to cache."""
        for text, embedding in zip(texts, embeddings):
            key = self._get_cache_key(text)
            
            # Save to memory cache
            self.memory_cache[key] = embedding
            
            # Save to disk cache
            try:
                cache_file = self.cache_dir / f"{key}.npy"
                np.save(cache_file, embedding)
            except Exception as e:
                logger.warning(f"Failed to save embedding to cache: {e}")


class EmbeddingProcessor:
    """Process and manage embeddings."""
    
    def __init__(self, model):
        self.model = model
        self.cache = EmbeddingCache()
    
    def encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """Encode texts with caching."""
        # Get cached embeddings
        cached, missing_indices = self.cache.get(texts)
        
        if not missing_indices:
            # All embeddings were cached
            return np.array([cached[i] for i in range(len(texts))])
        
        # Compute missing embeddings
        missing_texts = [texts[i] for i in missing_indices]
        new_embeddings = self.model.encode(missing_texts)
        
        # Save to cache
        self.cache.save(missing_texts, new_embeddings)
        
        # Combine cached and new embeddings
        all_embeddings = []
        new_idx = 0
        
        for i in range(len(texts)):
            if i in cached:
                all_embeddings.append(cached[i])
            else:
                all_embeddings.append(new_embeddings[new_idx])
                new_idx += 1
        
        return np.array(all_embeddings)
    
    def compute_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between two sets of embeddings."""
        # Normalize embeddings
        embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
        
        return similarity_matrix
