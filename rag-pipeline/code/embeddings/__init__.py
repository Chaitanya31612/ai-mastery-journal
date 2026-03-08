"""
# ============================================================================
# Embeddings Package — Public API
# ============================================================================
"""

from .base import EmbeddingProvider
from .sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerEmbedder",
]
