"""
# ============================================================================
# Embedding Engine — Base Classes
# ============================================================================
#
# This module defines the interface for embedding providers.
# An embedding provider converts text into numeric vectors (embeddings)
# that capture the semantic meaning of the text.
#
# WHERE EMBEDDINGS FIT IN THE PIPELINE:
# --------------------------------------
#     Chunks (text) ──▶ EmbeddingProvider ──▶ Vectors (numbers) ──▶ VectorDB
#
# The same provider is used for BOTH:
#   1. Generating embeddings during ingestion (chunks → vectors)
#   2. Generating embeddings during query (question → vector)
#
# Both MUST use the same model so they exist in the same vector space.
#
# ============================================================================
"""

from abc import ABC, abstractmethod
import numpy as np


# ============================================================================
# EmbeddingProvider — The Abstract Interface
# ============================================================================

class EmbeddingProvider(ABC):
    """
    Abstract base class for all embedding providers.

    -----------------------------------------------------------------------
    Every embedding provider must implement:
        - embed(texts)     : Convert a list of texts to vectors
        - embed_query(text): Convert a single query to a vector
        - dimension        : Return the vector dimension size

    Why two methods (embed vs embed_query)?
    Some providers (like certain API-based ones) handle single queries
    differently from batch documents (e.g., different prefixes or
    parameters). Having separate methods lets providers optimize each case.
    For most local models, embed_query just calls embed with one text.

    Example (using a provider):
        provider = SentenceTransformerEmbedder()

        # Embed documents (batch)
        vectors = provider.embed(["chunk 1 text", "chunk 2 text"])
        # vectors.shape = (2, 384)

        # Embed a query (single)
        query_vector = provider.embed_query("What is RAG?")
        # query_vector.shape = (384,)
    -----------------------------------------------------------------------
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        -----------------------------------------------------------------------
        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
            Each row is the embedding vector for the corresponding text.

        Example:
            vectors = provider.embed(["Hello world", "Goodbye world"])
            vectors.shape  # (2, 384)
            vectors[0]     # [0.12, -0.34, 0.56, ...] — embedding for "Hello world"
        -----------------------------------------------------------------------
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text into a vector.

        -----------------------------------------------------------------------
        Args:
            text: A single query string.

        Returns:
            numpy array of shape (dimension,).

        This is separated from embed() because some providers handle
        queries differently (e.g., adding a "query:" prefix for
        asymmetric search models).
        -----------------------------------------------------------------------
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimensionality of the embedding vectors.

        -----------------------------------------------------------------------
        Example: 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2.
        This is used by the vector store to know how many dimensions to expect.
        -----------------------------------------------------------------------
        """
        pass
