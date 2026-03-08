"""
# ============================================================================
# Vector Store — Base Classes
# ============================================================================
#
# This module defines the interface for vector stores — databases that
# store and search document embeddings.
#
# WHERE THE VECTOR STORE FITS:
# ----------------------------
#     Ingestion:  Chunks + Vectors ──▶ VectorStore.add()
#     Query:      Question Vector  ──▶ VectorStore.search() ──▶ Similar Chunks
#
# The vector store is the PERSISTENT MEMORY of the RAG pipeline.
# Once you ingest documents, the vector store remembers them across
# Python restarts (unlike in-memory lists that disappear).
#
# ============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from ..chunking.base import Chunk


# ============================================================================
# SearchResult — What You Get Back From a Search
# ============================================================================

@dataclass
class SearchResult:
    """
    A single result from a vector store search.

    -----------------------------------------------------------------------
    Contains the original chunk plus its similarity score to the query.

    Attributes:
        chunk:    The original Chunk object that matched.
        score:    Similarity score (higher = more similar).
                  Range depends on the distance metric:
                    - Cosine: -1 to 1 (1 = identical)
                    - L2:     0 to ∞ (0 = identical, but we invert it)
        rank:     Position in the results (0 = best match).

    Example:
        result = SearchResult(
            chunk=Chunk(content="Neural networks are...", ...),
            score=0.85,
            rank=0
        )
        print(f"Best match (score {result.score}): {result.chunk.content[:50]}")
    -----------------------------------------------------------------------
    """

    chunk: Chunk
    score: float = 0.0
    rank: int = 0


# ============================================================================
# VectorStore — The Abstract Interface
# ============================================================================

class VectorStore(ABC):
    """
    Abstract base class for all vector store implementations.

    -----------------------------------------------------------------------
    Every vector store must implement:
        - add()       : Store chunks and their embeddings
        - search()    : Find the most similar chunks to a query vector
        - delete()    : Remove chunks by their IDs
        - count()     : Return the number of stored chunks
        - reset()     : Clear all stored data

    Example:
        store = ChromaStore(path="./data/vectordb")

        # During ingestion
        store.add(chunks, embeddings)

        # During query
        results = store.search(query_vector, top_k=5)
        for result in results:
            print(f"Score {result.score:.2f}: {result.chunk.content[:80]}")
    -----------------------------------------------------------------------
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Store chunks and their corresponding embeddings.

        -----------------------------------------------------------------------
        Args:
            chunks:     List of Chunk objects to store.
            embeddings: numpy array of shape (len(chunks), dimension).
                        embeddings[i] is the vector for chunks[i].

        The implementation must:
            1. Store the embedding vectors (for similarity search)
            2. Store the chunk content (to return in search results)
            3. Store the chunk metadata (for filtering and attribution)
            4. Generate unique IDs (or use chunk.document_id)
        -----------------------------------------------------------------------
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        Find the most similar chunks to a query embedding.

        -----------------------------------------------------------------------
        Args:
            query_embedding:  numpy array of shape (dimension,).
            top_k:            Number of results to return.
            filter_metadata:  Optional dict to filter results by metadata.
                              Example: {"source": "report.pdf"} only returns
                              chunks from that specific file.

        Returns:
            List of SearchResult objects, sorted by score (best first).
        -----------------------------------------------------------------------
        """
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete chunks by their IDs."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored chunks."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear ALL stored data. Use with caution!"""
        pass
