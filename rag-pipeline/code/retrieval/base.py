"""
# ============================================================================
# Retrieval Engine — Base Classes
# ============================================================================
#
# The retriever sits between the vector store and the LLM. Its job is to
# take a user's question, find the most RELEVANT chunks, and return them.
#
# Simple retrieval: just ask the vector store for similar chunks.
# Better retrieval: get candidates from vector store, then RE-RANK them
# using a cross-encoder for much better relevance scoring.
#
# See docs/04-retrieval-mechanisms.md for the full theory.
#
# ============================================================================
"""

from abc import ABC, abstractmethod

from ..vectorstore.base import SearchResult


# ============================================================================
# Retriever — The Abstract Interface
# ============================================================================

class Retriever(ABC):
    """
    Abstract base class for all retrieval implementations.

    -----------------------------------------------------------------------
    Every retriever must implement:
        - retrieve(query, top_k) -> list[SearchResult]

    The retriever encapsulates the full retrieval logic:
        - VectorRetriever: Just does vector similarity search
        - (Future) HybridRetriever: Vector + keyword search
    -----------------------------------------------------------------------
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Retrieve the most relevant chunks for a query.

        -----------------------------------------------------------------------
        Args:
            query: The user's question as a string.
            top_k: How many chunks to return.

        Returns:
            List of SearchResult objects, sorted by relevance.
        -----------------------------------------------------------------------
        """
        pass
