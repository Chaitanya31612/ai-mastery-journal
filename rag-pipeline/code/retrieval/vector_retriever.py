"""
# ============================================================================
# Vector Retriever — Embedding-Based Document Retrieval
# ============================================================================
#
# This is the PRIMARY retriever. It takes a question, embeds it, searches
# the vector store, and returns the most similar chunks.
#
# WHERE IT FITS:
# --------------
#     User Question
#         │
#         ▼
#     ┌─────────────────────┐
#     │  VectorRetriever    │
#     │                     │
#     │  1. Embed question  │  ← Uses EmbeddingProvider
#     │  2. Search vectors  │  ← Uses VectorStore
#     │  3. Return results  │  → list[SearchResult]
#     └─────────────────────┘
#
# This retriever does ONE job well: convert questions to vectors and
# find the closest chunk vectors. For better relevance, pipe the
# results through the ReRanker (see reranker.py).
#
# ============================================================================
"""

from .base import Retriever
from ..vectorstore.base import VectorStore, SearchResult
from ..embeddings.base import EmbeddingProvider


# ============================================================================
# VectorRetriever — Similarity Search Retriever
# ============================================================================

class VectorRetriever(Retriever):
    """
    Retrieves chunks using vector similarity search.

    -----------------------------------------------------------------------
    This is the simplest and fastest retriever. It:
        1. Embeds the user's question using the embedding model
        2. Searches the vector store for similar vectors
        3. Returns the top-K most similar chunks

    It uses the SAME embedding model that was used during ingestion.
    This is critical — vectors from different models are incompatible.

    Args:
        vector_store:       The vector store to search (ChromaStore).
        embedding_provider: The embedding model (SentenceTransformerEmbedder).

    Example:
        retriever = VectorRetriever(
            vector_store=chroma_store,
            embedding_provider=embedder,
        )
        results = retriever.retrieve("What is pre-training?", top_k=5)
        for r in results:
            print(f"[{r.score:.2f}] {r.chunk.content[:80]}")
    -----------------------------------------------------------------------
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve chunks by embedding the query and searching the vector store.

        -----------------------------------------------------------------------
        Steps:
            1. Embed the query text → query vector (384 dimensions)
            2. Search the vector store for nearest neighbors
            3. Return results sorted by similarity score

        Args:
            query:            The user's question.
            top_k:            Number of results to return.
            filter_metadata:  Optional filter (e.g., {"source": "doc.pdf"}).

        Returns:
            List of SearchResult objects, best match first.

        Example:
            results = retriever.retrieve("How do transformers work?", top_k=3)
            # results[0].score = 0.89  (best match)
            # results[0].chunk.content = "Transformers use attention mechanisms..."
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Embed the question
        # "What is pre-training?" → [0.12, -0.34, 0.56, ...] (384 dims)
        # -----------------------------------------------------------------------
        query_embedding = self.embedding_provider.embed_query(query)

        # -----------------------------------------------------------------------
        # Step 2: Search the vector store
        # Find the top_k closest vectors (chunks) to the query vector
        # -----------------------------------------------------------------------
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        return results
