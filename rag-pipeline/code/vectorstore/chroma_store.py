"""
# ============================================================================
# ChromaDB Vector Store Implementation
# ============================================================================
#
# ChromaDB is an EMBEDDED vector database — it runs inside your Python
# process (no separate server, no Docker, no API key). Think of it like
# SQLite for vectors.
#
# WHY CHROMADB:
# -------------
# - Zero setup: pip install chromadb and you're done
# - Persistent: data survives Python restarts (saved to disk)
# - Pythonic: clean, intuitive API
# - Fast enough: handles up to ~1M vectors comfortably
# - Built-in embedding support (we use our own for more control)
#
# HOW CHROMADB WORKS INTERNALLY:
# -------------------------------
# ChromaDB uses HNSW (Hierarchical Navigable Small World) for vector search.
# Think of HNSW like a layered graph:
#
#     Layer 3 (sparse):   A ──── B ──── C
#                         │             │
#     Layer 2 (medium):   A ── D ── B ── C ── E
#                         │    │    │    │    │
#     Layer 1 (dense):    A-F-D-G-B-H-C-I-E-J
#
# To search, you start at the top layer (fast, approximate) and
# drill down to lower layers (slower, precise). This gives you
# O(log n) search time instead of O(n) for brute-force comparison.
#
# KEY CONCEPTS:
# -------------
# - Collection: Like a "table" — a named group of vectors + metadata
# - Document:   The original text stored alongside the vector
# - Metadata:   Key-value pairs for filtering (source, page, etc.)
# - ID:         Unique identifier for each stored item
#
# ============================================================================
"""

import os
import numpy as np
from .base import VectorStore, SearchResult
from ..chunking.base import Chunk


# ============================================================================
# ChromaStore — ChromaDB Vector Store
# ============================================================================

class ChromaStore(VectorStore):
    """
    Persistent vector store using ChromaDB.

    -----------------------------------------------------------------------
    Stores document chunks and their embeddings on disk. Supports:
        - Adding chunks with embeddings
        - Similarity search with optional metadata filtering
        - Deleting specific chunks
        - Resetting (clearing) all data

    Args:
        path:            Directory for persistent storage (default: "./data/vectordb").
        collection_name: Name of the ChromaDB collection (default: "rag_documents").

    Example:
        store = ChromaStore(path="./data/vectordb", collection_name="my_docs")

        # Add chunks
        store.add(chunks, embeddings)
        print(f"Stored {store.count()} chunks")

        # Search
        results = store.search(query_vector, top_k=5)
        for r in results:
            print(f"[{r.score:.2f}] {r.chunk.content[:80]}...")

        # Filter by source file
        results = store.search(query_vector, top_k=5,
                              filter_metadata={"file_name": "report.pdf"})
    -----------------------------------------------------------------------
    """

    def __init__(
        self,
        path: str = "./data/vectordb",
        collection_name: str = "rag_documents",
    ):
        self.path = path
        self.collection_name = collection_name

        # -----------------------------------------------------------------------
        # Initialize ChromaDB with persistent storage
        # -----------------------------------------------------------------------
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB is required for vector storage. "
                "Install with: pip install chromadb"
            )

        # -----------------------------------------------------------------------
        # PersistentClient saves data to disk in the specified path.
        # This means vectors survive Python restarts — you only need to
        # ingest documents once, then query as many times as you want.
        #
        # Internally, ChromaDB creates files like:
        #   ./data/vectordb/
        #   ├── chroma-collections.parquet
        #   ├── chroma-embeddings.parquet
        #   └── index/
        #       └── ... (HNSW index files)
        # -----------------------------------------------------------------------
        os.makedirs(path, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)

        # -----------------------------------------------------------------------
        # get_or_create_collection:
        #   - If the collection exists → returns it (with all its data)
        #   - If it doesn't exist → creates a new empty one
        #
        # metadata={"hnsw:space": "cosine"} tells ChromaDB to use
        # cosine similarity for distance calculations (the standard for text).
        # -----------------------------------------------------------------------
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Store chunks and their embeddings in ChromaDB.

        -----------------------------------------------------------------------
        ChromaDB requires four parallel lists:
            - ids:        Unique string IDs for each item
            - embeddings: The vector embeddings
            - documents:  The original text content
            - metadatas:  The metadata dictionaries

        We generate IDs from chunk.document_id and handle the conversion
        from our Chunk objects to ChromaDB's format.

        IMPORTANT: ChromaDB metadata values must be strings, ints, floats,
        or bools. Lists and nested dicts are NOT supported, so we convert
        them to JSON strings.
        -----------------------------------------------------------------------
        """
        if not chunks or len(chunks) == 0:
            return

        import json

        # -----------------------------------------------------------------------
        # Prepare data in ChromaDB's required format
        # -----------------------------------------------------------------------
        ids = []
        documents = []
        metadatas = []
        embedding_list = embeddings.tolist()   # ChromaDB wants Python lists, not numpy

        for i, chunk in enumerate(chunks):
            # ---------------------------------------------------------------
            # Generate a unique ID for each chunk
            # If the chunk already has a document_id, use it.
            # Otherwise generate one from the collection name + index.
            # ---------------------------------------------------------------
            chunk_id = chunk.document_id or f"{self.collection_name}_{i}"

            # ChromaDB requires unique IDs — append a counter to avoid collisions
            # when re-ingesting the same document
            unique_id = f"{chunk_id}_{self._collection.count() + i}"

            ids.append(unique_id)
            documents.append(chunk.content)

            # ---------------------------------------------------------------
            # Sanitize metadata for ChromaDB
            # ChromaDB only accepts: str, int, float, bool as metadata values
            # Lists and dicts must be JSON-serialized to strings
            # ---------------------------------------------------------------
            clean_meta = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                elif isinstance(value, (list, dict)):
                    clean_meta[key] = json.dumps(value)
                elif value is None:
                    clean_meta[key] = ""
                else:
                    clean_meta[key] = str(value)

            metadatas.append(clean_meta)

        # -----------------------------------------------------------------------
        # Add to ChromaDB in batches (ChromaDB handles batching internally,
        # but we limit to 5000 per call to avoid memory issues)
        # -----------------------------------------------------------------------
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                embeddings=embedding_list[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for the most similar chunks to a query embedding.

        -----------------------------------------------------------------------
        Args:
            query_embedding:  numpy array of shape (dimension,).
            top_k:            Number of results to return (default: 5).
            filter_metadata:  Optional metadata filter.
                              Example: {"file_name": "report.pdf"}
                              Only returns chunks from that file.

        Returns:
            List of SearchResult objects, sorted by similarity (best first).

        HOW SIMILARITY SCORING WORKS:
        ChromaDB returns "distances" (lower = more similar for cosine).
        We convert to "scores" (higher = more similar) for consistency:
            score = 1 - distance  (for cosine distance)

        Example:
            results = store.search(query_vec, top_k=3)
            for r in results:
                print(f"Score: {r.score:.3f}")
                print(f"Source: {r.chunk.metadata['file_name']}")
                print(f"Content: {r.chunk.content[:100]}")
                print("---")
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Handle edge case: collection is empty
        # -----------------------------------------------------------------------
        if self._collection.count() == 0:
            return []

        # Don't request more results than what's available
        actual_top_k = min(top_k, self._collection.count())

        # -----------------------------------------------------------------------
        # Build query parameters
        # -----------------------------------------------------------------------
        query_params = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": actual_top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        # -----------------------------------------------------------------------
        # Add metadata filter if specified
        #
        # ChromaDB uses a "where" clause for filtering:
        #   {"file_name": "report.pdf"} → only chunks from report.pdf
        #   {"page_number": {"$gte": 5}} → only chunks from page 5+
        # -----------------------------------------------------------------------
        if filter_metadata:
            query_params["where"] = filter_metadata

        # -----------------------------------------------------------------------
        # Execute the search
        # -----------------------------------------------------------------------
        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            print(f"⚠️  Search error: {e}")
            return []

        # -----------------------------------------------------------------------
        # Convert ChromaDB results to our SearchResult objects
        #
        # ChromaDB returns nested lists because it supports batch queries:
        #   results["documents"] = [["doc1", "doc2", "doc3"]]  ← list of lists
        #   results["distances"] = [[0.1, 0.3, 0.5]]          ← list of lists
        #
        # We always query with one vector, so we take [0] to un-nest.
        # -----------------------------------------------------------------------
        import json

        search_results = []
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for rank, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            # ---------------------------------------------------------------
            # Convert distance to similarity score
            # For cosine distance: score = 1 - distance
            # Score range: 0 (unrelated) to 1 (identical)
            # ---------------------------------------------------------------
            score = 1.0 - dist

            # ---------------------------------------------------------------
            # Reconstruct a Chunk object from the stored data
            # Deserialize any JSON strings back to lists/dicts
            # ---------------------------------------------------------------
            clean_meta = {}
            for key, value in (meta or {}).items():
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, (list, dict)):
                            clean_meta[key] = parsed
                        else:
                            clean_meta[key] = value
                    except (json.JSONDecodeError, TypeError):
                        clean_meta[key] = value
                else:
                    clean_meta[key] = value

            chunk = Chunk(
                content=doc or "",
                metadata=clean_meta,
                chunk_index=clean_meta.get("chunk_index", 0),
            )

            search_results.append(SearchResult(
                chunk=chunk,
                score=score,
                rank=rank,
            ))

        return search_results

    def delete(self, ids: list[str]) -> None:
        """
        Delete chunks by their IDs.

        -----------------------------------------------------------------------
        Example:
            store.delete(["doc1_chunk_0_15", "doc1_chunk_1_16"])
        -----------------------------------------------------------------------
        """
        if ids:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()

    def reset(self) -> None:
        """
        Delete ALL data in this collection and recreate it.

        -----------------------------------------------------------------------
        ⚠️  WARNING: This permanently removes all indexed documents!
        Use this when you want to start fresh (e.g., after changing
        the embedding model or chunk size).
        -----------------------------------------------------------------------
        """
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
