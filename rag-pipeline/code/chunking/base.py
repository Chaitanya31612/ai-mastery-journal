"""
# ============================================================================
# Chunking Engine — Base Classes
# ============================================================================
#
# This module defines the data structures and interface for chunking.
# Chunking = splitting documents into smaller, embeddable pieces.
#
# TWO KEY CLASSES:
# ----------------
# 1. Chunk             — A dataclass representing a piece of a document.
# 2. ChunkingStrategy  — Abstract base class that all chunkers implement.
#
# THE RELATIONSHIP BETWEEN DOCUMENTS AND CHUNKS:
# ------------------------------------------------
#     Document (from loader)     →     Chunks (from chunker)
#     ┌──────────────────────┐        ┌───────────────┐
#     │ "This is a long      │   →    │ "This is a    │  Chunk 0
#     │  document about      │        │  long document│
#     │  machine learning.   │        │  about..."    │
#     │  It covers many      │        └───────────────┘
#     │  topics including    │        ┌───────────────┐
#     │  neural networks,    │   →    │ "...many      │  Chunk 1
#     │  transformers, and   │        │  topics incl- │
#     │  attention..."       │        │  uding..."    │
#     └──────────────────────┘        └───────────────┘
#
# Each Chunk inherits metadata from its parent Document AND adds its own
# metadata (chunk index, position, etc.).
#
# ============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..loaders.base import Document


# ============================================================================
# Chunk — A Piece of a Document
# ============================================================================
# This is what gets embedded and stored in the vector database.
# It's similar to Document but with additional chunking-specific metadata.
# ============================================================================

@dataclass
class Chunk:
    """
    Represents a piece of a document after chunking.

    -----------------------------------------------------------------------
    Attributes:
        content       (str)  : The text content of this chunk.
        metadata      (dict) : Everything we know about this chunk's origin.
        chunk_index   (int)  : Position of this chunk (0-indexed) within
                               its parent document.
        document_id   (str)  : Unique identifier linking back to the source document.

    The metadata dictionary includes everything from the parent Document
    PLUS chunking-specific fields:
        - "chunk_index"    : Position in the document (0, 1, 2, ...)
        - "total_chunks"   : How many chunks the document was split into
        - "chunk_size"     : Character count of this chunk
        - "chunk_strategy" : Which chunking algorithm was used

    -----------------------------------------------------------------------
    Example:
        chunk = Chunk(
            content="Neural networks consist of layers of connected neurons...",
            metadata={
                "source": "ml_intro.md",
                "chunk_index": 3,
                "total_chunks": 12,
                "chunk_size": 487,
                "chunk_strategy": "recursive",
            },
            chunk_index=3,
            document_id="ml_intro.md_chunk_3"
        )
    -----------------------------------------------------------------------
    """

    content: str
    metadata: dict = field(default_factory=dict)
    chunk_index: int = 0
    document_id: str = ""

    def __len__(self) -> int:
        """Return the length of the chunk in characters."""
        return len(self.content)

    def __repr__(self) -> str:
        """
        Pretty representation for debugging.

        Example output:
            Chunk(index=3, chars=487, preview='Neural networks consist of...')
        """
        preview = self.content[:60].replace("\n", " ") + "..." if len(self.content) > 60 else self.content
        return f"Chunk(index={self.chunk_index}, chars={len(self.content)}, preview='{preview}')"


# ============================================================================
# ChunkingStrategy — The Abstract Interface
# ============================================================================
# All chunking algorithms implement this interface. The pipeline doesn't
# care HOW the chunking is done — it just calls strategy.chunk(document).
#
# This is the "Strategy Pattern" from design patterns:
#   - The algorithm varies (fixed, recursive, semantic, markdown)
#   - The interface stays the same (chunk method)
#   - You can swap strategies without changing the pipeline
# ============================================================================

class ChunkingStrategy(ABC):
    """
    Abstract base class for all chunking strategies.

    -----------------------------------------------------------------------
    Every chunking strategy must implement:
        - chunk(document: Document) -> list[Chunk]

    Every strategy gets for free:
        - _build_chunks() : Helper to create Chunk objects with proper metadata

    Subclasses:
        - FixedSizeChunker    : Splits every N characters (simplest)
        - RecursiveChunker    : Splits on natural boundaries (default)
        - MarkdownChunker     : Splits on Markdown headers
        - SemanticChunker     : Splits on topic boundaries using embeddings

    Example (using a strategy):
        strategy = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = strategy.chunk(document)
        for chunk in chunks:
            print(f"Chunk {chunk.chunk_index}: {len(chunk)} chars")
    -----------------------------------------------------------------------
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a Document into a list of Chunks.

        -----------------------------------------------------------------------
        Args:
            document: A Document object from any loader.

        Returns:
            A list of Chunk objects, each containing a piece of the
            document's content and combined metadata.
        -----------------------------------------------------------------------
        """
        pass

    def _build_chunks(
        self,
        texts: list[str],
        document: Document,
        strategy_name: str,
        extra_metadata: list[dict] | None = None,
    ) -> list[Chunk]:
        """
        Helper to create Chunk objects from a list of text pieces.

        -----------------------------------------------------------------------
        This is called by all chunking strategies after they've split the text.
        It handles:
            - Creating Chunk objects with proper metadata
            - Merging parent Document metadata with chunk-specific metadata
            - Assigning chunk indices and document IDs

        Args:
            texts:          List of text strings (the split pieces).
            document:       The parent Document (for inheriting metadata).
            strategy_name:  Name of the chunking strategy (e.g., "recursive").
            extra_metadata: Optional per-chunk metadata (must match len(texts)).
                            Example: [{"headers": ["Intro"]}, {"headers": ["Methods"]}]

        Returns:
            List of Chunk objects with complete metadata.

        Example:
            texts = ["First chunk text...", "Second chunk text..."]
            chunks = self._build_chunks(texts, document, "recursive")
            # chunks[0].metadata includes both document metadata AND:
            #   {"chunk_index": 0, "total_chunks": 2, "chunk_strategy": "recursive"}
        -----------------------------------------------------------------------
        """
        chunks = []
        total = len(texts)

        for i, text in enumerate(texts):
            # ---------------------------------------------------------------
            # Skip empty chunks (can happen with aggressive splitting)
            # ---------------------------------------------------------------
            if not text.strip():
                continue

            # ---------------------------------------------------------------
            # Merge parent document metadata with chunk-specific metadata
            # ---------------------------------------------------------------
            chunk_metadata = {
                **document.metadata,          # Inherit everything from parent
                "chunk_index": i,
                "total_chunks": total,
                "chunk_size": len(text),
                "chunk_strategy": strategy_name,
            }

            # Add any extra per-chunk metadata (e.g., headers for markdown)
            if extra_metadata and i < len(extra_metadata):
                chunk_metadata.update(extra_metadata[i])

            # ---------------------------------------------------------------
            # Create a unique document_id for this chunk
            # Format: "filename_chunk_0", "filename_chunk_1", etc.
            # ---------------------------------------------------------------
            source = document.metadata.get("file_name", "unknown")
            doc_id = f"{source}_chunk_{i}"

            chunks.append(Chunk(
                content=text,
                metadata=chunk_metadata,
                chunk_index=i,
                document_id=doc_id,
            ))

        return chunks
