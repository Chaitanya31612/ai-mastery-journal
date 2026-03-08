"""
# ============================================================================
# Chunking Package — Public API
# ============================================================================
#
# Exports all chunking strategies so you can import like:
#     from code.chunking import RecursiveChunker, MarkdownChunker, Chunk
#
# Includes a factory function to create chunkers by name (used by the
# pipeline's config system).
# ============================================================================
"""

from .base import Chunk, ChunkingStrategy
from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker
from .markdown_chunker import MarkdownChunker
from .semantic import SemanticChunker


# ============================================================================
# Chunker Factory — Create a chunker by name
# ============================================================================
# This is used by the pipeline to create the right chunker based on
# the config.chunking_strategy string.
#
# Example:
#     chunker = create_chunker("recursive", chunk_size=500, chunk_overlap=50)
#     chunks = chunker.chunk(document)
# ============================================================================

def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model=None,
    **kwargs,
) -> ChunkingStrategy:
    """
    Factory function to create a chunker by strategy name.

    -----------------------------------------------------------------------
    Args:
        strategy: Name of the chunking strategy.
                  Options: "fixed", "recursive", "markdown", "semantic"
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters to overlap between chunks.
        embedding_model: Required for "semantic" strategy only.
        **kwargs: Additional arguments passed to the chunker constructor.

    Returns:
        A ChunkingStrategy instance.

    Raises:
        ValueError: If strategy name is not recognized.

    Example:
        chunker = create_chunker("markdown", chunk_size=800)
        chunker = create_chunker("semantic", embedding_model=model)
    -----------------------------------------------------------------------
    """

    strategies = {
        "fixed": lambda: FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        "recursive": lambda: RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        "markdown": lambda: MarkdownChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        "semantic": lambda: SemanticChunker(embedding_model=embedding_model, **kwargs),
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Available: {list(strategies.keys())}"
        )

    return strategies[strategy]()


__all__ = [
    "Chunk",
    "ChunkingStrategy",
    "FixedSizeChunker",
    "RecursiveChunker",
    "MarkdownChunker",
    "SemanticChunker",
    "create_chunker",
]
