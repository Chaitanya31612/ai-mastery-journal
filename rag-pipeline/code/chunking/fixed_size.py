"""
# ============================================================================
# Fixed-Size Chunking Strategy
# ============================================================================
#
# The simplest chunking approach — split text every N characters.
#
# HOW IT WORKS:
# -------------
#     Text: "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
#     chunk_size=10, chunk_overlap=3
#
#     Chunk 0: "ABCDEFGHIJ"     (chars 0-9)
#     Chunk 1: "HIJKLMNOPQ"     (chars 7-16, overlaps "HIJ")
#     Chunk 2: "OPQRSTUVWX"     (chars 14-23, overlaps "OPQ")
#     Chunk 3: "VWXYZ"          (chars 21-25, overlaps "VWX")
#
#     Notice: each chunk starts (chunk_size - chunk_overlap) characters
#     after the previous one. The overlap ensures no content is "lost"
#     at the boundary between chunks.
#
# WHEN TO USE:
# ------------
# - Quick prototypes where you don't care about sentence boundaries
# - Uniform text without clear structure (logs, transcripts)
# - When you need predictable, equal-sized chunks
#
# WHEN NOT TO USE:
# ----------------
# - Structured text with paragraphs and headers → use RecursiveChunker
# - Markdown files → use MarkdownChunker
# - When retrieval quality matters most → use SemanticChunker
#
# ============================================================================
"""

from .base import ChunkingStrategy, Chunk, Document


# ============================================================================
# FixedSizeChunker
# ============================================================================

class FixedSizeChunker(ChunkingStrategy):
    """
    Splits documents into fixed-size character chunks with optional overlap.

    -----------------------------------------------------------------------
    This is the "dumbest" chunker — it just counts characters. It doesn't
    care about words, sentences, or paragraphs. A chunk might start mid-word
    or mid-sentence.

    Despite its simplicity, it's useful as a baseline and for log-style
    data where there's no natural structure.

    Args:
        chunk_size:    Maximum characters per chunk (default: 500).
        chunk_overlap: Characters to overlap between consecutive chunks (default: 50).

    Example:
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk(document)
        # Each chunk is ~200 chars, with 20-char overlap between neighbors
    -----------------------------------------------------------------------
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # -----------------------------------------------------------------------
        # Validate inputs — overlap must be smaller than chunk size,
        # otherwise chunks would never advance through the text.
        # -----------------------------------------------------------------------
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document into fixed-size chunks.

        -----------------------------------------------------------------------
        Algorithm (sliding window):
            1. Start at position 0
            2. Take chunk_size characters → that's one chunk
            3. Move forward by (chunk_size - chunk_overlap) characters
            4. Repeat until we've covered the entire text

        Visual:
            Text:     [==========|==========|==========|=====]
            Chunk 0:  [==========]
            Chunk 1:        [==========]        ← overlap region
            Chunk 2:              [==========]
            Chunk 3:                    [=====]  ← last chunk may be shorter

        Returns:
            list[Chunk] — The split chunks with inherited metadata.
        -----------------------------------------------------------------------
        """
        text = document.content

        # -----------------------------------------------------------------------
        # Edge case: if the text is shorter than one chunk, return it as-is
        # -----------------------------------------------------------------------
        if len(text) <= self.chunk_size:
            return self._build_chunks([text], document, "fixed")

        # -----------------------------------------------------------------------
        # Sliding window: advance by (chunk_size - overlap) each step
        # -----------------------------------------------------------------------
        texts = []
        step = self.chunk_size - self.chunk_overlap  # How far to move each iteration

        for start in range(0, len(text), step):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Skip chunks that are just whitespace
            if chunk_text.strip():
                texts.append(chunk_text)

            # If we've reached the end, stop
            if end >= len(text):
                break

        return self._build_chunks(texts, document, "fixed")
