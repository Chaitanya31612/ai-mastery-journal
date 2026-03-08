"""
# ============================================================================
# Recursive Character Chunking Strategy
# ============================================================================
#
# This is the DEFAULT and MOST COMMONLY USED chunking strategy in RAG systems.
# It's the same approach used by LangChain's RecursiveCharacterTextSplitter.
#
# THE BIG IDEA:
# -------------
# Instead of blindly cutting every N characters (like FixedSizeChunker),
# we try to split on NATURAL BOUNDARIES in a priority order:
#
#     1. First, try splitting on double newlines (paragraph boundaries)
#     2. If chunks are still too big, split on single newlines
#     3. If still too big, split on sentences (". ")
#     4. If still too big, split on words (" ")
#     5. Last resort: split on characters
#
# This means:
#   - Paragraphs stay together when possible ✅
#   - Sentences stay together when possible ✅
#   - Words are never split (unless the word itself is huge) ✅
#
# HOW IT WORKS (STEP BY STEP):
# ----------------------------
# Let's say chunk_size=200 and our text is:
#
#     "Paragraph one about topic A. It has multiple sentences.
#
#      Paragraph two about topic B. This paragraph is quite long and
#      contains a lot of detail about the subject matter that goes on
#      and on for many characters exceeding our chunk size limit."
#
# Step 1: Split on "\n\n" → ["Paragraph one...", "Paragraph two..."]
# Step 2: Check each piece:
#         - "Paragraph one..." (58 chars) → Under 200, keep as-is ✅
#         - "Paragraph two..." (230 chars) → Over 200, split further ⚠️
# Step 3: Split the oversized piece on "\n" → still too long
# Step 4: Split on ". " → ["Paragraph two about topic B.",
#                           "This paragraph is quite long..."]
# Step 5: Each piece now fits → Done ✅
#
# The key insight: we try the "biggest" separator first and fall back
# to smaller separators only when needed. This preserves as much
# natural structure as possible.
#
# ============================================================================
"""

from .base import ChunkingStrategy, Chunk, Document


# ============================================================================
# RecursiveChunker — The Default, Best-Practice Chunker
# ============================================================================

class RecursiveChunker(ChunkingStrategy):
    """
    Splits text on natural boundaries, recursively falling through separators.

    -----------------------------------------------------------------------
    This is the RECOMMENDED chunker for most use cases. It produces chunks
    that respect paragraph and sentence boundaries, leading to better
    retrieval quality than fixed-size chunking.

    Args:
        chunk_size:    Maximum characters per chunk (default: 500).
        chunk_overlap: Characters to overlap between chunks (default: 50).
        separators:    Ordered list of separator strings to try.
                       Default: ["\\n\\n", "\\n", ". ", " ", ""]

    Example:
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(document)
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # Default separator hierarchy (from largest to smallest boundary)
    # -----------------------------------------------------------------------
    DEFAULT_SEPARATORS = [
        "\n\n",   # Paragraph boundaries (strongest split)
        "\n",     # Line boundaries
        ". ",     # Sentence boundaries
        ", ",     # Clause boundaries
        " ",      # Word boundaries
        "",       # Character-level (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document using recursive character splitting.

        -----------------------------------------------------------------------
        Delegates to _recursive_split() which does the actual work.
        Then wraps the results in proper Chunk objects with metadata.
        -----------------------------------------------------------------------
        """
        texts = self._recursive_split(document.content, self.separators)
        return self._build_chunks(texts, document, "recursive")

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """
        The core recursive algorithm.

        -----------------------------------------------------------------------
        HOW THIS WORKS:

        1. Take the first separator from the list (e.g., "\\n\\n")
        2. Split the text on that separator
        3. For each piece:
           a. If it fits in chunk_size → add to results
           b. If it doesn't fit → recursively split with the NEXT separator
        4. After processing all pieces, merge small adjacent pieces together
           to avoid tiny chunks

        This is "recursive" because an oversized piece gets split again
        with a finer separator — paragraph → line → sentence → word.

        Visual example:
            Text: "A paragraph.\\n\\nA very long paragraph that exceeds the limit.\\n\\nShort."

            Split on "\\n\\n":
              ["A paragraph.", "A very long paragraph that...", "Short."]

            Piece 1: "A paragraph." → fits ✅
            Piece 2: "A very long..." → too big, recurse with "\\n" separator
              → Split on "\\n", then ". ", etc. until it fits
            Piece 3: "Short." → fits ✅
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Base case: no more separators to try → force-split by characters
        # -----------------------------------------------------------------------
        if not separators:
            return [text] if text.strip() else []

        # -----------------------------------------------------------------------
        # If the text already fits, no splitting needed
        # -----------------------------------------------------------------------
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # -----------------------------------------------------------------------
        # Pick the current separator and split the text
        # -----------------------------------------------------------------------
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # ---------------------------------------------------------------
            # Last resort: character-level splitting (like FixedSizeChunker)
            # ---------------------------------------------------------------
            pieces = [text[i:i + self.chunk_size]
                      for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        else:
            pieces = text.split(separator)

        # -----------------------------------------------------------------------
        # Process each piece: keep if small enough, recurse if too big
        # -----------------------------------------------------------------------
        final_chunks = []
        current_piece = ""     # Accumulator for merging small pieces

        for piece in pieces:
            # ---------------------------------------------------------------
            # Try merging this piece with the accumulated current_piece
            # ---------------------------------------------------------------
            if current_piece:
                merged = current_piece + separator + piece
            else:
                merged = piece

            if len(merged) <= self.chunk_size:
                # ---------------------------------------------------------------
                # Merged piece fits → keep accumulating
                # This prevents tiny chunks by combining small adjacent pieces
                #
                # Example: if pieces are ["Short.", "Also short."] and both fit,
                # we merge them: "Short.\n\nAlso short." instead of two chunks
                # ---------------------------------------------------------------
                current_piece = merged
            else:
                # ---------------------------------------------------------------
                # Merged piece is too big → flush current_piece and handle
                # the new piece separately
                # ---------------------------------------------------------------
                if current_piece.strip():
                    final_chunks.append(current_piece)

                if len(piece) <= self.chunk_size:
                    # This piece fits on its own
                    current_piece = piece
                else:
                    # This piece is STILL too big → recurse with next separator
                    sub_chunks = self._recursive_split(piece, remaining_separators)
                    final_chunks.extend(sub_chunks)
                    current_piece = ""

        # -----------------------------------------------------------------------
        # Don't forget the last accumulated piece
        # -----------------------------------------------------------------------
        if current_piece.strip():
            final_chunks.append(current_piece)

        return final_chunks
