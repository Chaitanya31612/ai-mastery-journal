"""
# ============================================================================
# Markdown-Aware Chunking Strategy
# ============================================================================
#
# Splits Markdown documents based on their header structure (# , ## , ### ).
# This is the BEST chunker for .md files because it respects the document's
# natural hierarchy.
#
# HOW IT DIFFERS FROM RECURSIVE CHUNKER:
# ----------------------------------------
#
#   RecursiveChunker: Splits on "\n\n" → "\n" → ". " → " " → ""
#     - Treats markdown like any text — headers have no special meaning
#     - A "## Background" header might end up in the middle of a chunk
#
#   MarkdownChunker: Splits specifically at header lines
#     - Each section becomes its own chunk
#     - Header hierarchy is preserved in metadata
#     - If a section is too long, falls back to recursive splitting WITHIN
#       that section (best of both worlds)
#
# EXAMPLE:
# --------
#     Input markdown:
#         # Introduction
#         RAG stands for Retrieval-Augmented Generation.
#
#         ## Why RAG?
#         LLMs have knowledge cutoffs and can hallucinate.
#
#         ## How RAG Works
#         The pipeline has three steps...
#         [500+ chars of detailed explanation]
#
#     Output chunks:
#         Chunk 0: "# Introduction\nRAG stands for..."
#                  metadata.headers = ["Introduction"]
#         Chunk 1: "## Why RAG?\nLLMs have knowledge..."
#                  metadata.headers = ["Introduction", "Why RAG?"]
#         Chunk 2: "## How RAG Works\nThe pipeline..." (part 1)
#                  metadata.headers = ["Introduction", "How RAG Works"]
#         Chunk 3: "...detailed explanation..." (part 2, from recursive split)
#                  metadata.headers = ["Introduction", "How RAG Works"]
#
# ============================================================================
"""

import re
from .base import ChunkingStrategy, Chunk, Document
from .recursive import RecursiveChunker


# ============================================================================
# MarkdownChunker — Header-Aware Markdown Chunker
# ============================================================================

class MarkdownChunker(ChunkingStrategy):
    """
    Splits Markdown documents at header boundaries.

    -----------------------------------------------------------------------
    Preserves document hierarchy by tracking header levels and using
    them as natural chunk boundaries. Sections that exceed chunk_size
    are recursively split using RecursiveChunker.

    Args:
        chunk_size:    Maximum characters per chunk (default: 500).
        chunk_overlap: Overlap for fallback recursive splitting (default: 50).

    Example:
        chunker = MarkdownChunker(chunk_size=500)
        chunks = chunker.chunk(document)
        for chunk in chunks:
            print(chunk.metadata.get("headers"))  # ["Intro", "Background"]
            print(chunk.content[:80])
    -----------------------------------------------------------------------
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # -----------------------------------------------------------------------
        # Fallback chunker for sections that are too long.
        # If a single ## section has 2000 chars and chunk_size is 500,
        # we use RecursiveChunker to split it into manageable pieces.
        # -----------------------------------------------------------------------
        self._fallback_chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a Markdown document into chunks at header boundaries.

        -----------------------------------------------------------------------
        Algorithm:
            1. Split the text into sections based on Markdown headers
            2. For each section:
               a. If it fits in chunk_size → create one chunk
               b. If it's too long → recursively split it
            3. Attach header hierarchy to each chunk's metadata

        Returns:
            list[Chunk] — Chunks with header-aware metadata.
        -----------------------------------------------------------------------
        """

        sections = self._split_by_headers(document.content)

        # -----------------------------------------------------------------------
        # Convert sections to chunks, splitting oversized ones
        # -----------------------------------------------------------------------
        all_texts = []
        all_extra_meta = []

        for section in sections:
            text = section["content"]
            headers = section["headers"]

            if len(text) <= self.chunk_size:
                # ---------------------------------------------------------------
                # Section fits → one chunk
                # ---------------------------------------------------------------
                all_texts.append(text)
                all_extra_meta.append({"headers": headers})
            else:
                # ---------------------------------------------------------------
                # Section too long → recursively split it but keep the header
                # metadata for ALL resulting sub-chunks
                # ---------------------------------------------------------------
                sub_texts = self._fallback_chunker._recursive_split(
                    text, self._fallback_chunker.separators
                )
                for sub_text in sub_texts:
                    all_texts.append(sub_text)
                    all_extra_meta.append({"headers": headers})

        return self._build_chunks(
            all_texts, document, "markdown", extra_metadata=all_extra_meta
        )

    def _split_by_headers(self, content: str) -> list[dict]:
        """
        Parse markdown text into sections based on header lines.

        -----------------------------------------------------------------------
        Returns a list of dictionaries, each with:
            - "content": The text of this section (including the header line)
            - "headers": List of header titles forming the hierarchy

        HOW HEADER HIERARCHY WORKS:
        ---------------------------
        We track the "current header stack". When we see a new header:
            - Same or higher level → replace from that level down
            - Lower level → add to the stack

        Example:
            # Intro                 → headers = ["Intro"]
            ## Background           → headers = ["Intro", "Background"]
            ### Details             → headers = ["Intro", "Background", "Details"]
            ## Methods              → headers = ["Intro", "Methods"]
                                      (## clears ### because Methods is same level as Background)
        -----------------------------------------------------------------------
        """

        sections = []
        lines = content.split("\n")

        current_lines = []
        current_headers = {}     # {level: title} dictionary
        in_code_block = False

        for line in lines:
            # ---------------------------------------------------------------
            # Track code blocks to avoid treating # comments as headers
            # ---------------------------------------------------------------
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # ---------------------------------------------------------------
            # Check for Markdown headers (only outside code blocks)
            # ---------------------------------------------------------------
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line) if not in_code_block else None

            if header_match:
                # ---------------------------------------------------------------
                # Save the previous section (if non-empty)
                # ---------------------------------------------------------------
                if current_lines:
                    section_text = "\n".join(current_lines)
                    if section_text.strip():
                        header_list = [current_headers[k] for k in sorted(current_headers.keys())]
                        sections.append({
                            "content": section_text,
                            "headers": header_list,
                        })

                # ---------------------------------------------------------------
                # Update header hierarchy
                # ---------------------------------------------------------------
                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Clear headers at this level and below
                current_headers = {k: v for k, v in current_headers.items() if k < level}
                current_headers[level] = title

                current_lines = [line]
            else:
                current_lines.append(line)

        # -----------------------------------------------------------------------
        # Don't forget the last section
        # -----------------------------------------------------------------------
        if current_lines:
            section_text = "\n".join(current_lines)
            if section_text.strip():
                header_list = [current_headers[k] for k in sorted(current_headers.keys())]
                sections.append({
                    "content": section_text,
                    "headers": header_list,
                })

        return sections
