"""
# ============================================================================
# Markdown Document Loader
# ============================================================================
#
# Parses Markdown (.md) files with awareness of their header structure.
# This is particularly useful because Markdown docs have natural hierarchy:
#
#   # Title           → level 1 (broadest)
#   ## Section        → level 2
#   ### Subsection    → level 3 (most specific)
#
# Unlike the TextLoader (which just reads raw text), this loader:
#   - Tracks the header hierarchy for each section
#   - Adds header information to metadata (useful for attribution)
#   - Can optionally split by sections (one Document per section)
#
# WHY A SEPARATE MARKDOWN LOADER:
# ---------------------------------
# Your learning docs (01-what-is-rag.md, etc.) are all Markdown. When you
# index them, you want the RAG system to know that a chunk came from
# "Section: Vector Similarity Search" under "Doc: Retrieval Mechanisms" —
# not just from some random chunk of text. This loader preserves that structure.
#
# EXAMPLE:
# --------
#     loader = MarkdownLoader(split_by_headers=True)
#     docs = loader.load("01-what-is-rag.md")
#     # Returns multiple Documents, one per section
#     # docs[0].metadata["headers"] = ["What is RAG?"]
#     # docs[1].metadata["headers"] = ["What is RAG?", "The Problem"]
#
# ============================================================================
"""

import os
import re
from .base import DocumentLoader, Document


# ============================================================================
# MarkdownLoader — Header-Aware Markdown Loader
# ============================================================================

class MarkdownLoader(DocumentLoader):
    """
    Loads Markdown files with optional section-based splitting.

    -----------------------------------------------------------------------
    Two modes:
        1. split_by_headers=False (default): Returns ONE Document with the
           entire file content. The markdown-aware chunker can split later.
        2. split_by_headers=True: Returns MULTIPLE Documents, one per
           top-level section. Useful for very large files.

    Supports: .md, .markdown files

    Example:
        # Mode 1: Single document (let the chunker handle splitting)
        loader = MarkdownLoader()
        docs = loader.load("guide.md")
        # Returns: [Document(content="entire file...", metadata={...})]

        # Mode 2: Split by headers (pre-split at load time)
        loader = MarkdownLoader(split_by_headers=True)
        docs = loader.load("guide.md")
        # Returns: [Document(section1...), Document(section2...), ...]
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # File extensions this loader handles
    # -----------------------------------------------------------------------
    supported_extensions = [".md", ".markdown"]

    def __init__(self, split_by_headers: bool = False, encoding: str = "utf-8"):
        """
        -----------------------------------------------------------------------
        Args:
            split_by_headers: If True, split the document into multiple
                             Documents at top-level headers (# and ##).
                             If False, return the entire file as one Document.
            encoding: Text encoding (default UTF-8).
        -----------------------------------------------------------------------
        """
        self.split_by_headers = split_by_headers
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        """
        Load a Markdown file and return Document(s).

        -----------------------------------------------------------------------
        Returns one or many Documents depending on split_by_headers setting.

        Metadata for each Document includes:
            - source      : File path
            - loader      : "markdown"
            - headers     : List of header hierarchy for this section
            - has_code    : Whether the section contains code blocks
            - char_count  : Number of characters in this section
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Validate and read the file
        # -----------------------------------------------------------------------
        if not os.path.exists(source):
            raise FileNotFoundError(f"Markdown file not found: {source}")

        with open(source, "r", encoding=self.encoding) as f:
            content = f.read()

        if not content.strip():
            return []

        # -----------------------------------------------------------------------
        # Step 2: Return based on mode
        # -----------------------------------------------------------------------
        if not self.split_by_headers:
            # --- Simple mode: return the whole file as one Document ---
            metadata = self._build_metadata(
                source,
                char_count=len(content),
                has_code=bool(re.search(r"```", content)),
            )
            return [Document(content=content, metadata=metadata)]

        # -----------------------------------------------------------------------
        # Step 3: Split by headers mode — parse sections
        # -----------------------------------------------------------------------
        return self._split_into_sections(content, source)

    def _split_into_sections(self, content: str, source: str) -> list[Document]:
        """
        Split markdown content into Documents at header boundaries.

        -----------------------------------------------------------------------
        HOW IT WORKS:
        1. We scan line by line, looking for lines starting with # (headers).
        2. When we find a header, we "close" the previous section and start
           a new one, tracking the header hierarchy.
        3. We skip content before the first header (usually just blank lines).

        Example input:
            # Introduction
            Some intro text.
            ## Background
            Background info here.
            ## Methods
            Methods details.

        Example output:
            Document(content="# Introduction\nSome intro text.\n",
                     metadata={"headers": ["Introduction"]})
            Document(content="## Background\nBackground info here.\n",
                     metadata={"headers": ["Introduction", "Background"]})
            Document(content="## Methods\nMethods details.\n",
                     metadata={"headers": ["Introduction", "Methods"]})
        -----------------------------------------------------------------------
        """

        documents = []
        lines = content.split("\n")

        # -----------------------------------------------------------------------
        # State tracking as we scan through lines
        # -----------------------------------------------------------------------
        current_section_lines = []       # Lines in the current section
        current_headers = {}             # Header hierarchy: {level: title}
        in_code_block = False            # Are we inside a ``` code block?

        for line in lines:
            # ---------------------------------------------------------------
            # Track code blocks — we don't want to split on # inside code
            # Example: ```python\n# This is a comment, NOT a header\n```
            # ---------------------------------------------------------------
            if line.strip().startswith("```"):
                in_code_block = not in_code_block

            # ---------------------------------------------------------------
            # Check if this line is a Markdown header (only outside code blocks)
            #
            # Regex: ^(#{1,6})\s+(.+)$
            #   ^       = start of line
            #   (#{1,6}) = 1 to 6 hash characters (header level)
            #   \s+     = one or more spaces
            #   (.+)$   = the header text until end of line
            # ---------------------------------------------------------------
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line) if not in_code_block else None

            if header_match:
                # ---------------------------------------------------------------
                # We found a new header! Save the previous section (if any)
                # ---------------------------------------------------------------
                if current_section_lines:
                    section_text = "\n".join(current_section_lines)
                    if section_text.strip():
                        header_list = [current_headers[k] for k in sorted(current_headers.keys())]
                        metadata = self._build_metadata(
                            source,
                            headers=header_list,
                            char_count=len(section_text),
                            has_code=bool(re.search(r"```", section_text)),
                        )
                        documents.append(Document(content=section_text, metadata=metadata))

                # ---------------------------------------------------------------
                # Start tracking the new section
                # ---------------------------------------------------------------
                level = len(header_match.group(1))   # Number of # characters
                title = header_match.group(2).strip()

                # Update header hierarchy:
                # When we see a ## after a ###, we clear the ### level
                current_headers[level] = title
                # Remove any deeper levels (a ## header clears all ### and below)
                current_headers = {k: v for k, v in current_headers.items() if k <= level}
                current_headers[level] = title

                current_section_lines = [line]
            else:
                current_section_lines.append(line)

        # -----------------------------------------------------------------------
        # Don't forget the last section!
        # -----------------------------------------------------------------------
        if current_section_lines:
            section_text = "\n".join(current_section_lines)
            if section_text.strip():
                header_list = [current_headers[k] for k in sorted(current_headers.keys())]
                metadata = self._build_metadata(
                    source,
                    headers=header_list,
                    char_count=len(section_text),
                    has_code=bool(re.search(r"```", section_text)),
                )
                documents.append(Document(content=section_text, metadata=metadata))

        return documents
