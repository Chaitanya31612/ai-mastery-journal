"""
# ============================================================================
# Text Document Loader
# ============================================================================
#
# The simplest loader — reads plain text files (.txt) and returns their
# content as a single Document.
#
# This is the "hello world" of document loaders. It's a great reference
# for understanding the loader pattern before looking at more complex ones.
#
# WHAT IT DOES:
# -------------
# 1. Opens the file
# 2. Reads all text content
# 3. Wraps it in a Document with metadata (source, file size, line count)
#
# EXAMPLE:
# --------
#     loader = TextLoader()
#     docs = loader.load("notes.txt")
#     print(docs[0].content)        # The full text content
#     print(docs[0].metadata)       # {"source": "notes.txt", "loader": "text", ...}
#
# ============================================================================
"""

import os
from .base import DocumentLoader, Document


# ============================================================================
# TextLoader — Plain Text File Loader
# ============================================================================

class TextLoader(DocumentLoader):
    """
    Loads plain text files (.txt) into Document objects.

    -----------------------------------------------------------------------
    This is the simplest loader. It reads the entire file as a string
    and returns it as a single Document.

    Supports:
        - .txt files
        - Any text encoding (configurable, defaults to UTF-8)

    Example:
        loader = TextLoader(encoding="utf-8")
        documents = loader.load("my_notes.txt")
        # Returns: [Document(content="...", metadata={...})]
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # File extensions this loader handles
    # -----------------------------------------------------------------------
    supported_extensions = [".txt"]

    def __init__(self, encoding: str = "utf-8"):
        """
        -----------------------------------------------------------------------
        Args:
            encoding: Text encoding to use when reading files.
                      Common values: "utf-8" (default, works for most files),
                      "latin-1" (for older files), "ascii" (strict ASCII only).
        -----------------------------------------------------------------------
        """
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        """
        Load a text file and return its content as a Document.

        -----------------------------------------------------------------------
        Args:
            source: Path to the .txt file.

        Returns:
            A list containing exactly ONE Document with:
              - content: The entire file text
              - metadata: source path, loader type, line count, char count

        Raises:
            FileNotFoundError: If the file doesn't exist.
            UnicodeDecodeError: If the file can't be decoded with the specified encoding.
        -----------------------------------------------------------------------

        Example:
            # File "notes.txt" contains:
            #   "Hello World\\nThis is line 2\\nAnd line 3"

            docs = TextLoader().load("notes.txt")
            len(docs)                          # 1
            docs[0].content                    # "Hello World\\nThis is line 2\\nAnd line 3"
            docs[0].metadata["line_count"]     # 3
            docs[0].metadata["char_count"]     # 38
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Validate the file exists
        # -----------------------------------------------------------------------
        if not os.path.exists(source):
            raise FileNotFoundError(f"Text file not found: {source}")

        # -----------------------------------------------------------------------
        # Step 2: Read the entire file content
        # -----------------------------------------------------------------------
        with open(source, "r", encoding=self.encoding) as f:
            content = f.read()

        # -----------------------------------------------------------------------
        # Step 3: Skip empty files — no point indexing nothing
        # -----------------------------------------------------------------------
        if not content.strip():
            return []

        # -----------------------------------------------------------------------
        # Step 4: Build metadata and return a Document
        # -----------------------------------------------------------------------
        metadata = self._build_metadata(
            source,
            line_count=content.count("\n") + 1,
            char_count=len(content),
        )

        return [Document(content=content, metadata=metadata)]
