"""
# ============================================================================
# Document Loader — Base Classes
# ============================================================================
#
# This module defines the foundational data structures and interface that
# ALL document loaders must follow. Think of it as the "contract" — any new
# loader (PDF, CSV, HTML, etc.) just needs to implement the `load` method.
#
# TWO KEY CLASSES:
# ----------------
# 1. Document   — A dataclass representing a loaded document's content + metadata.
# 2. DocumentLoader — An abstract base class that all loaders inherit from.
#
# WHY THIS DESIGN:
# ----------------
# By defining a common `Document` object, the rest of the pipeline (chunking,
# embedding, retrieval) doesn't care WHERE the data came from — it just works
# with `Document` objects. This is the "Dependency Inversion" principle:
#   - High-level modules (pipeline) depend on abstractions (DocumentLoader)
#   - Low-level modules (PDFLoader, CSVLoader) implement those abstractions
#
# EXAMPLE USAGE:
# ----------------
#     from loaders import TextLoader
#
#     loader = TextLoader()
#     documents = loader.load("path/to/file.txt")
#     for doc in documents:
#         print(doc.content[:100])   # First 100 chars of content
#         print(doc.metadata)        # {"source": "file.txt", "loader": "text", ...}
#
# ============================================================================
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ============================================================================
# Document — The Universal Data Container
# ============================================================================
# Every loader produces a list of Document objects. This is the "currency"
# that flows through the pipeline:
#
#   Loader → [Document] → Chunker → [Chunk] → Embedder → Vectors → VectorDB
#
# A Document holds:
#   - content:  The actual text content of the file (or a section of it)
#   - metadata: A dictionary of information ABOUT the content (source file,
#               page number, format, etc.)
#
# Metadata is crucial for RAG because when the system retrieves a chunk,
# you want to know WHERE it came from (which file, which page, which section).
# ============================================================================

@dataclass
class Document:
    """
    Represents a loaded document with its content and metadata.

    -----------------------------------------------------------------------
    Attributes:
        content  (str)  : The text content extracted from the source file.
        metadata (dict) : Key-value pairs describing the document's origin.
    -----------------------------------------------------------------------

    The metadata dictionary always includes at minimum:
        - "source"     : The file path or URL the document was loaded from
        - "loader"     : Which loader class was used (e.g., "pdf", "markdown")
        - "loaded_at"  : ISO timestamp of when the document was loaded

    Different loaders may add additional metadata:
        - PDF loader adds   : "page_number", "total_pages"
        - Markdown loader adds : "headers" (list of header hierarchy)
        - CSV loader adds   : "row_index", "column_names"

    -----------------------------------------------------------------------
    Example:
        doc = Document(
            content="Machine Learning is a subset of AI...",
            metadata={
                "source": "/docs/ml_intro.pdf",
                "loader": "pdf",
                "page_number": 3,
                "total_pages": 45,
                "loaded_at": "2025-03-07T10:30:00"
            }
        )
    -----------------------------------------------------------------------
    """

    content: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Ensure every Document has a 'loaded_at' timestamp in its metadata.
        This is called automatically after __init__ by the dataclass machinery.
        """
        if "loaded_at" not in self.metadata:
            self.metadata["loaded_at"] = datetime.now().isoformat()

    def __len__(self) -> int:
        """Return the length of the document content in characters."""
        return len(self.content)

    def __repr__(self) -> str:
        """
        Pretty representation for debugging.
        Shows the source and a preview of the content.

        Example output:
            Document(source='ml_intro.pdf', chars=4523, preview='Machine Learning is...')
        """
        source = self.metadata.get("source", "unknown")
        preview = self.content[:50].replace("\n", " ") + "..." if len(self.content) > 50 else self.content
        return f"Document(source='{source}', chars={len(self.content)}, preview='{preview}')"


# ============================================================================
# DocumentLoader — The Abstract Interface
# ============================================================================
# This is an abstract base class (ABC). You CANNOT instantiate it directly.
# Instead, concrete loaders (PDFLoader, TextLoader, etc.) inherit from it
# and implement the `load` method.
#
# Think of it like a Java interface or a TypeScript abstract class — it
# defines WHAT a loader must do, but not HOW.
#
#   class MyCustomLoader(DocumentLoader):
#       def load(self, source: str) -> list[Document]:
#           # Your custom loading logic here
#           ...
#
# ============================================================================

class DocumentLoader(ABC):
    """
    Abstract base class for all document loaders.

    -----------------------------------------------------------------------
    Every loader must implement:
        - load(source: str) -> list[Document]

    Every loader gets for free:
        - supported_extensions : Class variable listing handled file types
        - can_load(source)     : Check if this loader handles a given file
        - _build_metadata()    : Helper to build consistent metadata dicts
    -----------------------------------------------------------------------

    Example (creating a custom loader):
        class HTMLLoader(DocumentLoader):
            supported_extensions = [".html", ".htm"]

            def load(self, source: str) -> list[Document]:
                with open(source, "r") as f:
                    raw_html = f.read()
                text = strip_html_tags(raw_html)
                return [Document(content=text, metadata=self._build_metadata(source))]
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # Class variable: which file extensions this loader can handle.
    # Override in subclasses. Example: [".pdf"] for PDFLoader
    # -----------------------------------------------------------------------
    supported_extensions: list[str] = []

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """
        Load a file and return a list of Document objects.

        -----------------------------------------------------------------------
        Args:
            source: Path to the file to load (absolute or relative).

        Returns:
            A list of Document objects. Most loaders return a single Document,
            but some (like PDF) may return one Document per page.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the file format can't be parsed.
        -----------------------------------------------------------------------

        Why a list? Because some formats naturally split into multiple parts:
        - A PDF with 10 pages → 10 Documents (one per page)
        - A CSV with 100 rows → 100 Documents (one per row) or 1 Document
        - A single text file → 1 Document
        """
        pass

    def can_load(self, source: str) -> bool:
        """
        Check if this loader can handle the given file.

        -----------------------------------------------------------------------
        Uses the file extension to determine compatibility.

        Example:
            pdf_loader = PDFLoader()
            pdf_loader.can_load("report.pdf")   # True
            pdf_loader.can_load("readme.md")    # False
        -----------------------------------------------------------------------
        """
        import os
        _, ext = os.path.splitext(source)
        return ext.lower() in self.supported_extensions

    def _build_metadata(self, source: str, **extra) -> dict:
        """
        Build a consistent metadata dictionary for a document.

        -----------------------------------------------------------------------
        Creates the base metadata that every Document should have,
        and merges in any additional key-value pairs.

        Args:
            source: The file path being loaded.
            **extra: Additional metadata specific to the loader.
                     Example: page_number=3, total_pages=45

        Returns:
            dict with at least: source, loader, file_size, loaded_at

        Example:
            metadata = self._build_metadata(
                "report.pdf",
                page_number=3,
                total_pages=45
            )
            # {"source": "report.pdf", "loader": "pdf", "file_size": 12345,
            #  "page_number": 3, "total_pages": 45, "loaded_at": "2025-..."}
        -----------------------------------------------------------------------
        """
        import os

        base = {
            "source": os.path.abspath(source),
            "loader": self.__class__.__name__.replace("Loader", "").lower(),
            "file_name": os.path.basename(source),
            "file_size": os.path.getsize(source) if os.path.exists(source) else 0,
        }
        base.update(extra)
        return base
