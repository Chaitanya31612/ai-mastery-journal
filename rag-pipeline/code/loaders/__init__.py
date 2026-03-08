"""
# ============================================================================
# Loaders Package — Public API
# ============================================================================
#
# This __init__.py file defines what gets exported when you do:
#     from loaders import TextLoader, PDFLoader, DirectoryLoader, ...
#
# USAGE:
# ------
#     from code.loaders import DirectoryLoader, Document
#
#     loader = DirectoryLoader()
#     documents = loader.load("path/to/docs/")
#
# ============================================================================
"""

from .base import Document, DocumentLoader
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .markdown_loader import MarkdownLoader
from .csv_loader import CSVLoader
from .json_loader import JSONLoader
from .directory_loader import DirectoryLoader

__all__ = [
    "Document",
    "DocumentLoader",
    "TextLoader",
    "PDFLoader",
    "MarkdownLoader",
    "CSVLoader",
    "JSONLoader",
    "DirectoryLoader",
]
