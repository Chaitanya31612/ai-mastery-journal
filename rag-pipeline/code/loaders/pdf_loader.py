"""
# ============================================================================
# PDF Document Loader
# ============================================================================
#
# Extracts text from PDF files using PyMuPDF (imported as `fitz`).
# PDFs are the most common format for professional documents — financial
# reports, research papers, manuals — so this loader is critical for RAG.
#
# HOW PDF TEXT EXTRACTION WORKS:
# ------------------------------
# PDFs aren't like text files. They store content as positioned characters
# and shapes on a canvas. Text extraction "reads" these positioned characters
# and reconstructs the text flow. This means:
#   - Some PDFs extract perfectly (digitally created docs)
#   - Some extract poorly (scanned images, complex layouts)
#   - Tables and columns may come out garbled
#
# PyMuPDF (fitz) is one of the best libraries for this — it handles most
# PDFs well and is significantly faster than alternatives like PyPDF2.
#
# DESIGN CHOICE — ONE DOCUMENT PER PAGE:
# ----------------------------------------
# We return one Document per page (not one Document for the entire PDF).
# This is intentional:
#   - The chunker can work with page-level content (more precise)
#   - Metadata includes page numbers (great for source attribution)
#   - Memory-friendly for large PDFs (100+ pages)
#
# EXAMPLE:
# --------
#     loader = PDFLoader()
#     docs = loader.load("research_paper.pdf")
#     len(docs)                          # 12 (one per page)
#     docs[0].metadata["page_number"]    # 1
#     docs[0].metadata["total_pages"]    # 12
#
# ============================================================================
"""

import os
from .base import DocumentLoader, Document


# ============================================================================
# PDFLoader — PDF Document Loader (via PyMuPDF)
# ============================================================================

class PDFLoader(DocumentLoader):
    """
    Loads PDF files and returns one Document per page.

    -----------------------------------------------------------------------
    Uses PyMuPDF (fitz) for text extraction. Each page becomes its own
    Document object with page-level metadata.

    Supports:
        - .pdf files
        - Digitally created PDFs (best results)
        - Scanned PDFs with embedded text layers

    Does NOT support:
        - Scanned PDFs without OCR (just images, no text layer)
        - For those, you'd need an OCR step first (Tesseract, etc.)

    Example:
        loader = PDFLoader()
        documents = loader.load("report.pdf")
        for doc in documents:
            print(f"Page {doc.metadata['page_number']}: {doc.content[:80]}...")
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # File extensions this loader handles
    # -----------------------------------------------------------------------
    supported_extensions = [".pdf"]

    def load(self, source: str) -> list[Document]:
        """
        Load a PDF file and return one Document per page.

        -----------------------------------------------------------------------
        Args:
            source: Path to the .pdf file.

        Returns:
            A list of Document objects, one per page that has text content.
            Empty pages are skipped.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ImportError: If PyMuPDF is not installed.
            RuntimeError: If the PDF is corrupted or encrypted.

        The metadata for each Document includes:
            - source       : Full file path
            - loader       : "pdf"
            - page_number  : 1-indexed page number
            - total_pages  : Total pages in the PDF
            - file_name    : Just the filename (e.g., "report.pdf")
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Validate the file exists
        # -----------------------------------------------------------------------
        if not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")

        # -----------------------------------------------------------------------
        # Step 2: Import PyMuPDF (lazy import so the module loads even if
        #         fitz isn't installed — error only when actually used)
        # -----------------------------------------------------------------------
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF loading. "
                "Install it with: pip install PyMuPDF"
            )

        # -----------------------------------------------------------------------
        # Step 3: Open the PDF and extract text page by page
        # -----------------------------------------------------------------------
        documents = []

        # fitz.open() returns a document object we can iterate over
        with fitz.open(source) as pdf:
            total_pages = len(pdf)

            for page_num in range(total_pages):
                # ---------------------------------------------------------------
                # get_text() extracts text from a single page.
                # The "text" option gives us plain text (vs "html", "dict", etc.)
                # ---------------------------------------------------------------
                page = pdf[page_num]
                text = page.get_text("text")

                # ---------------------------------------------------------------
                # Skip pages with no meaningful text (blank pages, image-only pages)
                # ---------------------------------------------------------------
                if not text.strip():
                    continue

                # ---------------------------------------------------------------
                # Build page-level metadata and create a Document
                # ---------------------------------------------------------------
                metadata = self._build_metadata(
                    source,
                    page_number=page_num + 1,   # 1-indexed for human readability
                    total_pages=total_pages,
                )

                documents.append(Document(content=text, metadata=metadata))

        return documents
