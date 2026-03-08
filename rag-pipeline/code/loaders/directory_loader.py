"""
# ============================================================================
# Directory Loader — Auto-Detect and Load All Files in a Directory
# ============================================================================
#
# This is the "smart" loader that you'll use most often. Point it at a folder
# and it automatically:
#   1. Walks through all files (recursively or not)
#   2. Detects the file type by extension
#   3. Routes each file to the appropriate loader (PDF, Markdown, Text, etc.)
#   4. Returns a combined list of all Documents
#
# WHO DECIDES WHICH LOADER TO USE:
# ----------------------------------
# The DirectoryLoader has a REGISTRY — a dictionary mapping file extensions
# to their loader classes. When it sees a .pdf file, it uses PDFLoader.
# When it sees a .md file, it uses MarkdownLoader. And so on.
#
# You can customize this registry to add support for new formats.
#
# EXAMPLE:
# --------
#     loader = DirectoryLoader()
#     docs = loader.load("my_docs/")
#     # Automatically loads all .pdf, .md, .txt, .csv, .json files
#     # Returns a flat list of all Documents from all files
#
#     # See what was loaded:
#     for doc in docs:
#         print(f"{doc.metadata['file_name']}: {len(doc.content)} chars")
#
# ============================================================================
"""

import os
from .base import DocumentLoader, Document
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .markdown_loader import MarkdownLoader
from .csv_loader import CSVLoader
from .json_loader import JSONLoader


# ============================================================================
# DirectoryLoader — Orchestrator for Multi-Format Loading
# ============================================================================

class DirectoryLoader(DocumentLoader):
    """
    Loads all supported files from a directory using format-specific loaders.

    -----------------------------------------------------------------------
    This is the recommended entry point for loading documents into the
    RAG pipeline. It handles:
        - Recursive directory walking
        - File type detection
        - Automatic routing to the correct loader
        - Error handling (skip problematic files with warnings)

    Supports: All formats from registered loaders
        .txt, .pdf, .md, .markdown, .csv, .json

    Example:
        # Load everything in a folder
        loader = DirectoryLoader(recursive=True)
        docs = loader.load("/path/to/documents/")

        # Load only specific types
        loader = DirectoryLoader(allowed_extensions=[".md", ".txt"])
        docs = loader.load("/path/to/documents/")
    -----------------------------------------------------------------------
    """

    def __init__(
        self,
        recursive: bool = True,
        allowed_extensions: list[str] | None = None,
        show_progress: bool = True,
    ):
        """
        -----------------------------------------------------------------------
        Args:
            recursive: If True, search subdirectories too.
                       If False, only load files in the top-level directory.

            allowed_extensions: Optional whitelist of extensions to load.
                                Example: [".md", ".txt"] to skip PDFs.
                                If None, all supported formats are loaded.

            show_progress: If True, print progress messages to console.
        -----------------------------------------------------------------------
        """
        self.recursive = recursive
        self.allowed_extensions = allowed_extensions
        self.show_progress = show_progress

        # -----------------------------------------------------------------------
        # LOADER REGISTRY
        # -----------------------------------------------------------------------
        # This maps file extensions to loader instances.
        # To add support for a new format:
        #   1. Create a new loader class (e.g., HTMLLoader)
        #   2. Add it here: ".html": HTMLLoader()
        # -----------------------------------------------------------------------
        self._loader_registry: dict[str, DocumentLoader] = {
            ".txt": TextLoader(),
            ".pdf": PDFLoader(),
            ".md": MarkdownLoader(),
            ".markdown": MarkdownLoader(),
            ".csv": CSVLoader(),
            ".json": JSONLoader(),
        }

    def load(self, source: str) -> list[Document]:
        """
        Load all supported files from a directory.

        -----------------------------------------------------------------------
        Args:
            source: Path to the directory (or a single file).
                    If it's a file, loads just that one file.
                    If it's a directory, walks through all files.

        Returns:
            A flat list of all Documents from all files.

        Skips (with warning):
            - Files with unsupported extensions
            - Files that fail to load (corrupted, encoding issues, etc.)
            - Empty files

        Example:
            Given directory structure:
                my_docs/
                ├── intro.md          (loads with MarkdownLoader)
                ├── data.csv          (loads with CSVLoader)
                ├── paper.pdf         (loads with PDFLoader)
                ├── notes.txt         (loads with TextLoader)
                └── config.yaml       (SKIPPED — no YAML loader)

            loader = DirectoryLoader()
            docs = loader.load("my_docs/")
            # Returns Documents from intro.md + data.csv + paper.pdf + notes.txt
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Handle single file input (not a directory)
        # -----------------------------------------------------------------------
        if os.path.isfile(source):
            return self._load_single_file(source)

        # -----------------------------------------------------------------------
        # Validate directory exists
        # -----------------------------------------------------------------------
        if not os.path.isdir(source):
            raise FileNotFoundError(f"Directory not found: {source}")

        # -----------------------------------------------------------------------
        # Walk the directory and collect all file paths
        # -----------------------------------------------------------------------
        file_paths = self._collect_file_paths(source)

        if self.show_progress:
            print(f"📁 Found {len(file_paths)} files to process in '{source}'")

        # -----------------------------------------------------------------------
        # Load each file using the appropriate loader
        # -----------------------------------------------------------------------
        all_documents = []
        loaded_count = 0
        skipped_count = 0

        for file_path in file_paths:
            try:
                docs = self._load_single_file(file_path)
                all_documents.extend(docs)
                loaded_count += 1

                if self.show_progress:
                    print(f"  ✅ Loaded: {os.path.basename(file_path)} ({len(docs)} document(s))")

            except Exception as e:
                skipped_count += 1
                if self.show_progress:
                    print(f"  ⚠️  Skipped: {os.path.basename(file_path)} — {e}")

        if self.show_progress:
            print(f"\n📊 Summary: {loaded_count} files loaded, "
                  f"{skipped_count} skipped, "
                  f"{len(all_documents)} total documents")

        return all_documents

    def _collect_file_paths(self, directory: str) -> list[str]:
        """
        Walk a directory and return paths to all supported files.

        -----------------------------------------------------------------------
        If recursive=True, walks subdirectories too (using os.walk).
        If recursive=False, only lists files in the top-level directory.

        Files are filtered by:
            1. Extension must be in the loader registry
            2. Extension must be in allowed_extensions (if specified)
            3. Hidden files (starting with .) are skipped
        -----------------------------------------------------------------------
        """
        file_paths = []

        if self.recursive:
            # ---------------------------------------------------------------
            # os.walk yields (dirpath, dirnames, filenames) for each directory
            # in the tree. We skip hidden directories (like .git, .venv).
            # ---------------------------------------------------------------
            for dirpath, dirnames, filenames in os.walk(directory):
                # Skip hidden directories
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

                for filename in sorted(filenames):
                    if not filename.startswith("."):
                        file_paths.append(os.path.join(dirpath, filename))
        else:
            # ---------------------------------------------------------------
            # Non-recursive: only top-level files
            # ---------------------------------------------------------------
            for filename in sorted(os.listdir(directory)):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path) and not filename.startswith("."):
                    file_paths.append(full_path)

        # -----------------------------------------------------------------------
        # Filter by supported and allowed extensions
        # -----------------------------------------------------------------------
        filtered = []
        for path in file_paths:
            _, ext = os.path.splitext(path)
            ext = ext.lower()

            # Must have a loader registered
            if ext not in self._loader_registry:
                continue

            # Must be in allowed list (if specified)
            if self.allowed_extensions and ext not in self.allowed_extensions:
                continue

            filtered.append(path)

        return filtered

    def _load_single_file(self, file_path: str) -> list[Document]:
        """
        Load a single file using the appropriate loader.

        -----------------------------------------------------------------------
        Looks up the file extension in the loader registry and delegates
        to the matching loader.

        Raises ValueError if no loader is registered for the extension.
        -----------------------------------------------------------------------
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in self._loader_registry:
            raise ValueError(
                f"No loader registered for '{ext}' files. "
                f"Supported: {list(self._loader_registry.keys())}"
            )

        loader = self._loader_registry[ext]
        return loader.load(file_path)
