"""
# ============================================================================
# CSV Document Loader
# ============================================================================
#
# Loads CSV (Comma-Separated Values) files into Document objects.
#
# CSV files are tricky for RAG because they're TABULAR data, not narratives.
# A question like "What were the Q3 2024 sales?" needs the system to find
# the right ROW and COLUMN — not the most "similar" paragraph.
#
# STRATEGY:
# ---------
# We convert each row into a natural language string. This makes the tabular
# data searchable via embeddings.
#
# Example CSV:
#     Name, Age, City
#     Alice, 30, Mumbai
#     Bob, 25, Delhi
#
# Becomes:
#     Document: "Name: Alice | Age: 30 | City: Mumbai"
#     Document: "Name: Bob | Age: 25 | City: Delhi"
#
# WHY THIS APPROACH:
# ------------------
# - Each row is self-contained (good for embedding)
# - Column names provide context (the embedding knows "Age" is relevant)
# - The separator "|" keeps fields visually distinct
#
# EXAMPLE:
# --------
#     loader = CSVLoader()
#     docs = loader.load("employees.csv")
#     docs[0].content       # "Name: Alice | Age: 30 | City: Mumbai"
#     docs[0].metadata      # {"source": "employees.csv", "row_index": 0, ...}
#
# ============================================================================
"""

import os
import csv
from .base import DocumentLoader, Document


# ============================================================================
# CSVLoader — Tabular Data Loader
# ============================================================================

class CSVLoader(DocumentLoader):
    """
    Loads CSV files by converting each row into a text Document.

    -----------------------------------------------------------------------
    Each row becomes a Document with column names prefixed to values.
    This makes tabular data work with embedding-based search.

    Supports: .csv files

    Configuration:
        - delimiter: Column separator (default: auto-detect via csv.Sniffer)
        - join_str:  How to join column-value pairs (default: " | ")
        - encoding:  File encoding (default: UTF-8)

    Example:
        loader = CSVLoader(join_str=" | ")
        docs = loader.load("sales.csv")
        # Each doc = one row: "Product: Widget | Revenue: 5000 | Quarter: Q3"
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # File extensions this loader handles
    # -----------------------------------------------------------------------
    supported_extensions = [".csv"]

    def __init__(
        self,
        delimiter: str = ",",
        join_str: str = " | ",
        encoding: str = "utf-8",
    ):
        """
        -----------------------------------------------------------------------
        Args:
            delimiter: Column separator character.
                       "," for CSV (default), "\\t" for TSV (tab-separated).
            join_str:  String used to join "column: value" pairs.
                       Default " | " gives: "Name: Alice | Age: 30"
            encoding:  File encoding (default: UTF-8).
        -----------------------------------------------------------------------
        """
        self.delimiter = delimiter
        self.join_str = join_str
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        """
        Load a CSV file and return one Document per row.

        -----------------------------------------------------------------------
        How it works:
            1. Read the CSV header row to get column names
            2. For each data row, create "ColumnName: Value" pairs
            3. Join pairs with the join_str separator
            4. Wrap in a Document with row-level metadata

        Skips rows that are empty or contain only whitespace.

        Returns:
            List of Documents, one per non-empty row.

        Example:
            Given CSV:
                product,price,category
                Widget,29.99,Hardware
                Gadget,49.99,Electronics

            Returns:
                [Document(content="product: Widget | price: 29.99 | category: Hardware",
                          metadata={"row_index": 0, "column_names": [...], ...}),
                 Document(content="product: Gadget | price: 49.99 | category: Electronics",
                          metadata={"row_index": 1, ...})]
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Validate the file exists
        # -----------------------------------------------------------------------
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV file not found: {source}")

        documents = []

        # -----------------------------------------------------------------------
        # Step 2: Open and parse the CSV
        # -----------------------------------------------------------------------
        with open(source, "r", encoding=self.encoding) as f:
            # csv.DictReader automatically uses the first row as column names
            # and returns each subsequent row as a dict: {"col1": "val1", ...}
            reader = csv.DictReader(f, delimiter=self.delimiter)
            column_names = reader.fieldnames or []

            # ---------------------------------------------------------------
            # Step 3: Convert each row to a natural language string
            # ---------------------------------------------------------------
            for row_index, row in enumerate(reader):
                # Build "column: value" pairs, skipping empty values
                pairs = []
                for col in column_names:
                    value = (row.get(col) or "").strip()
                    if value:
                        pairs.append(f"{col}: {value}")

                # Skip entirely empty rows
                if not pairs:
                    continue

                # Join pairs: "Name: Alice | Age: 30 | City: Mumbai"
                content = self.join_str.join(pairs)

                # ---------------------------------------------------------------
                # Step 4: Build metadata and create Document
                # ---------------------------------------------------------------
                metadata = self._build_metadata(
                    source,
                    row_index=row_index,
                    column_names=column_names,
                    total_columns=len(column_names),
                )

                documents.append(Document(content=content, metadata=metadata))

        return documents
