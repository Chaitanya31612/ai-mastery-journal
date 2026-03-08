"""
# ============================================================================
# JSON Document Loader
# ============================================================================
#
# Loads JSON files by extracting text content from configurable fields.
#
# JSON is a structured format, and RAG needs flat text. The challenge is:
# which fields contain the "content" we want to search? This loader lets
# you specify exactly which fields to extract.
#
# SUPPORTED STRUCTURES:
# ---------------------
# 1. A single JSON object     → {"title": "...", "body": "..."}
# 2. An array of JSON objects  → [{"title": "...", "body": "..."}, ...]
# 3. Nested objects            → {"article": {"content": "..."}}
#
# EXAMPLE:
# --------
#     # Data file (articles.json):
#     # [
#     #     {"title": "RAG Intro", "body": "RAG stands for...", "author": "Alice"},
#     #     {"title": "Embeddings", "body": "Vectors are...", "author": "Bob"}
#     # ]
#
#     loader = JSONLoader(content_fields=["title", "body"])
#     docs = loader.load("articles.json")
#     docs[0].content    # "title: RAG Intro\nbody: RAG stands for..."
#     docs[1].content    # "title: Embeddings\nbody: Vectors are..."
#
# ============================================================================
"""

import os
import json
from .base import DocumentLoader, Document


# ============================================================================
# JSONLoader — Structured Data Loader
# ============================================================================

class JSONLoader(DocumentLoader):
    """
    Loads JSON files by extracting specified text fields.

    -----------------------------------------------------------------------
    Converts JSON objects into text Documents by pulling out the fields
    you care about and combining them into searchable text.

    Supports: .json files

    Configuration:
        - content_fields: Which JSON keys contain the text to index.
                          If None, converts the entire object to text.
        - jq_path:        Optional JSON path to reach the data array.
                          Example: "results.items" for {"results": {"items": [...]}}

    Example:
        # Simple: extract "title" and "content" fields
        loader = JSONLoader(content_fields=["title", "content"])
        docs = loader.load("data.json")

        # With nested path: data is under "response.articles"
        loader = JSONLoader(content_fields=["title", "body"], jq_path="response.articles")
        docs = loader.load("api_response.json")
    -----------------------------------------------------------------------
    """

    # -----------------------------------------------------------------------
    # File extensions this loader handles
    # -----------------------------------------------------------------------
    supported_extensions = [".json"]

    def __init__(
        self,
        content_fields: list[str] | None = None,
        jq_path: str | None = None,
        encoding: str = "utf-8",
    ):
        """
        -----------------------------------------------------------------------
        Args:
            content_fields: List of JSON keys to extract as text content.
                            Example: ["title", "body", "summary"]
                            If None, the entire object is converted to text.

            jq_path: Dot-separated path to reach the data in nested JSON.
                     Example: "data.articles" navigates to json["data"]["articles"]
                     If None, the root of the JSON is used.

            encoding: File encoding (default: UTF-8).
        -----------------------------------------------------------------------
        """
        self.content_fields = content_fields
        self.jq_path = jq_path
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        """
        Load a JSON file and return Document(s).

        -----------------------------------------------------------------------
        Handles two JSON structures:
            - A single object  → Returns 1 Document
            - An array of objects → Returns 1 Document per object

        If jq_path is set, navigates to that path first:
            JSON: {"response": {"items": [{"text": "hello"}, ...]}}
            jq_path: "response.items"
            Result: Documents from the "items" array

        Example:
            loader = JSONLoader(content_fields=["question", "answer"])
            docs = loader.load("faq.json")
            # faq.json = [{"question": "What is RAG?", "answer": "RAG is..."}]
            # docs[0].content = "question: What is RAG?\nanswer: RAG is..."
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Validate and read the JSON file
        # -----------------------------------------------------------------------
        if not os.path.exists(source):
            raise FileNotFoundError(f"JSON file not found: {source}")

        with open(source, "r", encoding=self.encoding) as f:
            data = json.load(f)

        # -----------------------------------------------------------------------
        # Step 2: Navigate to the data using jq_path (if specified)
        #
        # Example: jq_path = "response.articles"
        #   data = {"response": {"articles": [{...}, {...}]}}
        #   After navigation: data = [{...}, {...}]
        # -----------------------------------------------------------------------
        if self.jq_path:
            for key in self.jq_path.split("."):
                if isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    raise ValueError(
                        f"JSON path '{self.jq_path}' not found. "
                        f"Key '{key}' doesn't exist at this level."
                    )

        # -----------------------------------------------------------------------
        # Step 3: Normalize to a list (handle both single object and array)
        # -----------------------------------------------------------------------
        if isinstance(data, dict):
            items = [data]       # Single object → wrap in a list
        elif isinstance(data, list):
            items = data         # Already a list
        else:
            # If it's a string or number, wrap it as content
            return [Document(
                content=str(data),
                metadata=self._build_metadata(source),
            )]

        # -----------------------------------------------------------------------
        # Step 4: Convert each item to a Document
        # -----------------------------------------------------------------------
        documents = []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            content = self._extract_content(item)

            if not content.strip():
                continue

            metadata = self._build_metadata(
                source,
                item_index=index,
                total_items=len(items),
                fields_extracted=self.content_fields or list(item.keys()),
            )

            documents.append(Document(content=content, metadata=metadata))

        return documents

    def _extract_content(self, item: dict) -> str:
        """
        Extract text content from a JSON object.

        -----------------------------------------------------------------------
        If content_fields is specified:
            Pull only those fields → "field_name: field_value" per line.

        If content_fields is None:
            Convert the entire object to a readable format.

        Example (with content_fields=["title", "body"]):
            Input:  {"title": "RAG", "body": "RAG stands for...", "id": 123}
            Output: "title: RAG\nbody: RAG stands for..."
            (Note: "id" is excluded because it's not in content_fields)

        Example (without content_fields — extract all):
            Input:  {"title": "RAG", "body": "RAG stands for...", "id": 123}
            Output: "title: RAG\nbody: RAG stands for...\nid: 123"
        -----------------------------------------------------------------------
        """

        if self.content_fields:
            # --- Extract only specified fields ---
            parts = []
            for field in self.content_fields:
                value = item.get(field)
                if value is not None:
                    # Convert non-strings (lists, dicts) to their string representation
                    str_value = value if isinstance(value, str) else json.dumps(value)
                    parts.append(f"{field}: {str_value}")
            return "\n".join(parts)
        else:
            # --- Extract ALL fields ---
            parts = []
            for key, value in item.items():
                str_value = value if isinstance(value, str) else json.dumps(value)
                parts.append(f"{key}: {str_value}")
            return "\n".join(parts)
