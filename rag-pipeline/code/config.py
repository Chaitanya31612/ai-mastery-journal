"""
# ============================================================================
# RAG Pipeline — Central Configuration
# ============================================================================
#
# This module defines ALL the tuneable knobs for the RAG pipeline in one place.
# Instead of scattering magic numbers across files, everything lives here.
#
# HOW IT WORKS:
# ------------
# We use Python's @dataclass to create a configuration object with sensible
# defaults. You can create a config with defaults or override anything:
#
#     config = RAGConfig()                        # All defaults
#     config = RAGConfig(chunk_size=1000)          # Override chunk size
#     config = RAGConfig.from_dict({"chunk_size": 1000})  # From a dictionary
#
# WHY THIS PATTERN:
# -----------------
# - Single source of truth: change a default once, affects the whole pipeline
# - Self-documenting: each field has a comment explaining what it does
# - Type safe: IDE autocomplete and type checking work out of the box
# - Easy to serialize: can be loaded from JSON/YAML config files later
#
# ============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Optional


# ============================================================================
# RAG Pipeline Configuration
# ============================================================================
@dataclass
class RAGConfig:
    """
    Central configuration for the entire RAG pipeline.

    Every component reads its settings from this config object.
    This keeps configuration DRY (Don't Repeat Yourself) and makes
    it easy to experiment with different settings.

    Example:
        config = RAGConfig(chunk_size=300, chunk_overlap=30)
        pipeline = RAGPipeline(config)
        pipeline.ingest("my_docs/")
        answer = pipeline.query("What is RAG?")
    """

    # -----------------------------------------------------------------------
    # Chunking Settings
    # -----------------------------------------------------------------------
    # These control how documents are split into smaller pieces.
    # See docs/03-chunking-strategies.md for the theory.
    # -----------------------------------------------------------------------

    chunk_size: int = 500
    """
    Maximum number of characters per chunk.

    - Smaller (200-300): More precise retrieval, but chunks may lose context.
    - Larger (800-1500): More context preserved, but less precise matching.
    - Sweet spot: 500 chars ≈ 100-125 tokens ≈ about a paragraph.

    Must be smaller than your embedding model's max input length.
    Our model (all-MiniLM-L6-v2) handles up to ~256 word-piece tokens.
    """

    chunk_overlap: int = 50
    """
    Number of characters to overlap between consecutive chunks.

    Overlap prevents losing context at chunk boundaries. For example,
    if a sentence is split across two chunks, the overlap ensures both
    chunks contain the full sentence.

    Rule of thumb: 10-20% of chunk_size.
    Example: chunk_size=500, chunk_overlap=50 means 10% overlap.
    """

    chunking_strategy: str = "recursive"
    """
    Which chunking algorithm to use.

    Options:
    - "fixed"     : Simple character-count splitting. Fast but dumb.
    - "recursive" : Splits on natural boundaries (paragraphs > sentences > words).
                    This is the DEFAULT and works well for most content.
    - "markdown"  : Splits on Markdown headers (# , ## , ### ).
                    Best for .md files — preserves document structure.
    - "semantic"  : Uses embeddings to detect topic boundaries.
                    Best quality but slowest (needs to embed each sentence).
    """

    # -----------------------------------------------------------------------
    # Embedding Settings
    # -----------------------------------------------------------------------
    # These control how text is converted into numeric vectors.
    # See docs/02-vector-embeddings-explained.md for the theory.
    # -----------------------------------------------------------------------

    embedding_model: str = "all-MiniLM-L6-v2"
    """
    Which sentence-transformers model to use for generating embeddings.

    Options (from sentence-transformers library):
    - "all-MiniLM-L6-v2"    : 384 dims, very fast, good quality. OUR DEFAULT.
    - "all-mpnet-base-v2"   : 768 dims, slower, better quality.
    - "all-MiniLM-L12-v2"   : 384 dims, slightly better than L6, slightly slower.

    IMPORTANT: If you change this after indexing documents, you MUST re-index
    everything. You cannot mix embeddings from different models.
    """

    embedding_batch_size: int = 32
    """
    Number of texts to embed in a single batch.

    Higher = faster (GPU parallelization) but uses more memory.
    32 is a safe default that works on most hardware.
    """

    # -----------------------------------------------------------------------
    # Vector Store Settings
    # -----------------------------------------------------------------------
    # These control where and how embeddings are stored.
    # See docs/04-retrieval-mechanisms.md for the theory.
    # -----------------------------------------------------------------------

    vectordb_path: str = "./data/vectordb"
    """
    Directory where ChromaDB stores its data on disk.

    ChromaDB persists automatically to this path, so your indexed
    documents survive Python restarts. Delete this folder to start fresh.
    """

    collection_name: str = "rag_documents"
    """
    Name of the ChromaDB collection (like a "table" in a database).

    You could have multiple collections for different document sets:
    - "rag_documents" for general docs
    - "codebase" for code files
    - "research_papers" for academic PDFs
    """

    # -----------------------------------------------------------------------
    # Retrieval Settings
    # -----------------------------------------------------------------------
    # These control how chunks are found and ranked when a user asks a question.
    # See docs/04-retrieval-mechanisms.md for the theory.
    # -----------------------------------------------------------------------

    search_top_k: int = 20
    """
    Number of candidate chunks to retrieve from the vector store.

    This is the FIRST stage (fast, broad). We cast a wide net to make
    sure we don't miss relevant chunks. These candidates are then
    re-ranked by the cross-encoder for precise relevance scoring.

    Higher = more candidates to re-rank (slower but more thorough).
    """

    rerank_top_k: int = 5
    """
    Number of chunks to keep AFTER re-ranking.

    These are the final chunks that get sent to the LLM as context.
    More chunks = more context but also more noise and higher cost.

    Typical values: 3-5 for focused answers, 5-10 for broad questions.
    """

    use_reranker: bool = True
    """
    Whether to use cross-encoder re-ranking after vector search.

    True  = Two-stage retrieval (vector search → cross-encoder re-rank).
            Better quality but slower (~100ms per candidate).
    False = Just use vector search results directly.
            Faster but less precise.

    Set to False if you need speed over quality, or during development.
    """

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    """
    Which cross-encoder model to use for re-ranking.

    The cross-encoder processes (question, chunk) pairs together,
    enabling much better relevance scoring than bi-encoder similarity.

    "cross-encoder/ms-marco-MiniLM-L-6-v2" is small, fast, and effective.
    """

    similarity_threshold: float = 0.0
    """
    Minimum similarity score to include a chunk in results.

    Range: 0.0 to 1.0 (for cosine similarity)
    - 0.0: Include everything (no filtering)
    - 0.3: Moderate filtering
    - 0.7: Only very similar chunks

    Default is 0.0 (no filtering) because we rely on top-k + re-ranking
    instead. Set higher if you want to exclude clearly irrelevant results.
    """

    # -----------------------------------------------------------------------
    # AI / Generation Settings
    # -----------------------------------------------------------------------
    # These control how the LLM generates answers from retrieved context.
    # Uses the existing AI factory (code/ai/).
    # -----------------------------------------------------------------------

    ai_provider: str = "groq"
    """
    Which AI provider to use for generating answers.

    Options: "groq", "gemini", "ollama"
    These map to your existing AIProvider enum and AIAnalyzerFactory.
    """

    ai_model: Optional[str] = None
    """
    Which model to use (provider-specific).

    None = Use the provider's default model.

    For Groq: "fast" (llama-3.1-8b), "smart" (llama-3.3-70b), "mixtral"
    For Gemini: "flash", "pro"
    For Ollama: model name string
    """

    max_context_tokens: int = 3000
    """
    Maximum number of tokens to use for retrieved context in the prompt.

    This prevents exceeding the LLM's context window. The remaining
    tokens are used for the system prompt, question, and generated answer.

    Rough formula:
      LLM context window = system_prompt + retrieved_context + question + answer
      4096 (Groq default)  ≈  500         + 3000              + 100      + 496
    """

    temperature: float = 0.1
    """
    LLM temperature for answer generation.

    - 0.0: Deterministic, always picks the most likely word.
    - 0.1: Nearly deterministic — good for factual RAG answers.
    - 0.7: Creative — good for brainstorming, bad for factual retrieval.
    - 1.0: Maximum randomness.

    For RAG, keep this LOW (0.0-0.2) because we want factual, grounded answers.
    """

    # -----------------------------------------------------------------------
    # Convenience Methods
    # -----------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        """
        Create a RAGConfig from a dictionary (e.g., loaded from JSON).
        Ignores unknown keys so you can pass partial overrides.

        Example:
            config = RAGConfig.from_dict({"chunk_size": 1000, "use_reranker": False})
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """
        Convert config to a plain dictionary.

        Example:
            config = RAGConfig(chunk_size=300)
            print(config.to_dict())
            # {"chunk_size": 300, "chunk_overlap": 50, ...}
        """
        from dataclasses import asdict
        return asdict(self)
