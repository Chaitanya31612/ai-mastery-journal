"""
# ============================================================================
# Sentence Transformer Embedding Provider
# ============================================================================
#
# This is our PRIMARY embedding engine. It uses the `sentence-transformers`
# library to generate high-quality text embeddings LOCALLY on your machine.
#
# WHY SENTENCE-TRANSFORMERS:
# ---------------------------
# - FREE: No API key, no usage limits, no costs
# - LOCAL: Runs on your CPU/GPU, no internet required
# - FAST: all-MiniLM-L6-v2 is optimized for speed
# - GOOD QUALITY: Excellent for semantic search (384 dimensions)
# - WIDELY USED: Most RAG tutorials use this model
#
# HOW IT WORKS UNDER THE HOOD:
# ----------------------------
#     Input text ──▶ Tokenizer ──▶ Tokens ──▶ Transformer ──▶ Pooling ──▶ Vector
#
# 1. TOKENIZER: Splits text into subword tokens
#    "Machine learning" → ["Machine", "learn", "##ing"]
#
# 2. TRANSFORMER: Processes tokens through attention layers
#    (This is the same architecture from your LLM studies)
#
# 3. POOLING: Combines all token embeddings into ONE vector
#    Using "mean pooling" — average of all token embeddings
#
# The result is a 384-dimensional vector that captures the
# semantic meaning of the entire input text.
#
# FIRST RUN NOTE:
# ----------------
# The first time you use this, it downloads the model (~80MB).
# After that, it loads from cache (usually ~/.cache/torch/sentence_transformers/).
#
# ============================================================================
"""

import numpy as np
from .base import EmbeddingProvider


# ============================================================================
# SentenceTransformerEmbedder — Local Embedding Model
# ============================================================================

class SentenceTransformerEmbedder(EmbeddingProvider):
    """
    Generates embeddings using a local sentence-transformers model.

    -----------------------------------------------------------------------
    This is the DEFAULT embedding provider for our RAG pipeline.
    It runs entirely on your machine — no API keys or internet needed
    (after the initial model download).

    Args:
        model_name: Name of the sentence-transformers model to use.
                    Default: "all-MiniLM-L6-v2" (384 dims, very fast)
                    Alternative: "all-mpnet-base-v2" (768 dims, better quality)
        batch_size: How many texts to embed at once (default: 32).
                    Higher = faster but more memory.
        device:     "cpu", "cuda", or None (auto-detect).

    Example:
        embedder = SentenceTransformerEmbedder()  # Uses default model

        # Embed documents
        vectors = embedder.embed(["Hello world", "Machine learning is great"])
        print(vectors.shape)  # (2, 384)

        # Embed a question
        q_vector = embedder.embed_query("What is ML?")
        print(q_vector.shape)  # (384,)
    -----------------------------------------------------------------------
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimension = None

        # -----------------------------------------------------------------------
        # Lazy loading: we don't load the model until it's actually used.
        # This makes imports fast and avoids loading a 80MB model into
        # memory if you're just testing other parts of the pipeline.
        # -----------------------------------------------------------------------
        self._model = None
        self._device = device

    def _load_model(self):
        """
        Load the sentence-transformers model (lazy initialization).

        -----------------------------------------------------------------------
        This is called once, on the first call to embed() or embed_query().
        After that, the model stays in memory for fast subsequent calls.

        The model is downloaded from HuggingFace Hub on first use (~80MB)
        and cached locally for future runs.
        -----------------------------------------------------------------------
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

        print(f"🔄 Loading embedding model: {self.model_name}...")

        # -----------------------------------------------------------------------
        # SentenceTransformer handles:
        #   - Downloading the model (first time only)
        #   - Loading onto CPU/GPU
        #   - Tokenization, encoding, and pooling
        # -----------------------------------------------------------------------
        self._model = SentenceTransformer(self.model_name, device=self._device)
        self._dimension = self._model.get_sentence_embedding_dimension()

        print(f"✅ Model loaded: {self.model_name} ({self._dimension} dimensions)")

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        -----------------------------------------------------------------------
        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).

        Performance note:
            Batch processing is MUCH faster than one-at-a-time:
              embed(["text1", "text2", ...])  ← Fast (batched, parallel)
              embed(["text1"]) × N times      ← Slow (sequential)

        Example:
            chunks = ["First chunk about ML...", "Second chunk about NLP..."]
            vectors = embedder.embed(chunks)
            # vectors[0] = [0.12, -0.34, ...] (384 numbers)
            # vectors[1] = [0.45, -0.67, ...] (384 numbers)
        -----------------------------------------------------------------------
        """
        self._load_model()

        # -----------------------------------------------------------------------
        # model.encode() handles:
        #   - Tokenization (text → tokens)
        #   - Transformer forward pass
        #   - Pooling (token embeddings → sentence embedding)
        #   - Batching (processes batch_size texts at a time)
        #
        # convert_to_numpy=True ensures we get numpy arrays (not PyTorch tensors)
        # normalize_embeddings=True L2-normalizes vectors so cosine similarity
        # equals dot product (slight speedup for search)
        # -----------------------------------------------------------------------
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,   # Only show bar for large batches
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text into a vector.

        -----------------------------------------------------------------------
        For sentence-transformers, there's no difference between document
        and query embedding — they use the same model. So this is just
        a convenience wrapper around embed() for a single text.

        Note: Some API-based models (like Cohere) DO use different
        models/prefixes for queries vs documents. That's why we have
        separate methods in the interface.

        Returns:
            numpy array of shape (dimension,).
        -----------------------------------------------------------------------
        """
        self._load_model()

        # -----------------------------------------------------------------------
        # embed() returns shape (1, 384), but for a single query
        # we want shape (384,) — a flat 1D array
        # -----------------------------------------------------------------------
        result = self.embed([text])
        return result[0]

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension (e.g., 384 for MiniLM).

        -----------------------------------------------------------------------
        Triggers model loading if not already loaded, since we need
        the model to know its dimension.
        -----------------------------------------------------------------------
        """
        self._load_model()
        return self._dimension
