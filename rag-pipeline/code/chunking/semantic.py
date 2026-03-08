"""
# ============================================================================
# Semantic Chunking Strategy
# ============================================================================
#
# The most SOPHISTICATED chunking approach. Instead of splitting on
# characters or headers, it uses EMBEDDINGS to detect topic boundaries.
#
# THE INTUITION:
# --------------
# Imagine reading a document. At some point, the topic shifts — you can
# "feel" it even without a header. Semantic chunking detects these shifts
# mathematically by measuring how much the MEANING changes between
# consecutive sentences.
#
# HOW IT WORKS:
# -------------
#     1. Split the text into sentences
#     2. Embed each sentence (using your embedding model)
#     3. Compare each sentence's embedding with the next sentence's embedding
#     4. Where similarity DROPS significantly → that's a topic boundary
#     5. Split the text at those boundaries
#
#     Sentence-by-sentence similarity scores:
#         S1→S2: 0.85  ════════ (same topic)
#         S2→S3: 0.82  ════════ (same topic)
#         S3→S4: 0.35  ══      ← BIG DROP! Topic change → SPLIT HERE
#         S4→S5: 0.79  ════════ (same topic)
#         S5→S6: 0.81  ════════ (same topic)
#         S6→S7: 0.28  ══      ← BIG DROP! Topic change → SPLIT HERE
#         S7→S8: 0.88  ════════ (same topic)
#
# TRADE-OFF:
# ----------
# + Best semantic coherence — each chunk is about ONE topic
# + Adapts to content naturally (no fixed size needed)
# - SLOW — needs to embed every sentence during chunking
# - Requires the embedding model at chunk time (not just at index time)
# - Chunk sizes are unpredictable
#
# WHEN TO USE:
# ------------
# - When retrieval quality is absolutely critical
# - Documents with flowing prose (no clear structure)
# - When you can afford the extra computation during ingestion
#
# ============================================================================
"""

import re
import numpy as np
from .base import ChunkingStrategy, Chunk, Document


# ============================================================================
# SemanticChunker — Embedding-Based Topic Boundary Detection
# ============================================================================

class SemanticChunker(ChunkingStrategy):
    """
    Splits documents at semantic topic boundaries using embeddings.

    -----------------------------------------------------------------------
    Uses an embedding model to detect where topics change in the text,
    and splits at those natural boundaries.

    Args:
        embedding_model: A sentence-transformers model (or any model with
                         an `encode` method that returns numpy arrays).
        threshold_percentile: Topic boundaries are detected where similarity
                              drops below this percentile (default: 25).
                              Lower = fewer splits (bigger chunks).
                              Higher = more splits (smaller chunks).
        min_chunk_size: Minimum characters per chunk (default: 100).
                        Prevents tiny chunks from single sentences.
        max_chunk_size: Maximum characters per chunk (default: 2000).
                        If a semantic chunk exceeds this, force-split it.

    Example:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        chunker = SemanticChunker(embedding_model=model)
        chunks = chunker.chunk(document)
    -----------------------------------------------------------------------
    """

    def __init__(
        self,
        embedding_model=None,
        threshold_percentile: int = 25,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        self.embedding_model = embedding_model
        self.threshold_percentile = threshold_percentile
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a document at semantic topic boundaries.

        -----------------------------------------------------------------------
        Algorithm:
            1. Split text into sentences
            2. Embed all sentences
            3. Calculate similarity between consecutive sentences
            4. Find where similarity drops below the threshold
            5. Group sentences between boundaries into chunks
            6. Merge tiny chunks, split oversized ones

        Falls back to simple sentence grouping if embedding model
        is not available.
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: Split into sentences
        # -----------------------------------------------------------------------
        sentences = self._split_into_sentences(document.content)

        if len(sentences) <= 1:
            return self._build_chunks([document.content], document, "semantic")

        # -----------------------------------------------------------------------
        # Step 2: If no embedding model is provided, fall back to
        # simple sentence grouping (chunks of ~5 sentences each)
        # -----------------------------------------------------------------------
        if self.embedding_model is None:
            return self._fallback_chunk(sentences, document)

        # -----------------------------------------------------------------------
        # Step 3: Embed all sentences
        # -----------------------------------------------------------------------
        embeddings = self.embedding_model.encode(
            sentences,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # -----------------------------------------------------------------------
        # Step 4: Calculate cosine similarity between consecutive sentences
        #
        # For N sentences, we get N-1 similarity scores:
        #   sim[0] = similarity(sentence_0, sentence_1)
        #   sim[1] = similarity(sentence_1, sentence_2)
        #   sim[2] = similarity(sentence_2, sentence_3)
        #   ...
        # -----------------------------------------------------------------------
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # -----------------------------------------------------------------------
        # Step 5: Find topic boundaries
        #
        # A "boundary" is where similarity drops below a threshold.
        # We use a percentile-based threshold so it adapts to the document:
        #   - threshold_percentile=25 means "split where similarity is in
        #     the lowest 25% of all similarity scores"
        # -----------------------------------------------------------------------
        threshold = np.percentile(similarities, self.threshold_percentile)
        boundaries = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

        # -----------------------------------------------------------------------
        # Step 6: Group sentences between boundaries into chunks
        # -----------------------------------------------------------------------
        chunks_text = []
        start = 0

        for boundary in boundaries:
            chunk_text = " ".join(sentences[start:boundary])
            chunks_text.append(chunk_text)
            start = boundary

        # Don't forget the last group
        if start < len(sentences):
            chunk_text = " ".join(sentences[start:])
            chunks_text.append(chunk_text)

        # -----------------------------------------------------------------------
        # Step 7: Post-process — merge tiny chunks, split oversized ones
        # -----------------------------------------------------------------------
        final_texts = self._post_process(chunks_text)

        return self._build_chunks(final_texts, document, "semantic")

    def _split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using regex-based rules.

        -----------------------------------------------------------------------
        We use a simple but effective heuristic:
            Split on periods, question marks, and exclamation marks
            followed by a space and an uppercase letter (or end of string).

        This handles most English text well. For production, you'd use
        a library like spaCy or nltk for sentence tokenization.

        Example:
            "Hello world. This is great! How are you?"
            → ["Hello world.", "This is great!", "How are you?"]
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Split on sentence-ending punctuation followed by space
        # The regex keeps the punctuation with the sentence
        # -----------------------------------------------------------------------
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter out empty strings and whitespace-only strings
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        -----------------------------------------------------------------------
        Formula: cos(A, B) = (A · B) / (|A| × |B|)

        Returns a value between -1 and 1:
            1.0  = identical direction (same meaning)
            0.0  = perpendicular (unrelated)
           -1.0  = opposite direction (opposite meaning)

        Example:
            vec_a = [1, 0, 0]
            vec_b = [1, 0, 0]
            cosine_similarity(vec_a, vec_b) = 1.0  (identical)

            vec_a = [1, 0, 0]
            vec_b = [0, 1, 0]
            cosine_similarity(vec_a, vec_b) = 0.0  (unrelated)
        -----------------------------------------------------------------------
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _post_process(self, chunks: list[str]) -> list[str]:
        """
        Merge tiny chunks and split oversized ones.

        -----------------------------------------------------------------------
        - Chunks smaller than min_chunk_size get merged with their neighbor
        - Chunks larger than max_chunk_size get force-split
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Pass 1: Merge tiny chunks with their previous neighbor
        # -----------------------------------------------------------------------
        merged = []
        for chunk in chunks:
            if merged and len(merged[-1]) < self.min_chunk_size:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        # -----------------------------------------------------------------------
        # Pass 2: Split oversized chunks
        # -----------------------------------------------------------------------
        final = []
        for chunk in merged:
            if len(chunk) <= self.max_chunk_size:
                final.append(chunk)
            else:
                # Force-split at max_chunk_size boundaries
                for i in range(0, len(chunk), self.max_chunk_size):
                    piece = chunk[i:i + self.max_chunk_size]
                    if piece.strip():
                        final.append(piece)

        return final

    def _fallback_chunk(self, sentences: list[str], document: Document) -> list[Chunk]:
        """
        Simple fallback: group sentences into chunks of ~5 each.

        -----------------------------------------------------------------------
        Used when no embedding model is provided. Just groups consecutive
        sentences together without any semantic analysis.
        -----------------------------------------------------------------------
        """
        group_size = 5
        texts = []

        for i in range(0, len(sentences), group_size):
            group = sentences[i:i + group_size]
            texts.append(" ".join(group))

        return self._build_chunks(texts, document, "semantic_fallback")
