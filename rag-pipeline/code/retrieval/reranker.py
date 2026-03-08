"""
# ============================================================================
# Re-Ranker — Cross-Encoder Based Relevance Scoring
# ============================================================================
#
# The re-ranker is the "quality upgrade" for retrieval. It takes the
# candidates from vector search and re-scores them using a cross-encoder
# model that's MUCH better at judging relevance.
#
# WHY RE-RANKING IS SO MUCH BETTER:
# -----------------------------------
#
#   BI-ENCODER (vector search):
#   ┌──────────────┐     ┌──────────────┐
#   │   Question   │     │    Chunk     │
#   │     ↓        │     │     ↓        │
#   │  [Encoder]   │     │  [Encoder]   │    ← Encoded SEPARATELY
#   │     ↓        │     │     ↓        │
#   │  vector_q    │     │  vector_c    │
#   └──────┬───────┘     └──────┬───────┘
#          │                     │
#          └──── cosine ────────┘    ← Compare vectors AFTER encoding
#                similarity
#
#   CROSS-ENCODER (re-ranker):
#   ┌─────────────────────────────┐
#   │  Question + Chunk           │
#   │  [together as one input]    │    ← Encoded TOGETHER
#   │          ↓                  │
#   │    [Cross-Encoder]          │
#   │          ↓                  │
#   │    relevance score (0→1)    │    ← Direct relevance judgment
#   └─────────────────────────────┘
#
# The cross-encoder "sees" both the question and the chunk at the same
# time, so it can reason about whether the chunk actually ANSWERS the
# question — not just whether they're topically similar.
#
# EXAMPLE OF THE DIFFERENCE:
# ---------------------------
#   Question: "What happens AFTER pre-training?"
#
#   Bi-encoder thinks: "pre-training" is in both → HIGH similarity
#     → Returns chunk about pre-training (wrong!)
#
#   Cross-encoder thinks: "The question asks about what comes AFTER,
#     so I need a chunk that discusses the next phase..."
#     → Returns chunk about instruction fine-tuning (correct! ✅)
#
# TRADE-OFF:
# ----------
# Cross-encoders are SLOW (process each pair individually) but ACCURATE.
# That's why we use a two-stage approach:
#   Stage 1: Vector search → fast, get 20 candidates
#   Stage 2: Cross-encoder → slow, re-rank to find best 5
#
# ============================================================================
"""

from ..vectorstore.base import SearchResult


# ============================================================================
# ReRanker — Cross-Encoder Re-Ranking
# ============================================================================

class ReRanker:
    """
    Re-ranks retrieval results using a cross-encoder model.

    -----------------------------------------------------------------------
    Takes candidate chunks from vector search and re-scores them
    for actual relevance to the question.

    Args:
        model_name: Which cross-encoder model to use.
                    Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
                    (small, fast, trained on MS MARCO ranking dataset)

    Example:
        reranker = ReRanker()

        # Get 20 candidates from vector search
        candidates = retriever.retrieve(question, top_k=20)

        # Re-rank to find the 5 most relevant
        best = reranker.rerank(question, candidates, top_k=5)
        # best[0] is now the MOST RELEVANT chunk, not just most similar
    -----------------------------------------------------------------------
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None     # Lazy loading (same pattern as embedding model)

    def _load_model(self):
        """
        Load the cross-encoder model (lazy initialization).

        -----------------------------------------------------------------------
        Downloads the model on first use (~80MB) and caches it.
        The cross-encoder is from the sentence-transformers library.
        -----------------------------------------------------------------------
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install with: pip install sentence-transformers"
            )

        print(f"🔄 Loading re-ranker model: {self.model_name}...")
        self._model = CrossEncoder(self.model_name)
        print(f"✅ Re-ranker loaded: {self.model_name}")

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Re-rank search results using the cross-encoder.

        -----------------------------------------------------------------------
        Algorithm:
            1. For each candidate, create a (query, chunk_text) pair
            2. Feed ALL pairs through the cross-encoder
            3. Get a relevance score for each pair (0 to 1)
            4. Sort by relevance score (highest first)
            5. Return the top_k most relevant results

        Args:
            query:   The user's question.
            results: Candidate SearchResults from vector search.
            top_k:   How many results to keep after re-ranking.

        Returns:
            The top_k most RELEVANT results (not just most similar).

        Performance:
            Re-ranking 20 candidates takes ~50-200ms on CPU.
            This is much slower than vector search (~5ms) but much
            more accurate for relevance.
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Edge cases: nothing to re-rank or very few candidates
        # -----------------------------------------------------------------------
        if not results:
            return []
        if len(results) <= top_k:
            return results

        self._load_model()

        # -----------------------------------------------------------------------
        # Step 1: Create (query, chunk_text) pairs for the cross-encoder
        #
        # The cross-encoder processes PAIRS — it needs to see the question
        # and the chunk together to judge relevance.
        #
        # Example:
        #   pairs = [
        #     ("What is pre-training?", "Pre-training is the first phase..."),
        #     ("What is pre-training?", "The model uses attention mechanisms..."),
        #     ("What is pre-training?", "Pre-training requires massive data..."),
        #   ]
        # -----------------------------------------------------------------------
        pairs = [(query, result.chunk.content) for result in results]

        # -----------------------------------------------------------------------
        # Step 2: Score all pairs with the cross-encoder
        #
        # model.predict() processes all pairs and returns a score for each:
        #   scores = [0.92, 0.15, 0.88]
        #   → Pair 0 is highly relevant (0.92)
        #   → Pair 1 is not relevant (0.15)
        #   → Pair 2 is quite relevant (0.88)
        # -----------------------------------------------------------------------
        scores = self._model.predict(pairs)

        # -----------------------------------------------------------------------
        # Step 3: Update results with cross-encoder scores and sort
        # -----------------------------------------------------------------------
        for result, score in zip(results, scores):
            result.score = float(score)    # Replace vector similarity with relevance score

        # Sort by cross-encoder score (highest = most relevant)
        results.sort(key=lambda r: r.score, reverse=True)

        # -----------------------------------------------------------------------
        # Step 4: Update ranks and return top_k
        # -----------------------------------------------------------------------
        top_results = results[:top_k]
        for i, result in enumerate(top_results):
            result.rank = i

        return top_results
