# Retrieval Mechanisms

> **Prerequisites**: [02 — Vector Embeddings](./02-vector-embeddings-explained.md), [03 — Chunking Strategies](./03-chunking-strategies.md)
> **Reading time**: ~12 minutes

---

## The Goal of Retrieval

You have a vector database full of document chunks. A user asks a question. The retrieval system's job is to find the **most relevant** chunks—not just the most similar ones.

This distinction matters:
- **Similarity** = "these texts are about the same topic"
- **Relevance** = "this text actually answers the question"

Good retrieval is the difference between a RAG system that gives great answers and one that hallucinates despite having the right data.

---

## Method 1: Vector Similarity Search (Nearest Neighbors)

The foundational retrieval method. Embed the question, find the closest chunk vectors.

### How It Works

```
┌─────────────────────────────────────────────────┐
│  Vector Space (simplified to 2D)                │
│                                                  │
│          • chunk_7                                │
│    • chunk_2                                     │
│              ★ QUERY              • chunk_9      │
│         • chunk_4                                │
│    • chunk_1           • chunk_8                 │
│                  • chunk_5                       │
│   • chunk_3                     • chunk_6        │
│                                                  │
│   Nearest to query: chunk_4, chunk_2, chunk_1    │
└─────────────────────────────────────────────────┘
```

### k-Nearest Neighbors (kNN) vs Approximate Nearest Neighbors (ANN)

| Type | How | Speed | Accuracy |
|------|-----|-------|----------|
| **kNN (exact)** | Compare query to EVERY vector | O(n) — slow for millions | 100% accurate |
| **ANN (approximate)** | Use index structures (HNSW, IVF) | O(log n) — fast | ~98-99% accurate |

**ChromaDB uses HNSW** (Hierarchical Navigable Small World) — an ANN algorithm that's fast and accurate. For our learning purposes, you don't need to worry about this. Just know that when you call `collection.query()`, it's doing efficient approximate search.

### Basic Usage with ChromaDB

```python
# Search for the top 5 most similar chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=5,                    # Top-K
    where={"source": "my_doc.md"},  # Optional metadata filter
)

# Results contain:
# - documents: the chunk texts
# - distances: similarity scores
# - metadatas: chunk metadata
# - ids: chunk identifiers
```

### Tuning Top-K

| Top-K | Effect |
|-------|--------|
| K=1 | Only the single best match. High precision, may miss relevant info |
| K=3 | Good balance for most queries |
| K=5 | More context, but more noise too |
| K=10+ | For broad questions or when quality varies |

**Rule of thumb**: Start with K=3-5. If answers miss context, increase. If answers include irrelevant info, decrease.

---

## Method 2: Keyword Search (BM25 / TF-IDF)

The "old school" information retrieval approach. Matches based on **exact word overlap**.

### How BM25 Works (Simplified)
BM25 scores documents based on:
1. **Term Frequency (TF)**: How often does the query word appear in this chunk?
2. **Inverse Document Frequency (IDF)**: How rare is this word across ALL chunks? Rare words = more informative
3. **Document Length**: Normalizes for chunk length

```
Query: "transformer attention mechanism"

Chunk A: "The transformer uses attention mechanisms to focus on..."
→ Contains "transformer" (1x), "attention" (1x), "mechanism" (1x)
→ High BM25 score ✅

Chunk B: "Attention is important in many aspects of life..."
→ Contains "attention" (1x), but not in the right context
→ Medium score (IDF of "attention" is low because it's common)

Chunk C: "The model processes sequences using various techniques"
→ No matching terms
→ Score = 0
```

### When Keyword Search Wins

```
Question: "What is RLHF?"

Vector search might return: "The model is trained with human feedback to improve..."
→ Semantically similar but doesn't mention "RLHF" ❌

BM25 would return: "RLHF (Reinforcement Learning from Human Feedback) is..."
→ Exact term match finds the definition ✅
```

**Acronyms, proper nouns, and technical terms** are where keyword search shines.

---

## Method 3: Hybrid Search (Vector + Keyword)

Combines the best of both worlds.

### How It Works

```
     User Question
         │
    ┌────┴────┐
    ▼         ▼
 Vector    Keyword
 Search    Search
 (BM25)    (kNN)
    │         │
    │  Results │
    └────┬────┘
         │
    ┌────▼────┐
    │  Merge  │   ← Reciprocal Rank Fusion (RRF)
    │  & Rank │      or weighted combination
    └────┬────┘
         │
    Final Results
```

### Reciprocal Rank Fusion (RRF)
A popular method to combine results from multiple search methods:

```
RRF_score = Σ (1 / (k + rank_i))

k = 60 (constant, prevents high-ranked items from dominating)
rank_i = rank in each search result list

Example:
Chunk A: rank 1 in vector, rank 3 in BM25
  → 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

Chunk B: rank 5 in vector, rank 1 in BM25
  → 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

Chunk A wins slightly (consistently good in both)
```

### Pros
- Catches both semantic matches AND exact keyword matches
- More robust than either method alone
- Handles edge cases better (acronyms, technical terms)

### Our Pipeline
We implement vector search as the primary method. Hybrid search can be added as an enhancement since ChromaDB supports metadata-based filtering which enables basic keyword matching.

---

## Method 4: Re-Ranking

The secret weapon for retrieval quality. A **two-stage** approach:

### Stage 1: Retrieve (Fast, Broad)
Get top-K candidates using vector search (fast but approximate).

### Stage 2: Re-rank (Slow, Precise)
Use a more powerful model to re-score each candidate for actual relevance.

```
     Question
         │
         ▼
    ┌─────────┐
    │ Vector   │    Stage 1: Get 20 candidates (fast)
    │ Search   │
    │ top-20   │
    └────┬────┘
         │
   20 candidates
         │
         ▼
    ┌─────────┐
    │ Cross-   │    Stage 2: Score each for relevance (slower, precise)
    │ Encoder  │    Using question-chunk PAIRS
    │ Reranker │
    └────┬────┘
         │
   Reranked top-5
```

### Why Re-ranking Works Better

**Bi-encoder** (embedding model): Encodes question and chunk **independently**, then compares vectors.
```
Question ──▶ [encoder] ──▶ vector_q ─┐
                                      ├─ cosine_similarity
Chunk    ──▶ [encoder] ──▶ vector_c ─┘
```

**Cross-encoder** (reranker): Processes question and chunk **together**, enabling direct comparison.
```
[Question + Chunk] ──▶ [encoder] ──▶ relevance_score (0-1)
```

The cross-encoder "sees" both texts simultaneously, so it can reason about whether the chunk actually *answers* the question — not just whether they're topically similar.

### Example

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score each candidate
pairs = [(question, chunk.text) for chunk in candidates]
scores = reranker.predict(pairs)

# Sort by relevance score
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

# Take top 3
final_chunks = [c for c, s in ranked[:3]]
```

### The Trade-off
| Aspect | Bi-encoder (vector search) | Cross-encoder (reranker) |
|--------|---------------------------|-------------------------|
| Speed | Fast (search millions in ms) | Slow (score each pair individually) |
| Quality | Good similarity matching | Excellent relevance scoring |
| Use case | First-pass retrieval | Second-pass refinement |

**That's why we do both**: Fast vector search to get 20 candidates, then slow re-ranking to pick the best 3-5.

---

## Vector Databases Compared

Where do you store all those vectors? Here's how the options compare:

| Database | Type | Setup | Scalability | Best For |
|----------|------|-------|-------------|----------|
| **ChromaDB** ✅ | Embedded | `pip install` | 100K-1M vectors | Learning, prototypes, small-medium apps |
| FAISS | Library | `pip install` | 1M-100M vectors | When you need raw speed |
| Pinecone | Cloud | API key | Unlimited | Production, managed service |
| Weaviate | Server | Docker | 1M+ vectors | Enterprise, hybrid search built-in |
| Qdrant | Server | Docker | 1M+ vectors | Performance-critical production |
| pgvector | Extension | PostgreSQL | 1M+ vectors | Already using PostgreSQL |

### Why ChromaDB for Us
```python
import chromadb

# That's it. No server, no Docker, no API key.
client = chromadb.PersistentClient(path="./data/vectordb")
collection = client.get_or_create_collection("my_docs")

# Add vectors
collection.add(
    documents=["chunk text 1", "chunk text 2"],
    metadatas=[{"source": "doc1.md"}, {"source": "doc2.md"}],
    ids=["id1", "id2"]
)

# Search
results = collection.query(
    query_texts=["my question"],  # ChromaDB can embed for you too!
    n_results=3
)
```

**ChromaDB is perfect for learning** because:
- ✅ Install with pip, no infrastructure
- ✅ Persistent — survives restarts
- ✅ Built-in embedding support
- ✅ Metadata filtering
- ✅ Pythonic API

---

## Putting It All Together: Our Retrieval Flow

```
                       User Question
                            │
                            ▼
                     ┌──────────────┐
                     │   Embed      │
                     │   Question   │
                     │   (MiniLM)   │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Vector     │
                     │   Search     │   Top-20 candidates
                     │   (ChromaDB) │
                     └──────┬───────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Re-Rank    │
                     │   (Cross-    │   Top-5 most relevant
                     │   Encoder)   │
                     └──────┬───────┘
                            │
                            ▼
                     Retrieved Chunks
                     (ready for LLM)
```

---

## Key Takeaways

1. **Vector similarity search** is the foundation — fast but approximate
2. **Keyword search (BM25)** catches exact terms that embeddings miss
3. **Hybrid search** combines both for better coverage
4. **Re-ranking with cross-encoders** dramatically improves relevance
5. **Two-stage retrieval** (fast broad search → precise re-ranking) is the best practice
6. **ChromaDB** is the perfect starting vector DB — zero infrastructure, Python-native

---

## What's Next?

Now you know all the building blocks. Let's see how they fit together into a complete system design.

→ **[05 — RAG Pipeline Architecture](./05-rag-pipeline-architecture.md)**
