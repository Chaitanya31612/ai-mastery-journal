# Vector Embeddings Explained

> **Prerequisites**: [01 — What is RAG?](./01-what-is-rag.md)
> **Reading time**: ~15 minutes

---

## The Core Idea

You already know from your LLM studies that text can be converted into numeric representations — this is exactly what **word embeddings** do. Your LLM overview mentioned:

> *"We can take a sentence and turn it into a sequence of numeric inputs, i.e., the word embeddings, which contain semantic and syntactic meaning."*

In the context of RAG, we take this concept further. Instead of embedding individual *words*, we embed entire *sentences* or *paragraphs* into a single vector. This allows us to compare the **meaning** of a user's question with the **meaning** of document chunks.

---

## From Words to Vectors

### What Is a Vector?

A vector is simply an ordered list of numbers. In the context of embeddings:

```
"The cat sat on the mat" → [0.12, -0.34, 0.56, 0.78, -0.91, ...]
                            ↑
                            A single vector with 384-1536 dimensions
```

Each number (dimension) captures some aspect of the meaning. You never look at individual dimensions — they only make sense as a whole.

### Why This Works

The magic is that **similar meanings produce similar vectors**:

```
"How do neural networks learn?"  → [0.45, -0.23, 0.67, ...]  ─┐
"What is the training process     → [0.43, -0.21, 0.65, ...]  ─┤ Close together!
 for deep learning models?"                                     │
                                                                │
"Best recipe for chocolate cake"  → [-0.82, 0.91, -0.15, ...] ─┘ Far apart!
```

This is the foundation of **semantic search** — searching by meaning rather than by keyword matching.

---

## How Embedding Models Work

### The Architecture

Modern embedding models are based on **transformer** architectures (the same "T" in GPT that you studied). But instead of generating the next word, they're trained to output a **fixed-size vector** that captures the meaning of the entire input.

```
┌────────────────────────────────────────────────┐
│              Embedding Model                    │
│                                                 │
│  Input text ──▶ Transformer ──▶ Pooling ──▶ Vector
│                  Encoder        Layer       [384 dims]
│                                                 │
│  "What is RAG?" ──────────────────────▶ [0.12, -0.34, ...]
└────────────────────────────────────────────────┘
```

### Training Objective

Embedding models are trained on pairs of texts that are related:

- Question + its answer
- Sentence + its paraphrase
- Title + its article

The training teaches the model: "these pairs should have similar vectors, while unrelated texts should have distant vectors."

### Popular Embedding Models

| Model                         | Dimensions | Speed        | Quality   | Cost            | Notes                                            |
| ----------------------------- | ---------- | ------------ | --------- | --------------- | ------------------------------------------------ |
| **all-MiniLM-L6-v2**    | 384        | ⚡ Very fast | Good      | Free (local)    | **Our choice** — best speed/quality ratio |
| all-mpnet-base-v2             | 768        | Fast         | Better    | Free (local)    | More accurate, 2x slower                         |
| OpenAI text-embedding-3-small | 1536       | Fast (API)   | Great     | $0.02/1M tokens | Best for production                              |
| OpenAI text-embedding-3-large | 3072       | Fast (API)   | Excellent | $0.13/1M tokens | Maximum quality                                  |
| Cohere embed-v3               | 1024       | Fast (API)   | Great     | $0.10/1M tokens | Good multilingual                                |

**We're using `all-MiniLM-L6-v2`** from the `sentence-transformers` library because:

- ✅ Free — no API key needed
- ✅ Local — runs on your machine, no internet required
- ✅ Fast — 384 dimensions means faster search
- ✅ Good enough — excellent for learning and most use cases
- ✅ Widely used — most tutorials use this model

---

## Measuring Similarity

Once you have vectors, you need a way to measure how "close" they are. There are three main methods:

### 1. Cosine Similarity (Most Common for RAG)

Measures the **angle** between two vectors, ignoring  magnitude.

```
                    B
                   /
                  / θ ← angle
                 /
                /
    ───────────A───────────▶

    Cosine Similarity = cos(θ)
    Range: -1 (opposite) to 1 (identical)
    Typical threshold: > 0.7 is "similar"
```

**Formula**: `cos(A, B) = (A · B) / (|A| × |B|)`

**Why it's preferred**: It doesn't care about vector length, only direction. A long document chunk and a short question can still be "close" if they're about the same topic.

```python
# Python example
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

# Similar texts → high similarity
sim = cosine_similarity(embed("How do LLMs work?"),
                        embed("Explain language model mechanics"))
# sim ≈ 0.85

# Different texts → low similarity
sim = cosine_similarity(embed("How do LLMs work?"),
                        embed("Best pizza in New York"))
# sim ≈ 0.12
```

## Cosine Similarity

**Cosine similarity** measures how similar two vectors are by calculating the cosine of the angle between them. It focuses purely on  **direction** , not magnitude.

### The Intuition

Imagine two arrows starting from the same point:

* If they point in the **same direction** → cosine similarity = **1** (identical meaning)
* If they're **perpendicular** (90°) → cosine similarity = **0** (unrelated)
* If they point in **opposite directions** → cosine similarity = **-1** (opposite meaning)

### The Math (Simplified)

<pre><div node="[object Object]" class="relative whitespace-pre-wrap word-break-all my-2 rounded-lg bg-list-hover-subtle border border-gray-500/20"><div class="min-h-7 relative box-border flex flex-row items-center justify-between rounded-t border-b border-gray-500/20 px-2 py-0.5"><div class="font-sans text-sm text-ide-text-color opacity-60"></div><div class="flex flex-row gap-2 justify-end"></div></div><div class="p-3"><div class="w-full h-full text-xs cursor-text"><div class="code-block"><div class="code-line" data-line-number="1" data-line-start="1" data-line-end="1"><div class="line-content"><span class="mtk1">cos(A, B) = (A · B) / (|A| × |B|)</span></div></div></div></div></div></div></pre>

* **A · B** = dot product (multiply corresponding elements, sum them up)
* **|A|** and **|B|** = magnitudes (lengths) of the vectors

**Example** with 3D vectors:

<pre><div node="[object Object]" class="relative whitespace-pre-wrap word-break-all my-2 rounded-lg bg-list-hover-subtle border border-gray-500/20"><div class="min-h-7 relative box-border flex flex-row items-center justify-between rounded-t border-b border-gray-500/20 px-2 py-0.5"><div class="font-sans text-sm text-ide-text-color opacity-60"></div><div class="flex flex-row gap-2 justify-end"></div></div><div class="p-3"><div class="w-full h-full text-xs cursor-text"><div class="code-block"><div class="code-line" data-line-number="1" data-line-start="1" data-line-end="1"><div class="line-content"><span class="mtk1">A = [1, 2, 3]</span></div></div><div class="code-line" data-line-number="2" data-line-start="2" data-line-end="2"><div class="line-content"><span class="mtk1">B = [2, 4, 6]    ← same direction, just scaled</span></div></div><div class="code-line" data-line-number="3" data-line-start="3" data-line-end="3"><div class="line-content"><span class="mtk1"></span></div></div><div class="code-line" data-line-number="4" data-line-start="4" data-line-end="4"><div class="line-content"><span class="mtk1">A · B = (1×2) + (2×4) + (3×6) = 2 + 8 + 18 = 28</span></div></div><div class="code-line" data-line-number="5" data-line-start="5" data-line-end="5"><div class="line-content"><span class="mtk1">|A| = √(1² + 2² + 3²) = √14</span></div></div><div class="code-line" data-line-number="6" data-line-start="6" data-line-end="6"><div class="line-content"><span class="mtk1">|B| = √(2² + 4² + 6²) = √56</span></div></div><div class="code-line" data-line-number="7" data-line-start="7" data-line-end="7"><div class="line-content"><span class="mtk1"></span></div></div><div class="code-line" data-line-number="8" data-line-start="8" data-line-end="8"><div class="line-content"><span class="mtk1">cos(A, B) = 28 / (√14 × √56) = 28 / 28 = 1.0  ← perfectly similar!</span></div></div></div></div></div></div></pre>

---

### Why It's the Go-To for RAG

1. **Length-agnostic** : A 500-word document chunk and a 10-word query can still score high similarity if they're about the same topic. Embeddings of different-length texts have different magnitudes — cosine similarity **ignores this** and only cares about semantic direction.
2. **Semantic matching** : In embedding space, vectors that point in similar directions encode similar  *meaning* . Cosine similarity directly captures this.
3. **Efficient retrieval** : Vector databases (Pinecone, Chroma, FAISS) are optimized around cosine similarity for fast approximate nearest-neighbor (ANN) search across millions of embeddings.
4. **Normalized scores** : The fixed `[-1, 1]` range makes it easy to set thresholds (e.g., "only retrieve chunks with similarity > 0.7").


### 2. Euclidean Distance (L2)

Measures the **straight-line distance** between two points in vector space.

```
    B •
    |  \
    |   \ distance = √((x₁-x₂)² + (y₁-y₂)²)
    |    \
    A •───

    Range: 0 (identical) to ∞
    Lower = more similar
```

**When to use**: When the magnitude of vectors is meaningful (less common in text).

### 3. Dot Product

Simply multiplies corresponding elements and sums them.

```
A · B = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ
```

**When to use**: When vectors are already normalized (in which case it equals cosine similarity).

### Comparison

| Metric      | Best for                            | ChromaDB default? |
| ----------- | ----------------------------------- | ----------------- |
| Cosine      | Text similarity (direction matters) | ✅ Yes            |
| Euclidean   | Spatial distance                    | No                |
| Dot Product | Normalized vectors                  | No                |



### Why Not the Alternatives?

| Method                       | What it measures                      | RAG drawback                                                                                      |
| ---------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Euclidean distance** | Straight-line distance between points | Penalizes magnitude — a long doc and short query look "far apart" even if semantically identical |
| **Dot product**        | Unnormalized similarity               | Biased toward longer vectors; scores aren't bounded                                               |
| **Cosine similarity**  | Angular similarity                    | ✅ Direction-only, bounded, works great for comparing texts of varying lengths                    |

---

## The Embedding Process in Our Pipeline

Here's exactly how embeddings fit into our RAG system:

### During Ingestion (Offline)

```python
# 1. Load document
document = "LLMs are trained to predict the next word..."

# 2. Chunk it (covered in 03-chunking-strategies.md)
chunks = ["LLMs are trained to predict...",
          "The training uses massive amounts...",
          "Fine-tuning helps align the model..."]

# 3. Embed each chunk
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vectors = embedding_model.encode(chunks)
# vectors[0] = [0.12, -0.34, 0.56, ...] (384 dimensions)
# vectors[1] = [0.23, -0.45, 0.67, ...]
# vectors[2] = [0.34, -0.56, 0.78, ...]

# 4. Store in ChromaDB (vector + original text + metadata)
collection.add(
    embeddings=vectors,
    documents=chunks,
    metadatas=[{"source": "llm_overview.md", "page": 1}, ...]
)
```

### During Query (Online)

```python
# 1. User asks a question
question = "How are LLMs trained?"

# 2. Embed the question with the SAME model
question_vector = embedding_model.encode(question)
# [0.15, -0.38, 0.59, ...]  ← similar direction to chunk about training!

# 3. Search ChromaDB for nearest vectors
results = collection.query(
    query_embeddings=[question_vector],
    n_results=3  # Top-K = 3
)
# Returns the 3 chunks most similar to the question
```

> **Critical**: The question and documents MUST use the same embedding model. If you re-embed your documents with a different model, you must re-index everything.

---

## Why "Similarity ≠ Relevance"

This is an important concept that PageIndex (which we'll explore later) challenges directly.

Vector similarity captures **semantic closeness**, but relevance often requires **reasoning**:

### Where Vector Search Works Well

```
Question: "What is pre-training?"
Chunk:    "Pre-training is the first phase of LLM development where..."
→ High similarity ✓ AND highly relevant ✓
```

### Where Vector Search Struggles

```
Question: "What happens AFTER pre-training?"
Chunk 1:  "Pre-training uses massive datasets..." (high similarity, low relevance)
Chunk 2:  "Instruction fine-tuning comes next..." (lower similarity, HIGH relevance!)
```

The word "pre-training" appears in both the question and Chunk 1, making them semantically similar. But the *answer* is actually in Chunk 2, which talks about fine-tuning. Understanding that "after pre-training" means "fine-tuning" requires **reasoning**, not just similarity matching.

This limitation is exactly what:

- **Re-ranking** helps with (see `04-retrieval-mechanisms.md`)
- **PageIndex** solves fundamentally (see `07-vectorless-rag-pageindex.md`)

---

## Practical Considerations

### Embedding Speed and Batch Processing

```python
# BAD: Embed one at a time
for chunk in chunks:
    vector = model.encode(chunk)  # Slow!

# GOOD: Embed in batches
vectors = model.encode(chunks, batch_size=32)  # Much faster!
```

### Maximum Input Length

Every embedding model has a max token limit:

- `all-MiniLM-L6-v2`: 256 tokens (word-pieces)
- `all-mpnet-base-v2`: 384 tokens
- OpenAI ada-002: 8,191 tokens

**If your chunk is longer than the model's limit, it gets truncated.** This is why chunk size matters (next doc).

### Dimensionality Trade-offs

- **More dimensions** (768, 1536) = more nuance, better quality, more storage, slower search
- **Fewer dimensions** (384) = faster search, less storage, slightly less nuance
- For learning and most applications, 384 dimensions is excellent

### Storage Requirements

```
1 vector × 384 dimensions × 4 bytes (float32) = 1,536 bytes ≈ 1.5 KB
1,000 chunks = ~1.5 MB
1,000,000 chunks = ~1.5 GB
```

---

## Key Takeaways

1. **Embeddings convert text into vectors** of fixed-size numbers that capture semantic meaning
2. **Similar meanings → similar vectors** — this enables semantic search
3. **Cosine similarity** is the standard metric for comparing text embeddings
4. **Same model for questions and documents** — always use the same embedding model for both
5. **Similarity ≠ Relevance** — vector search finds similar text, but "relevant" sometimes requires reasoning
6. **We use `all-MiniLM-L6-v2`** — free, local, fast, 384 dimensions, excellent quality

---

## What's Next?

You know how text becomes searchable vectors. But how big should each chunk be? How do you split a document?

→ **[03 — Chunking Strategies](./03-chunking-strategies.md)**
