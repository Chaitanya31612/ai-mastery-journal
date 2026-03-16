# 08 — Comparison: Vector RAG vs Vectorless RAG (PageIndex)

> A side-by-side comparison of the two RAG approaches we've studied and built.

---

## Architecture Comparison

```
  VECTOR RAG (Our Pipeline)              PAGEINDEX (Vectorless RAG)
  ──────────────────────────             ──────────────────────────

  INGESTION:                             INGESTION:
  Document                               Document
     │                                      │
     ▼                                      ▼
  Split into chunks (500 chars)          LLM reads entire document
     │                                      │
     ▼                                      ▼
  Embed each chunk → vectors             Build JSON tree index
     │                                   (hierarchical ToC)
     ▼                                      │
  Store in ChromaDB (vectors)            Save tree_index.json
                                         (no vectors, no DB)

  RETRIEVAL:                             RETRIEVAL:
  Question                               Question
     │                                      │
     ▼                                      ▼
  Embed question → vector                Load tree index into
     │                                   LLM context window
     ▼                                      │
  Cosine similarity search               LLM reasons: "Which section
     │                                   likely has the answer?"
     ▼                                      │
  Return top-K chunks                    LLM navigates tree →
     │                                   retrieves full sections
     ▼                                      │
  (Optional) Re-rank with                Loop until enough info
  cross-encoder                             │
     │                                      ▼
     ▼                                   Generate answer
  Generate answer
```

---

## Feature-by-Feature Comparison

| Feature | Vector RAG (Our Pipeline) | PageIndex (Vectorless) |
|---------|--------------------------|----------------------|
| **Retrieval method** | Cosine similarity on embeddings | LLM reasoning over tree structure |
| **Index type** | Vector embeddings in ChromaDB | Hierarchical JSON tree (ToC) |
| **Chunking** | Required (fixed / recursive / semantic) | None — uses natural sections |
| **Embedding model** | `all-MiniLM-L6-v2` (local, free) | Not needed |
| **Storage** | Vector DB (ChromaDB, ~MB per 1K docs) | JSON file (KB) |
| **Search speed** | ~5ms (vector kNN) | ~2–5s (LLM API call per step) |
| **Accuracy** | Good for general Q&A (~75-85%) | Excellent for structured docs (~98%) |
| **Cost per query** | Free (local embeddings) | $$ (LLM API calls for retrieval) |
| **Explainability** | Low (opaque vectors) | High (reasoning trace visible) |
| **Cross-references** | ❌ Can't follow "see Appendix G" | ✅ Navigates tree to references |
| **Chat history** | ❌ Each query independent | ✅ Context-aware multi-turn |
| **Offline capable** | ✅ Everything runs locally | ❌ Requires OpenAI API |
| **Setup complexity** | Medium (install models, ChromaDB) | Simple (just needs OpenAI key) |
| **Best document types** | Short docs, blogs, notes | Long, structured professional docs |

---

## The Fundamental Difference

```
Vector RAG asks:  "What text LOOKS LIKE this question?"
                   → Statistical similarity

PageIndex asks:   "WHERE in this document would the answer BE?"
                   → Logical reasoning
```

### Example: "What was the company's net income in Q3 2023?"

**Vector RAG approach:**
```
1. Embed the question → [0.12, -0.34, 0.56, ...]
2. Search ChromaDB for similar vectors
3. Find chunks containing "net income", "Q3", "2023"
4. Problem: May return chunks from Q1 or Q2 (similar words!)
5. Re-ranker might fix it, might not
```

**PageIndex approach:**
```
1. LLM reads the tree index:
   - "Financial Overview" (pages 1-50)
     - "Quarterly Results" (pages 15-40)
       - "Q1 Results" (pages 15-22)
       - "Q2 Results" (pages 23-30)
       - "Q3 Results" (pages 31-38)  ← This one!
       - "Q4 Results" (pages 39-45)
     - "Annual Summary" (pages 41-50)

2. LLM reasons: "Q3 Results on pages 31-38 is exactly what I need"
3. Retrieves pages 31-38 in full
4. Finds net income figure with full context
```

---

## Cost Analysis

### Vector RAG (Our Pipeline)

| Component | Cost |
|-----------|------|
| Embedding model | Free (local sentence-transformers) |
| ChromaDB | Free (local, embedded) |
| Cross-encoder re-ranking | Free (local model) |
| LLM generation only | ~$0.001/query (Groq/Gemini) |
| **Total per query** | **~$0.001** |

### PageIndex

| Component | Cost |
|-----------|------|
| Tree building (one-time) | ~$0.10–$0.50/doc (GPT-4o) |
| Tree search per query | ~$0.01–$0.05/query (multiple LLM calls) |
| Answer generation | ~$0.01/query |
| **Total per query** | **~$0.02–$0.06** |

PageIndex costs **20-60x more per query**, but delivers significantly higher accuracy on complex documents.

---

## Strengths & Weaknesses Summary

### Vector RAG Wins When:
- ✅ You need **fast** responses (~5ms retrieval)
- ✅ You're on a **budget** (free local embeddings)
- ✅ Documents are **short and simple** (blog posts, notes, FAQs)
- ✅ You need to work **offline** (no internet required)
- ✅ You have **many small documents** (hundreds of files)

### PageIndex Wins When:
- ✅ Documents are **long and structured** (100+ pages)
- ✅ **Accuracy is critical** (financial, legal, medical)
- ✅ Documents have **cross-references** ("see Section 4.2")
- ✅ Users ask **follow-up questions** (multi-turn conversations)
- ✅ You need **explainable retrieval** (audit trails)

---

## The Future: Hybrid Approaches

The ideal system might combine both:

```
Question
   │
   ▼
┌────────────────┐
│ Query Classifier│  "Is this a simple factual question,
│                │   or a complex multi-step question?"
└───────┬────────┘
        │
   Simple ──────────▶ Vector RAG (fast, cheap)
        │
   Complex ─────────▶ PageIndex-style reasoning (accurate, slower)
```

This is where the field is heading — adaptive retrieval that picks the right strategy based on query complexity. Our vector RAG pipeline provides the foundation for the fast path, while understanding PageIndex prepares you for the reasoning-based path.

---

## Key Takeaways

1. **Vector RAG** is a great general-purpose solution — fast, free, works offline
2. **PageIndex** excels at complex, structured documents where similarity ≠ relevance
3. Neither is universally "better" — they solve different problems
4. The skill is knowing **when to use which** approach
5. Understanding both gives you the vocabulary for the next wave of RAG innovations
