# Chunking Strategies

> **Prerequisites**: [01 — What is RAG?](./01-what-is-rag.md), [02 — Vector Embeddings](./02-vector-embeddings-explained.md)
> **Reading time**: ~12 minutes

---

## Why Chunking Matters

You can't feed an entire 100-page PDF into an embedding model — it has a token limit (our `all-MiniLM-L6-v2` accepts max 256 tokens). And even if you could, one massive vector for a whole document would be too broad to match specific questions accurately.

**Chunking** = breaking documents into smaller, meaningful pieces that can be individually embedded and retrieved.

This is arguably the **most impactful** design decision in a RAG pipeline. Too large and your chunks are unfocused; too small and you lose context.

---

## The Chunking Trade-off

```
     ◀────── Chunk Size Spectrum ──────▶

  Small chunks              Large chunks
  (100-200 tokens)          (1000-2000 tokens)

  ✅ Precise matching       ✅ More context preserved
  ✅ Less noise             ✅ Better for complex topics
  ❌ Loses context          ❌ Mixes multiple topics
  ❌ May need many chunks   ❌ Less precise matching
```

**Sweet spot for most use cases**: 200-500 tokens (roughly 800-2000 characters)

---

## Strategy 1: Fixed-Size Chunking

The simplest approach — split text every N characters.

### How It Works
```
Document: "The quick brown fox jumps over the lazy dog. Machine learning
           is a subset of AI. Neural networks have layers."

Chunk size: 50 chars, Overlap: 10 chars

Chunk 1: "The quick brown fox jumps over the lazy dog. Mac"
Chunk 2: "lazy dog. Machine learning is a subset of AI. Ne"
Chunk 3: "of AI. Neural networks have layers."
```

### The Overlap Concept
Without overlap, a sentence split across two chunks is fragmented in both — neither chunk has the full meaning. **Overlap** duplicates some text at chunk boundaries so sentences aren't cut in half.

```
Without overlap:
  Chunk 1: "...model learns to predict the"    ← incomplete thought
  Chunk 2: "next word in a sequence..."        ← missing context

With overlap (50 chars):
  Chunk 1: "...model learns to predict the next word in a"  ← complete!
  Chunk 2: "predict the next word in a sequence of..."      ← also has it!
```

**Typical overlap**: 10-20% of chunk size

### Pros & Cons
| Pros | Cons |
|------|------|
| Dead simple to implement | May split mid-sentence or mid-word |
| Predictable chunk sizes | No awareness of document structure |
| Fast | Chunks may lack semantic coherence |

### When to Use
- Quick prototypes
- Uniform documents without clear structure (plain text logs, etc.)

---

## Strategy 2: Recursive Character Splitting

The **most commonly used** strategy (LangChain's default). It tries to split on natural boundaries.

### How It Works
Uses a hierarchy of separators, trying the largest one first:
1. Split by `\n\n` (paragraphs)
2. If chunks are still too big, split by `\n` (lines)
3. If still too big, split by `. ` (sentences)
4. If still too big, split by ` ` (words)
5. Last resort: split by character

```python
separators = ["\n\n", "\n", ". ", " ", ""]

# Document with paragraphs:
"""
Paragraph 1 about topic A that is long enough to need splitting.

Paragraph 2 about topic B with important details.

Paragraph 3 about topic C with more information.
"""

# Step 1: Split on "\n\n" → 3 chunks (one per paragraph)
# If any chunk > max_size, recursively split with next separator
```

### Example
```
Input document (600 chars, max_chunk_size=200):

"Machine learning is a subset of AI focused on pattern recognition.
It has various applications in modern technology.

Deep learning uses neural networks with many layers. These networks
can model complex relationships. The transformer architecture
revolutionized natural language processing.

Large Language Models are trained on massive text datasets. They
predict the next token in a sequence."

Split on "\n\n" → 3 paragraphs

Paragraph 1 (134 chars): ✅ Under 200, keep as one chunk
Paragraph 2 (193 chars): ✅ Under 200, keep as one chunk
Paragraph 3 (113 chars): ✅ Under 200, keep as one chunk

Result: 3 clean, paragraph-aligned chunks!
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Respects natural text boundaries | Chunk sizes can vary a lot |
| Works well for most document types | Still doesn't understand semantics |
| Configurable separator hierarchy | May split related paragraphs apart |

### When to Use
- **Default choice** for most RAG systems
- Works well for articles, documentation, books
- Our pipeline implements this as the primary strategy

---

## Strategy 3: Markdown-Aware Chunking

Splits specifically based on **Markdown headers** (`#`, `##`, `###`), preserving document structure.

### How It Works
```markdown
# Introduction                    ──▶ Chunk 1: "Introduction" section
This is the introduction...

## Background                     ──▶ Chunk 2: "Background" section
Some background info...

### Technical Details              ──▶ Chunk 3: "Technical Details" section
The details of the approach...

## Results                        ──▶ Chunk 4: "Results" section
Our results show...
```

Each section becomes a chunk, and the header hierarchy provides natural boundaries. If a section is too long, it splits on sub-headers or falls back to recursive splitting within that section.

### Metadata Bonus
You get rich metadata for free:
```python
Chunk(
    content="The details of the approach...",
    metadata={
        "headers": ["Introduction", "Background", "Technical Details"],
        "header_level": 3,
        "source": "document.md"
    }
)
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Preserves document hierarchy | Only works with Markdown |
| Natural section boundaries | Sections may be very unequal in size |
| Rich metadata from headers | Requires well-structured documents |

### When to Use
- Markdown documentation (READMEs, wikis, your learning docs!)
- Technical documentation
- Any structured document with clear headers

---

## Strategy 4: Semantic Chunking

The most sophisticated approach — uses **embedding similarity** to find natural topic boundaries.

### How It Works
1. Split text into sentences
2. Embed each sentence
3. Compare each sentence's embedding with its neighbors
4. Where similarity drops significantly → that's a topic boundary → split there

```
Sentence 1: "LLMs predict the next word."           ─┐
Sentence 2: "Training uses massive datasets."        ─┤ Similar → Same chunk
Sentence 3: "The data comes from the internet."      ─┘
                                                      ← Big similarity drop!
Sentence 4: "Transformers use attention mechanisms."  ─┐
Sentence 5: "The attention weights show relevance."   ─┘ Similar → Same chunk
```

```
Similarity between consecutive sentences:
  S1-S2: 0.82  ──────── high (same topic)
  S2-S3: 0.78  ──────── high (same topic)
  S3-S4: 0.31  ──── ⚡ LOW (topic change!) → Split here!
  S4-S5: 0.85  ──────── high (same topic)
```

### Pros & Cons
| Pros | Cons |
|------|------|
| Most semantically coherent chunks | Slow (needs embedding each sentence) |
| Adapts to content structure | Chunk sizes are unpredictable |
| Best theoretical retrieval quality | More complex to implement |

### When to Use
- When retrieval quality is critical
- Documents with flowing text and no clear structure
- When you can afford the extra computation during ingestion

---

## Choosing a Strategy

Here's a decision flowchart:

```
                    Is your document Markdown?
                    /                         \
                  Yes                          No
                  /                              \
    Use Markdown-Aware                  Is document structure clear?
    Chunking                            (paragraphs, sections)
                                        /                     \
                                      Yes                      No
                                      /                          \
                            Use Recursive                Is retrieval quality
                            Character Splitting          more important than speed?
                                                        /                      \
                                                      Yes                       No
                                                      /                          \
                                            Use Semantic                Use Fixed-Size
                                            Chunking                   Chunking
```

### Our Pipeline's Approach

We implement **all four strategies** so you can experiment:

```python
from chunking import (
    FixedSizeChunker,       # Simple, fast
    RecursiveChunker,       # Default, balanced
    MarkdownChunker,        # For .md files
    SemanticChunker,        # Best quality, slowest
)

# Default: RecursiveChunker with 500 char chunks, 50 char overlap
chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
chunks = chunker.chunk(document)
```

---

## Practical Tips

### 1. Always Include Metadata
```python
# Don't just store the text
chunk = Chunk(
    content="The training process involves...",
    metadata={
        "source": "llm_overview.md",
        "chunk_index": 3,
        "total_chunks": 12,
        "created_at": "2025-03-06"
    }
)
```
Metadata helps you trace answers back to source documents and filter during retrieval.

### 2. Test With Your Actual Data
There's no universally "best" chunk size. It depends on:
- Your documents (short articles vs. long manuals)
- Your questions (specific facts vs. broad understanding)
- Your embedding model's max token limit

### 3. Common Chunk Size Guidelines

| Document Type | Recommended Size | Overlap |
|---------------|-----------------|---------|
| Technical docs | 500-1000 chars | 100 chars |
| Articles/blogs | 300-500 chars | 50 chars |
| Code files | By function/class | None |
| Legal/financial | 1000-2000 chars | 200 chars |
| Chat logs | By conversation turn | None |

### 4. Watch for the "Lost in the Middle" Problem
Research shows LLMs pay less attention to middle chunks in the context. If you retrieve 5 chunks, chunks 1 and 5 get more attention than chunks 2-4. Solutions:
- Retrieve fewer, higher-quality chunks
- Re-rank so the best chunk is first
- Use smaller chunks for more precise matches

---

## Key Takeaways

1. **Chunking is the most impactful RAG decision** — it directly affects retrieval quality
2. **Recursive splitting is the default** — use it unless you have a good reason not to
3. **Markdown-aware chunking for .md files** — preserves structure and gives you free metadata
4. **Semantic chunking is best but slowest** — use when quality matters more than speed
5. **Overlap prevents losing context** at chunk boundaries (10-20% of chunk size)
6. **There's no perfect chunk size** — test with your actual data and questions

---

## What's Next?

You know how to create chunks. Now let's learn how to *find* the right ones when a user asks a question.

→ **[04 — Retrieval Mechanisms](./04-retrieval-mechanisms.md)**
