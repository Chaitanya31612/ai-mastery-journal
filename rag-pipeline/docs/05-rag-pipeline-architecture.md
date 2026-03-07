# RAG Pipeline Architecture

> **Prerequisites**: All previous docs (01-04)
> **Reading time**: ~10 minutes

---

## The Complete Picture

Now that you understand each building block, let's see how they connect into a complete RAG system. This document maps directly to the code we'll build.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         RAG Pipeline                                 │
│                                                                      │
│  ┌────────────────────── INGESTION ──────────────────────┐           │
│  │                                                        │           │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────┐ │           │
│  │  │ Document │──▶│ Chunking │──▶│Embedding │──▶│Vec │ │           │
│  │  │ Loaders  │   │ Engine   │   │ Engine   │   │ DB │ │           │
│  │  │          │   │          │   │          │   │    │ │           │
│  │  │ PDF      │   │ Fixed    │   │ MiniLM   │   │Chro│ │           │
│  │  │ Markdown │   │ Recursive│   │ L6-v2    │   │maDB│ │           │
│  │  │ Text     │   │ Markdown │   │          │   │    │ │           │
│  │  │ CSV      │   │ Semantic │   │          │   │    │ │           │
│  │  │ JSON     │   │          │   │          │   │    │ │           │
│  │  └──────────┘   └──────────┘   └──────────┘   └────┘ │           │
│  └────────────────────────────────────────────────────────┘           │
│                                                                      │
│  ┌────────────────────── QUERY ─────────────────────────┐           │
│  │                                                        │           │
│  │  Question ──▶ Embed ──▶ Search ──▶ Rerank ──▶ Augment │           │
│  │                                       │         │      │           │
│  │                                       │    ┌────▼────┐ │           │
│  │                                       │    │   LLM   │ │           │
│  │                                       │    │ (Groq/  │ │           │
│  │                                       │    │ Gemini/ │ │           │
│  │                                       │    │ Ollama) │ │           │
│  │                                       │    └────┬────┘ │           │
│  │                                       │         │      │           │
│  │                                  Answer + Sources      │           │
│  └────────────────────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Map (Code ↔ Concept)

| Concept | Code Module | Key Class | Purpose |
|---------|------------|-----------|---------|
| Document Loading | `code/loaders/` | `DocumentLoader` | Parse different file formats into `Document` objects |
| Chunking | `code/chunking/` | `ChunkingStrategy` | Split documents into `Chunk` objects |
| Embedding | `code/embeddings/` | `EmbeddingProvider` | Convert text to vectors |
| Vector Storage | `code/vectorstore/` | `VectorStore` | Store and search vectors (ChromaDB) |
| Retrieval | `code/retrieval/` | `Retriever` | Search + re-rank to find relevant chunks |
| Generation | `code/ai/` | `AIAnalyzerInterface` | Generate answers using LLMs (existing factory) |
| Orchestration | `code/pipeline.py` | `RAGPipeline` | Ties everything together |
| Configuration | `code/config.py` | `RAGConfig` | Central settings |

---

## Data Flow: Ingestion

```python
# 1. User provides a source
pipeline.ingest("path/to/documents/")

# 2. Directory loader auto-detects file types
#    ".md" → MarkdownLoader
#    ".pdf" → PDFLoader
#    ".txt" → TextLoader
#    etc.
documents: list[Document] = loader.load(source)

# 3. Each document is chunked
chunks: list[Chunk] = chunker.chunk(document)
#    Document { content: "...(5000 chars)", source: "doc.md" }
#    → Chunk { content: "...(500 chars)", metadata: {source, index, ...} }
#    → Chunk { content: "...(500 chars)", metadata: {source, index, ...} }
#    → ... (10 chunks)

# 4. Chunks are embedded
vectors: list[list[float]] = embedder.embed([c.content for c in chunks])
#    Each chunk → [0.12, -0.34, 0.56, ...] (384 dims)

# 5. Stored in ChromaDB
vectorstore.add(chunks, vectors)
#    Persisted to disk at ./data/vectordb/
```

## Data Flow: Query

```python
# 1. User asks a question
answer = pipeline.query("What is pre-training?")

# 2. Question is embedded
query_vector = embedder.embed("What is pre-training?")

# 3. Vector search in ChromaDB (top-20)
candidates = vectorstore.search(query_vector, top_k=20)

# 4. Re-rank for relevance (top-5)
relevant_chunks = reranker.rerank(question, candidates, top_k=5)

# 5. Build augmented prompt
prompt = f"""Answer based on this context:
{chr(10).join(c.content for c in relevant_chunks)}

Question: What is pre-training?"""

# 6. Generate answer using AI factory
response = ai_analyzer.analyze(content=prompt, prompt=SYSTEM_PROMPT)

# 7. Return answer + sources
return {
    "answer": response.content,
    "sources": [c.metadata["source"] for c in relevant_chunks]
}
```

---

## Configuration Design

Everything is tunable through `RAGConfig`:

```python
@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 500           # Characters per chunk
    chunk_overlap: int = 50         # Overlap between chunks
    chunking_strategy: str = "recursive"  # fixed, recursive, markdown, semantic

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector Store
    vectordb_path: str = "./data/vectordb"
    collection_name: str = "documents"

    # Retrieval
    search_top_k: int = 20         # Candidates from vector search
    rerank_top_k: int = 5          # Final chunks after re-ranking
    use_reranker: bool = True      # Enable/disable re-ranking

    # Generation
    ai_provider: str = "groq"      # groq, gemini, ollama
    ai_model: str = "smart"        # Provider-specific model name
    max_tokens: int = 4096
    temperature: float = 0.1       # Low for factual answers
```

---

## Interface Design Pattern

Every component follows the same pattern — **abstract base class + concrete implementations**:

```python
# Base interface (what the component CAN do)
class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        pass

# Concrete implementations (HOW it does it)
class RecursiveChunker(ChunkingStrategy):
    def chunk(self, document: Document) -> list[Chunk]:
        # Recursive character splitting logic
        ...

class MarkdownChunker(ChunkingStrategy):
    def chunk(self, document: Document) -> list[Chunk]:
        # Header-aware splitting logic
        ...
```

This means you can **swap any component** without changing the rest of the pipeline:
- Switch from `RecursiveChunker` to `SemanticChunker`
- Switch from `ChromaStore` to a future `FAISSStore`
- Switch from `SentenceTransformerEmbedder` to an `OpenAIEmbedder`

This is the same factory pattern you already use in your `ai/` module!

---

## Error Handling Strategy

```
Pipeline gracefully handles:
├── Missing files          → Skip with warning
├── Unsupported formats    → Skip with warning
├── Empty documents        → Skip silently
├── Embedding failures     → Retry once, then skip chunk
├── Vector store errors    → Raise (critical)
├── AI provider failures   → Return error response (uses AIResponse.error)
└── Rate limits            → Built into GroqAnalyzer._rate_limit()
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sync, not async | `def` not `async def` | Simpler for learning; LLM calls are already the bottleneck |
| Dataclasses, not dicts | `Document`, `Chunk`, `AIResponse` | Type safety, IDE autocomplete, self-documenting |
| Factory pattern | `AIAnalyzerFactory` | Already proven in your codebase, extensible |
| Strategy pattern | Swappable chunkers/retrievers | Easy experimentation without code changes |
| Persistent ChromaDB | Files on disk | Survives restarts, no server needed |

---

## What's Next?

With the architecture understood, the next docs are:
- **[06 — Implementation Walkthrough](./06-implementation-walkthrough.md)** — Detailed code walkthrough (created after implementation)
- **[07 — Vectorless RAG: PageIndex](./07-vectorless-rag-pageindex.md)** — The alternative approach
- **[08 — Comparison: Vector vs Vectorless](./08-comparison-vector-vs-vectorless.md)** — Side-by-side analysis
