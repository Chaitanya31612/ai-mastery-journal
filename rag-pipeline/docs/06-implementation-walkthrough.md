# 06 — Implementation Walkthrough

> A complete guide to how the RAG pipeline code works, module by module.
> Read this alongside the code — every section maps to a folder in `code/`.

---

## Project Structure

```
rag-pipeline/
├── docs/                          # You're reading these
│   ├── 01-what-is-rag.md
│   ├── 02-vector-embeddings-explained.md
│   ├── 03-chunking-strategies.md
│   ├── 04-retrieval-mechanisms.md
│   ├── 05-rag-pipeline-architecture.md
│   └── 06-implementation-walkthrough.md  ← this file
│
└── code/                          # The actual pipeline
    ├── requirements.txt           # Dependencies (CPU-only torch)
    ├── .env.example               # API key template
    ├── config.py                  # Central configuration
    ├── pipeline.py                # Main orchestrator (Facade Pattern)
    ├── main.py                    # CLI entry point
    │
    ├── loaders/                   # Step 1: Document ingestion
    │   ├── base.py                # Document dataclass + LoaderABC
    │   ├── text_loader.py         # .txt files
    │   ├── pdf_loader.py          # .pdf files (PyMuPDF)
    │   ├── markdown_loader.py     # .md files (header-aware)
    │   ├── csv_loader.py          # .csv files (row-to-text)
    │   ├── json_loader.py         # .json files (field extraction)
    │   └── directory_loader.py    # Auto-detect + recursive scan
    │
    ├── chunking/                  # Step 2: Text splitting
    │   ├── base.py                # Chunk dataclass + StrategyABC
    │   ├── fixed_size.py          # Sliding window
    │   ├── recursive.py           # Natural boundaries (default)
    │   ├── markdown_chunker.py    # Header-aware splitting
    │   └── semantic.py            # Embedding-based topic detection
    │
    ├── embeddings/                # Step 3: Vectorization
    │   ├── base.py                # EmbeddingProvider ABC
    │   └── sentence_transformer.py # Local model (all-MiniLM-L6-v2)
    │
    ├── vectorstore/               # Step 4: Storage
    │   ├── base.py                # VectorStore ABC + SearchResult
    │   └── chroma_store.py        # ChromaDB (persistent, embedded)
    │
    ├── retrieval/                  # Step 5: Search
    │   ├── base.py                # Retriever ABC
    │   ├── vector_retriever.py    # Embed query → similarity search
    │   └── reranker.py            # Cross-encoder re-ranking
    │
    └── ai/                        # Step 6: Generation (pre-existing)
        ├── base.py                # AIProvider enum + AIResponse
        ├── factory.py             # AIAnalyzerFactory
        ├── groq_analyzer.py       # Groq integration
        ├── gemini_analyzer.py     # Gemini integration
        └── ollama_analyzer.py     # Ollama integration
```

---

## Data Flow — How It All Connects

```
                         INGESTION (pipeline.ingest)
                         ─────────────────────────
  Source File/Dir
       │
       ▼
  ┌─────────────────┐     Document = {content: str, metadata: dict}
  │  DirectoryLoader │──▶  One Document per file (or per page for PDFs)
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐     Chunk = {content: str, metadata: dict, chunk_index: int}
  │ ChunkingStrategy │──▶  Multiple Chunks per Document
  └─────────────────┘     (default: RecursiveChunker at 500 chars)
       │
       ▼
  ┌─────────────────┐     numpy array of shape (num_chunks, 384)
  │ EmbeddingProvider│──▶  One 384-dim vector per Chunk
  └─────────────────┘     (model: all-MiniLM-L6-v2)
       │
       ▼
  ┌─────────────────┐     Chunks + Vectors saved to disk
  │   ChromaStore    │──▶  Persistent via ChromaDB (HNSW index)
  └─────────────────┘


                         QUERY (pipeline.query)
                         ─────────────────────
  "What is RAG?"
       │
       ▼
  ┌─────────────────┐     Query → 384-dim vector → cosine search
  │ VectorRetriever  │──▶  Top-20 candidate chunks
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐     Cross-encoder re-scores each (query, chunk) pair
  │    ReRanker      │──▶  Top-5 most RELEVANT chunks
  └─────────────────┘
       │
       ▼
  ┌─────────────────┐     Question + Context → Augmented Prompt
  │  AI Analyzer     │──▶  LLM generates grounded answer
  └─────────────────┘
       │
       ▼
  QueryResult = {answer: str, sources: [Chunk], metadata: dict}
```

---

## Module-by-Module Walkthrough

### 1. Configuration (`config.py`)

The `RAGConfig` dataclass centralizes every tunable parameter:

```python
@dataclass
class RAGConfig:
    # Chunking
    chunk_size: int = 500           # Max chars per chunk
    chunk_overlap: int = 50         # Overlap between chunks
    chunking_strategy: str = "recursive"  # fixed | recursive | markdown | semantic

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"   # 384 dimensions
    embedding_batch_size: int = 32

    # Vector Store
    vectordb_path: str = "./data/vectordb"
    collection_name: str = "rag_documents"

    # Retrieval
    search_top_k: int = 20         # Candidates for re-ranking
    rerank_top_k: int = 5          # Final results after re-ranking
    use_reranker: bool = True

    # AI Generation
    ai_provider: str = "groq"      # groq | gemini | ollama
```

**Why this matters**: One object controls the entire pipeline. Change `chunk_size` from 500 to 1000 and every component automatically adapts. No hunting through 10 files to update settings.

---

### 2. Document Loaders (`loaders/`)

**Pattern**: Strategy Pattern — all loaders implement `DocumentLoader.load(path) → list[Document]`

The `Document` dataclass is the universal container:
```python
@dataclass
class Document:
    content: str          # The raw text
    metadata: dict        # Source info (file_name, page_number, etc.)
```

| Loader | File Type | Key Feature |
|--------|-----------|-------------|
| `TextLoader` | `.txt` | Simplest — reference implementation |
| `PDFLoader` | `.pdf` | Uses PyMuPDF, one Document per page |
| `MarkdownLoader` | `.md` | Optionally splits by headers, preserves hierarchy |
| `CSVLoader` | `.csv` | Converts each row to natural language text |
| `JSONLoader` | `.json` | Configurable field extraction, handles nested paths |
| `DirectoryLoader` | directories | Auto-detects file types, recursive scanning, registry pattern |

**`DirectoryLoader`** is the main entry point. It maintains a registry that maps file extensions to loader instances:

```python
self._registry = {
    ".txt": TextLoader(),
    ".pdf": PDFLoader(),
    ".md":  MarkdownLoader(),
    ".csv": CSVLoader(),
    ".json": JSONLoader(),
}
```

When you call `loader.load("docs/")`, it walks the directory, matches each file to the right loader, and returns all Documents.

---

### 3. Chunking Engine (`chunking/`)

**Pattern**: Strategy Pattern — all chunkers implement `ChunkingStrategy.chunk(document) → list[Chunk]`

The `Chunk` dataclass extends Document with chunking metadata:
```python
@dataclass
class Chunk:
    content: str         # The chunk text
    metadata: dict       # Inherited from Document + chunk_index, total_chunks, etc.
    chunk_index: int     # Position within the parent document
    document_id: str     # Unique ID for vector store
```

| Strategy | Algorithm | Best For |
|----------|-----------|----------|
| `FixedSizeChunker` | Sliding window, every N characters | Speed, predictability |
| `RecursiveChunker` | Split on paragraphs → sentences → words | **Default** — best balance |
| `MarkdownChunker` | Split at headers, preserve hierarchy | Structured .md files |
| `SemanticChunker` | Embedding similarity to detect topic shifts | Highest quality, slowest |

**`RecursiveChunker`** (the default) tries separators in order of "naturalness":
1. `"\n\n"` (paragraph breaks) — best boundaries
2. `"\n"` (line breaks)
3. `". "` (sentences)
4. `" "` (words) — last resort

The factory function `create_chunker("recursive", chunk_size=500)` creates the right chunker by name.

---

### 4. Embeddings (`embeddings/`)

**Key concepts**:
- An embedding converts text → a fixed-size numeric vector (384 numbers)
- Semantically similar texts produce similar vectors
- Same model MUST be used for both ingestion and querying

**`SentenceTransformerEmbedder`** wraps the `sentence-transformers` library:

```python
embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

# Batch embed documents
vectors = embedder.embed(["chunk 1 text", "chunk 2 text"])  # shape: (2, 384)

# Embed a query
query_vec = embedder.embed_query("What is RAG?")  # shape: (384,)
```

**Lazy loading**: The ~80MB model downloads on first use, then caches locally. The `__init__` is instant — the heavy lifting only happens when you actually call `embed()`.

**Normalization**: Vectors are L2-normalized, meaning cosine similarity = dot product. This gives a slight speedup during search.

---

### 5. Vector Store (`vectorstore/`)

**ChromaDB** acts as an embedded database — think "SQLite for vectors":

```python
store = ChromaStore(path="./data/vectordb", collection_name="my_docs")

# Add chunks (persisted to disk)
store.add(chunks, embeddings)

# Search (returns SearchResult objects)
results = store.search(query_vector, top_k=5)
# results[0].score = 0.87 (cosine similarity)
# results[0].chunk.content = "The relevant text..."
```

**Key implementation details**:

1. **Metadata sanitization**: ChromaDB only accepts `str|int|float|bool` as metadata values. Lists and dicts are JSON-serialized on write and deserialized on read.

2. **Distance → Score conversion**: ChromaDB returns distances (lower = better). We convert with `score = 1 - distance` so higher = better.

3. **HNSW index**: ChromaDB uses Hierarchical Navigable Small World graphs internally — O(log n) search instead of O(n) brute-force.

---

### 6. Retrieval (`retrieval/`)

**Two-stage retrieval** for the best results:

**Stage 1 — `VectorRetriever`** (fast, approximate):
```python
retriever = VectorRetriever(vector_store=store, embedding_provider=embedder)
candidates = retriever.retrieve("What is pre-training?", top_k=20)
```
Embeds the query → searches vector store → returns top-20 candidates by cosine similarity.

**Stage 2 — `ReRanker`** (slow, accurate):
```python
reranker = ReRanker()
best = reranker.rerank("What is pre-training?", candidates, top_k=5)
```

The key difference:
- **Bi-encoder** (vector search): Encodes query and chunks separately, compares vectors
- **Cross-encoder** (re-ranker): Encodes query + chunk together, judges actual relevance

The cross-encoder "sees" both texts at once, so it catches nuances like "What happens AFTER pre-training?" (needs info about fine-tuning, not pre-training itself).

---

### 7. Pipeline Orchestrator (`pipeline.py`)

**Pattern**: Facade Pattern — simple interface to a complex system.

```python
pipeline = RAGPipeline(config)
pipeline.ingest("docs/")                     # Run full ingestion
result = pipeline.query("What is RAG?")      # Run full query
print(result.answer)                         # Grounded answer
print(result.sources)                        # Source chunks used
```

The `__init__` wires everything together:
```
RAGPipeline
  ├── SentenceTransformerEmbedder  (lazy model loading)
  ├── ChromaStore                  (persistent on disk)
  ├── VectorRetriever              (embedder + store)
  ├── ReRanker                     (lazy model loading)
  └── AIAnalyzer                   (lazy, via AIAnalyzerFactory)
```

**AI integration** uses the pre-existing `ai/` module (Groq, Gemini, Ollama). The prompt template instructs the LLM to answer ONLY from the retrieved context and cite sources.

---

### 8. CLI (`main.py`)

Four subcommands, each mapping to a pipeline method:

```bash
# Index documents
python -m code.main ingest path/to/docs/ --chunk-strategy recursive --chunk-size 500

# Ask questions
python -m code.main query "What is RAG?" --top-k 5 --provider groq

# Check status
python -m code.main status

# Clear all data
python -m code.main reset
```

---

## Design Patterns Used

| Pattern | Where | Why |
|---------|-------|-----|
| **Strategy** | Chunkers, Loaders | Swap algorithms without changing pipeline |
| **Factory** | `create_chunker()`, `AIAnalyzerFactory` | Create objects by name/config |
| **Facade** | `RAGPipeline` | Simple interface to complex subsystem |
| **Registry** | `DirectoryLoader._registry` | Map file extensions to loaders |
| **Lazy Loading** | Embedder, ReRanker, AI | Fast imports, load models only when needed |
| **Dataclass** | `Document`, `Chunk`, `SearchResult`, `QueryResult`, `RAGConfig` | Clean data containers with defaults |

---

## Setup & Running

```bash
cd rag-pipeline/code

# Create venv and install
python3 -m venv venv
source venv/bin/activate

# IMPORTANT: Install CPU-only torch first (~200MB vs ~1.5GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install everything else
pip install -r requirements.txt

# Copy and edit .env with your API keys
cp .env.example .env

# Use the pipeline
python -m code.main ingest ../docs/
python -m code.main query "What is RAG?"
python -m code.main status
```

---

## What's Next

- **Phase 3**: Explore PageIndex (vectorless RAG) as an alternative approach
- **Phase 4**: End-to-end testing and quality verification
