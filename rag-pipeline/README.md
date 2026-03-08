# RAG Pipeline

A modular Retrieval-Augmented Generation pipeline built from scratch in Python — ingest documents, chunk and embed them, store in a vector database, and query with LLM-powered answers.

## Folder Structure

```
rag-pipeline/
├── code/                        # Pipeline source code
│   ├── main.py                  # CLI entry point (ingest / query / status / reset)
│   ├── pipeline.py              # Core RAG pipeline orchestration
│   ├── config.py                # Centralised configuration
│   ├── requirements.txt         # Python dependencies
│   ├── .env.example             # Environment variable template
│   ├── loaders/                 # Document loaders
│   │   ├── text_loader.py
│   │   ├── markdown_loader.py
│   │   ├── pdf_loader.py
│   │   ├── csv_loader.py
│   │   ├── json_loader.py
│   │   └── directory_loader.py
│   ├── chunking/                # Chunking strategies
│   │   ├── fixed_size.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   └── markdown_chunker.py
│   ├── embeddings/              # Embedding models
│   │   └── sentence_transformer.py
│   ├── vectorstore/             # Vector database
│   │   └── chroma_store.py
│   ├── retrieval/               # Retrieval & re-ranking
│   │   ├── vector_retriever.py
│   │   └── reranker.py
│   └── ai/                     # LLM providers
│       ├── gemini_analyzer.py
│       ├── groq_analyzer.py
│       └── ollama_analyzer.py
├── docs/                        # Learning notes (theory & walkthroughs)
│   ├── 01-what-is-rag.md
│   ├── 02-vector-embeddings-explained.md
│   ├── 03-chunking-strategies.md
│   ├── 04-retrieval-mechanisms.md
│   ├── 05-rag-pipeline-architecture.md
│   └── 06-implementation-walkthrough.md
└── data/
    └── vectordb/                # Persisted ChromaDB data
```

## Quick Start

```bash
# 1. Install dependencies
cd rag-pipeline/code
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your Groq / Gemini / Ollama keys

# 3. Ingest documents
python -m code.main ingest /path/to/documents

# 4. Query
python -m code.main query "your question here"

# 5. Check pipeline status
python -m code.main status

# 6. Reset indexed data
python -m code.main reset
```

## CLI Commands

| Command  | Description |
|----------|-------------|
| `ingest` | Load and index documents from a given path |
| `query`  | Search indexed documents and generate an LLM answer |
| `status` | Show pipeline state (indexed chunks, active models) |
| `reset`  | Clear all indexed data |

### Key Flags

- `--chunk-strategy` — Override chunking strategy (`fixed`, `recursive`, `semantic`, `markdown`)
- `--provider` — Choose LLM provider (`groq`, `gemini`, `ollama`)
- `--no-rerank` — Skip cross-encoder re-ranking for faster results
- `--collection` — Target a specific ChromaDB collection

## Supported Formats

Text · Markdown · PDF · CSV · JSON — plus a `directory_loader` that auto-detects file types.

## Learning Docs

The [`docs/`](docs/) folder contains concept notes that explain the theory behind each stage:

1. [What is RAG](docs/01-what-is-rag.md)
2. [Vector Embeddings Explained](docs/02-vector-embeddings-explained.md)
3. [Chunking Strategies](docs/03-chunking-strategies.md)
4. [Retrieval Mechanisms](docs/04-retrieval-mechanisms.md)
5. [RAG Pipeline Architecture](docs/05-rag-pipeline-architecture.md)
6. [Implementation Walkthrough](docs/06-implementation-walkthrough.md)
