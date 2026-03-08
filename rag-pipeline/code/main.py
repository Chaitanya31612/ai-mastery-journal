"""
# ============================================================================
# RAG Pipeline — CLI Entry Point
# ============================================================================
#
# This is the command-line interface for the RAG pipeline. It lets you
# interact with the pipeline from your terminal:
#
#   python -m code.main ingest path/to/docs/     # Index documents
#   python -m code.main query "My question?"      # Ask questions
#   python -m code.main status                    # Check pipeline status
#   python -m code.main reset                     # Clear all indexed data
#
# HOW THIS IS STRUCTURED:
# -----------------------
# We use Python's argparse module for command-line argument parsing.
# The CLI has four subcommands, each mapping to a pipeline method:
#
#   ingest → pipeline.ingest(source)
#   query  → pipeline.query(question)
#   status → pipeline.status()
#   reset  → pipeline.reset()
#
# ============================================================================
"""

import argparse
import os
import sys

from .config import RAGConfig
from .pipeline import RAGPipeline

# ============================================================================
# Load environment variables from .env file (for API keys)
# ============================================================================
try:
    from dotenv import load_dotenv
    # Look for .env file in the code/ directory (where this file lives)
    _code_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(_code_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"📋 Loaded environment from: {env_path}")
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars


# ============================================================================
# CLI Command Handlers
# ============================================================================
# Each function handles one subcommand. They create the pipeline,
# call the appropriate method, and format the output.
# ============================================================================

def handle_ingest(args):
    """
    Handle the 'ingest' subcommand.

    -----------------------------------------------------------------------
    Loads documents from the given source path and indexes them
    in the vector store.

    Usage:
        python -m code.main ingest path/to/docs/
        python -m code.main ingest my_file.pdf
        python -m code.main ingest ../llm-working/   # Relative path works too

    Options:
        --chunk-size     : Override default chunk size (default: 500)
        --chunk-strategy : Override chunking strategy (default: recursive)
        --collection     : Use a specific vector store collection name
    -----------------------------------------------------------------------
    """
    config = RAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunk_strategy,
        collection_name=args.collection,
        vectordb_path=args.db_path,
    )

    pipeline = RAGPipeline(config)
    stats = pipeline.ingest(args.source)

    return stats


def handle_query(args):
    """
    Handle the 'query' subcommand.

    -----------------------------------------------------------------------
    Searches the indexed documents and generates an answer using the LLM.

    Usage:
        python -m code.main query "What is pre-training?"
        python -m code.main query "How do transformers work?" --top-k 10
        python -m code.main query "Explain RAG" --no-rerank

    Options:
        --top-k      : Number of chunks to retrieve (default: 5)
        --no-rerank  : Disable cross-encoder re-ranking (faster)
        --provider   : AI provider to use (groq, gemini, ollama)
    -----------------------------------------------------------------------
    """
    config = RAGConfig(
        rerank_top_k=args.top_k,
        use_reranker=not args.no_rerank,
        ai_provider=args.provider,
        collection_name=args.collection,
        vectordb_path=args.db_path,
    )

    pipeline = RAGPipeline(config)
    result = pipeline.query(args.question)

    # -----------------------------------------------------------------------
    # Format the output nicely
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"❓ QUESTION: {result.query}")
    print(f"{'='*60}")
    print(f"\n💡 ANSWER:\n{result.answer}")
    print(f"\n{'─'*60}")
    print(f"📚 SOURCES ({len(result.sources)} chunks used):")
    print(f"{'─'*60}")

    for i, chunk in enumerate(result.sources, 1):
        source = chunk.metadata.get("file_name", "unknown")
        chunk_idx = chunk.metadata.get("chunk_index", "?")
        preview = chunk.content[:120].replace("\n", " ")
        print(f"  [{i}] {source} (chunk {chunk_idx})")
        print(f"      \"{preview}...\"")

    print(f"\n{'─'*60}")
    print(f"📊 Metadata: provider={result.metadata.get('provider', '?')}, "
          f"model={result.metadata.get('model', '?')}, "
          f"reranked={result.metadata.get('reranked', '?')}")
    print(f"{'='*60}\n")

    return result


def handle_status(args):
    """
    Handle the 'status' subcommand.

    -----------------------------------------------------------------------
    Shows the current state of the RAG pipeline — how many chunks are
    indexed, which models are being used, etc.

    Usage:
        python -m code.main status
    -----------------------------------------------------------------------
    """
    config = RAGConfig(
        collection_name=args.collection,
        vectordb_path=args.db_path,
    )

    pipeline = RAGPipeline(config)
    status = pipeline.status()

    print(f"\n{'='*60}")
    print(f"📊 RAG Pipeline Status")
    print(f"{'='*60}")
    print(f"  Total indexed chunks  : {status['total_chunks']}")
    print(f"  Embedding model       : {status['embedding_model']}")
    print(f"  Chunking strategy     : {status['chunking_strategy']}")
    print(f"  Chunk size            : {status['chunk_size']} chars")
    print(f"  AI provider           : {status['ai_provider']}")
    print(f"  Re-ranker enabled     : {status['use_reranker']}")
    print(f"  Vector DB path        : {status['vectordb_path']}")
    print(f"  Collection name       : {status['collection_name']}")
    print(f"{'='*60}\n")

    return status


def handle_reset(args):
    """
    Handle the 'reset' subcommand.

    -----------------------------------------------------------------------
    Clears all indexed data. You'll need to re-ingest after this.

    Usage:
        python -m code.main reset
        python -m code.main reset --collection my_other_collection
    -----------------------------------------------------------------------
    """
    config = RAGConfig(
        collection_name=args.collection,
        vectordb_path=args.db_path,
    )

    # -----------------------------------------------------------------------
    # Confirm before deleting
    # -----------------------------------------------------------------------
    pipeline = RAGPipeline(config)
    count = pipeline.status()["total_chunks"]

    if count == 0:
        print("ℹ️  Vector store is already empty.")
        return

    confirm = input(f"⚠️  This will delete {count} indexed chunks. Continue? (y/N): ")
    if confirm.lower() != "y":
        print("❌ Reset cancelled.")
        return

    pipeline.reset()


# ============================================================================
# Argument Parser Setup
# ============================================================================
# Defines the CLI structure with subcommands and their options.
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    -----------------------------------------------------------------------
    Structure:
        python -m code.main <command> [options]

    Commands:
        ingest   — Load and index documents
        query    — Ask questions against indexed docs
        status   — Show pipeline status
        reset    — Clear all indexed data

    Global Options (available for all commands):
        --collection : ChromaDB collection name
        --db-path    : Path to vector database directory
    -----------------------------------------------------------------------
    """
    parser = argparse.ArgumentParser(
        prog="rag-pipeline",
        description="🔍 RAG Pipeline — Retrieve, Augment, Generate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m code.main ingest ./docs/                       # Index all docs in a folder
  python -m code.main ingest report.pdf                    # Index a single PDF
  python -m code.main query "What is RAG?"                 # Ask a question
  python -m code.main query "Explain embeddings" --top-k 3 # Retrieve 3 chunks
  python -m code.main status                               # Check pipeline status
  python -m code.main reset                                # Clear all data
        """,
    )

    # -----------------------------------------------------------------------
    # Global arguments (shared by all subcommands)
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--collection", default="rag_documents",
        help="ChromaDB collection name (default: rag_documents)",
    )
    parser.add_argument(
        "--db-path", dest="db_path", default="./data/vectordb",
        help="Path to vector database directory (default: ./data/vectordb)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -----------------------------------------------------------------------
    # 'ingest' subcommand
    # -----------------------------------------------------------------------
    ingest_parser = subparsers.add_parser(
        "ingest", help="Load and index documents into the vector store",
    )
    ingest_parser.add_argument(
        "source", help="Path to file or directory to ingest",
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Maximum characters per chunk (default: 500)",
    )
    ingest_parser.add_argument(
        "--chunk-overlap", type=int, default=50,
        help="Overlap between chunks in characters (default: 50)",
    )
    ingest_parser.add_argument(
        "--chunk-strategy", default="recursive",
        choices=["fixed", "recursive", "markdown", "semantic"],
        help="Chunking strategy (default: recursive)",
    )

    # -----------------------------------------------------------------------
    # 'query' subcommand
    # -----------------------------------------------------------------------
    query_parser = subparsers.add_parser(
        "query", help="Ask a question against indexed documents",
    )
    query_parser.add_argument(
        "question", help="The question to ask",
    )
    query_parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    query_parser.add_argument(
        "--no-rerank", action="store_true",
        help="Disable cross-encoder re-ranking (faster but less accurate)",
    )
    query_parser.add_argument(
        "--provider", default="groq",
        choices=["groq", "gemini", "ollama"],
        help="AI provider for answer generation (default: groq)",
    )

    # -----------------------------------------------------------------------
    # 'status' subcommand
    # -----------------------------------------------------------------------
    subparsers.add_parser("status", help="Show pipeline status and statistics")

    # -----------------------------------------------------------------------
    # 'reset' subcommand
    # -----------------------------------------------------------------------
    subparsers.add_parser("reset", help="Clear all indexed data")

    return parser


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """
    Parse CLI arguments and dispatch to the appropriate handler.
    """
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # -----------------------------------------------------------------------
    # Dispatch to the right handler based on the subcommand
    # -----------------------------------------------------------------------
    handlers = {
        "ingest": handle_ingest,
        "query": handle_query,
        "status": handle_status,
        "reset": handle_reset,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
