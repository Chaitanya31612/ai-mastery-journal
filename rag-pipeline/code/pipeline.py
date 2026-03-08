"""
# ============================================================================
# RAG Pipeline — The Orchestrator
# ============================================================================
#
# This is the HEART of the RAG system. It ties together ALL the components
# you've learned about into a single, simple interface:
#
#     pipeline = RAGPipeline(config)
#     pipeline.ingest("path/to/docs/")      # Index documents
#     answer = pipeline.query("My question") # Ask questions
#
# WHAT THIS FILE DOES:
# --------------------
# The RAGPipeline class follows the "Facade Pattern" — it provides a
# simple interface to a complex system. Behind `pipeline.ingest()` and
# `pipeline.query()`, there are loaders, chunkers, embedders, vector
# stores, retrievers, re-rankers, and LLMs all working together.
#
# You don't NEED to use this class — you could wire up the components
# yourself. But this orchestrator handles all the wiring for you.
#
# THE COMPLETE FLOW:
# ------------------
#
#   INGEST FLOW (pipeline.ingest):
#   ┌─────────────────────────────────────────────────────────────┐
#   │ Source path                                                  │
#   │    ↓                                                         │
#   │ DirectoryLoader → [Documents]                                │
#   │    ↓                                                         │
#   │ ChunkingStrategy → [Chunks]                                  │
#   │    ↓                                                         │
#   │ EmbeddingProvider → [Vectors]                                │
#   │    ↓                                                         │
#   │ VectorStore.add(chunks, vectors) → Persisted to disk         │
#   └─────────────────────────────────────────────────────────────┘
#
#   QUERY FLOW (pipeline.query):
#   ┌─────────────────────────────────────────────────────────────┐
#   │ Question                                                     │
#   │    ↓                                                         │
#   │ VectorRetriever → [Top-20 candidates]                        │
#   │    ↓                                                         │
#   │ ReRanker → [Top-5 most relevant]                             │
#   │    ↓                                                         │
#   │ Build augmented prompt (question + context)                  │
#   │    ↓                                                         │
#   │ AIAnalyzer.analyze() → Grounded answer                       │
#   │    ↓                                                         │
#   │ Return answer + source chunks                                │
#   └─────────────────────────────────────────────────────────────┘
#
# ============================================================================
"""


import os
import sys
from dataclasses import dataclass
from typing import Optional

from .config import RAGConfig
from .loaders import DirectoryLoader, MarkdownLoader, TextLoader, Document
from .chunking import create_chunker, Chunk
from .embeddings import SentenceTransformerEmbedder
from .vectorstore import ChromaStore
from .retrieval import VectorRetriever, ReRanker


# ============================================================================
# QueryResult — The Answer Container
# ============================================================================

@dataclass
class QueryResult:
    """
    The result of a RAG query — contains the answer and its sources.

    -----------------------------------------------------------------------
    Attributes:
        answer:   The LLM-generated answer grounded in retrieved context.
        sources:  List of source chunks that were used as context.
        query:    The original question that was asked.
        metadata: Additional info (provider used, tokens, etc.)

    Example:
        result = pipeline.query("What is RAG?")
        print(result.answer)
        for src in result.sources:
            print(f"  From: {src.metadata.get('file_name')} — {src.content[:60]}...")
    -----------------------------------------------------------------------
    """
    answer: str
    sources: list[Chunk]
    query: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self):
        return f"QueryResult(answer='{self.answer[:80]}...', sources={len(self.sources)})"


# ============================================================================
# RAGPipeline — The Main Orchestrator
# ============================================================================

class RAGPipeline:
    """
    The main RAG pipeline that ties everything together.

    -----------------------------------------------------------------------
    This is the facade that you interact with. Behind the scenes, it
    coordinates: loaders → chunkers → embedders → vector store →
    retrievers → re-ranker → LLM.

    Args:
        config: RAGConfig object with all pipeline settings.

    Two main methods:
        - ingest(source) : Load, chunk, embed, and store documents
        - query(question): Retrieve context and generate an answer

    Example:
        from code.config import RAGConfig
        from code.pipeline import RAGPipeline

        config = RAGConfig(chunk_size=500, ai_provider="groq")
        pipeline = RAGPipeline(config)

        # Step 1: Ingest your documents
        pipeline.ingest("path/to/my/docs/")

        # Step 2: Ask questions
        result = pipeline.query("What is pre-training?")
        print(result.answer)
        print(f"Based on {len(result.sources)} source chunks")
    -----------------------------------------------------------------------
    """

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()

        # -----------------------------------------------------------------------
        # Initialize components (lazy where possible)
        # -----------------------------------------------------------------------

        # Embedding provider — loaded lazily on first use
        self._embedder = SentenceTransformerEmbedder(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )

        # Vector store — connects to persistent ChromaDB
        self._vector_store = ChromaStore(
            path=self.config.vectordb_path,
            collection_name=self.config.collection_name,
        )

        # Retriever — wires together embedder and vector store
        self._retriever = VectorRetriever(
            vector_store=self._vector_store,
            embedding_provider=self._embedder,
        )

        # Re-ranker — loaded lazily on first query (if enabled)
        self._reranker = None
        if self.config.use_reranker:
            self._reranker = ReRanker(model_name=self.config.reranker_model)

        # AI analyzer — loaded lazily on first query
        self._ai_analyzer = None

    # ==========================================================================
    #  INGEST — Load, Chunk, Embed, and Store Documents
    # ==========================================================================

    def ingest(self, source: str) -> dict:
        """
        Ingest documents from a file or directory into the vector store.

        -----------------------------------------------------------------------
        This runs the full ingestion pipeline:
            1. LOAD: Read files using the appropriate loader
            2. CHUNK: Split documents into smaller pieces
            3. EMBED: Convert chunks to vectors
            4. STORE: Save to ChromaDB for later retrieval

        Args:
            source: Path to a file or directory to ingest.
                    Supports: .pdf, .md, .txt, .csv, .json
                    If a directory, all supported files are loaded recursively.

        Returns:
            dict with ingestion statistics:
                {
                    "documents_loaded": 5,
                    "chunks_created": 47,
                    "source": "/path/to/docs/",
                }

        Example:
            # Ingest a single file
            stats = pipeline.ingest("my_notes.md")

            # Ingest an entire directory
            stats = pipeline.ingest("docs/")
            print(f"Ingested {stats['documents_loaded']} docs → {stats['chunks_created']} chunks")
        -----------------------------------------------------------------------
        """
        print(f"\n{'='*60}")
        print(f"📥 INGESTING: {source}")
        print(f"{'='*60}")

        # -----------------------------------------------------------------------
        # Step 1: LOAD — Read files into Document objects
        # -----------------------------------------------------------------------
        print("\n📄 Step 1/4: Loading documents...")

        loader = DirectoryLoader(recursive=True, show_progress=True)
        documents = loader.load(source)

        if not documents:
            print("⚠️  No documents found to ingest.")
            return {"documents_loaded": 0, "chunks_created": 0, "source": source}

        print(f"   Loaded {len(documents)} document(s)")

        # -----------------------------------------------------------------------
        # Step 2: CHUNK — Split documents into smaller pieces
        # -----------------------------------------------------------------------
        print(f"\n✂️  Step 2/4: Chunking (strategy: {self.config.chunking_strategy})...")

        chunker = create_chunker(
            strategy=self.config.chunking_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            embedding_model=(
                self._embedder._model if self.config.chunking_strategy == "semantic" else None
            ),
        )

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk(doc)
            all_chunks.extend(chunks)

        print(f"   Created {len(all_chunks)} chunks from {len(documents)} documents")
        if all_chunks:
            avg_size = sum(len(c.content) for c in all_chunks) / len(all_chunks)
            print(f"   Average chunk size: {avg_size:.0f} characters")

        # -----------------------------------------------------------------------
        # Step 3: EMBED — Convert chunk text to vectors
        # -----------------------------------------------------------------------
        print("\n🧮 Step 3/4: Generating embeddings...")

        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embedder.embed(chunk_texts)

        print(f"   Generated {len(embeddings)} embeddings "
              f"({embeddings.shape[1]} dimensions each)")

        # -----------------------------------------------------------------------
        # Step 4: STORE — Save to ChromaDB
        # -----------------------------------------------------------------------
        print("\n💾 Step 4/4: Storing in vector database...")

        self._vector_store.add(all_chunks, embeddings)

        print(f"   Stored {len(all_chunks)} chunks")
        print(f"   Total chunks in database: {self._vector_store.count()}")

        stats = {
            "documents_loaded": len(documents),
            "chunks_created": len(all_chunks),
            "source": source,
        }

        print(f"\n{'='*60}")
        print(f"✅ INGESTION COMPLETE")
        print(f"   {stats['documents_loaded']} documents → {stats['chunks_created']} chunks")
        print(f"{'='*60}\n")

        return stats

    # ==========================================================================
    #  QUERY — Retrieve Context and Generate an Answer
    # ==========================================================================

    def query(self, question: str) -> QueryResult:
        """
        Ask a question and get an answer grounded in your documents.

        -----------------------------------------------------------------------
        This runs the full query pipeline:
            1. RETRIEVE: Find similar chunks using vector search
            2. RE-RANK:  (Optional) Re-score for relevance using cross-encoder
            3. AUGMENT:  Build a prompt with question + retrieved context
            4. GENERATE: Send to LLM and get a grounded answer

        Args:
            question: The user's question as a string.

        Returns:
            QueryResult with the answer, source chunks, and metadata.

        Example:
            result = pipeline.query("What is the transformer architecture?")
            print(result.answer)
            for src in result.sources:
                print(f"  Source: {src.metadata.get('file_name')}")
        -----------------------------------------------------------------------
        """

        # -----------------------------------------------------------------------
        # Step 1: RETRIEVE — Find candidate chunks
        # -----------------------------------------------------------------------
        search_top_k = self.config.search_top_k if self.config.use_reranker else self.config.rerank_top_k

        results = self._retriever.retrieve(
            query=question,
            top_k=search_top_k,
        )

        if not results:
            return QueryResult(
                answer="I couldn't find any relevant information in the indexed documents.",
                sources=[],
                query=question,
                metadata={"status": "no_results"},
            )

        # -----------------------------------------------------------------------
        # Step 2: RE-RANK — Score for relevance (if enabled)
        # -----------------------------------------------------------------------
        if self._reranker and self.config.use_reranker:
            results = self._reranker.rerank(
                query=question,
                results=results,
                top_k=self.config.rerank_top_k,
            )

        # -----------------------------------------------------------------------
        # Step 3: AUGMENT — Build the prompt with context
        # -----------------------------------------------------------------------
        source_chunks = [r.chunk for r in results]
        context = self._build_context(source_chunks)
        augmented_prompt = self._build_prompt(question, context)

        # -----------------------------------------------------------------------
        # Step 4: GENERATE — Send to LLM
        # -----------------------------------------------------------------------
        ai_response = self._generate_answer(augmented_prompt, question)

        return QueryResult(
            answer=ai_response.get("content", "Failed to generate answer."),
            sources=source_chunks,
            query=question,
            metadata={
                "provider": ai_response.get("provider", "unknown"),
                "model": ai_response.get("model", "unknown"),
                "chunks_retrieved": len(results),
                "reranked": self.config.use_reranker,
            },
        )

    # ==========================================================================
    #  PRIVATE HELPERS
    # ==========================================================================

    def _build_context(self, chunks: list[Chunk]) -> str:
        """
        Combine retrieved chunks into a context string for the LLM.

        -----------------------------------------------------------------------
        Each chunk is clearly labeled with its source and index, so the
        LLM can reference specific sources in its answer.

        Example output:
            --- Source: report.pdf (chunk 3/12) ---
            The transformer architecture uses attention mechanisms...

            --- Source: notes.md (chunk 1/5) ---
            Pre-training is the first phase of LLM development...
        -----------------------------------------------------------------------
        """
        context_parts = []

        for chunk in chunks:
            source = chunk.metadata.get("file_name", "unknown")
            index = chunk.metadata.get("chunk_index", "?")
            total = chunk.metadata.get("total_chunks", "?")

            header = f"--- Source: {source} (chunk {index}/{total}) ---"
            context_parts.append(f"{header}\n{chunk.content}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the augmented prompt that gets sent to the LLM.

        -----------------------------------------------------------------------
        The system prompt instructs the LLM to:
            1. Answer ONLY based on the provided context
            2. Say "I don't know" if the context doesn't have the answer
            3. Reference specific sources when possible
            4. Be concise but thorough

        This is the "Augment" step in Retrieve-Augment-Generate.
        -----------------------------------------------------------------------
        """
        return f"""You are a helpful assistant that answers questions based on the provided context documents.

INSTRUCTIONS:
- Answer the question using ONLY the information from the context below.
- If the context doesn't contain enough information to answer, say "Based on the available documents, I don't have enough information to answer this question."
- Reference the source documents when possible (e.g., "According to [source]...").
- Be concise but thorough.
- Do not make up information that isn't in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    def _generate_answer(self, prompt: str, question: str) -> dict:
        """
        Generate an answer using the AI factory.

        -----------------------------------------------------------------------
        Uses the existing AIAnalyzerFactory from code/ai/ to create
        the appropriate AI provider (Groq, Gemini, or Ollama) and
        generate an answer.

        Falls back to returning the raw context if no AI provider is
        available (e.g., no API keys set).
        -----------------------------------------------------------------------
        """

        try:
            # ---------------------------------------------------------------
            # Lazy-load the AI analyzer on first query
            # ---------------------------------------------------------------
            if self._ai_analyzer is None:
                from .ai import AIAnalyzerFactory, AIProvider

                provider_map = {
                    "groq": AIProvider.GROQ,
                    "gemini": AIProvider.GEMINI,
                    "ollama": AIProvider.OLLAMA,
                }

                provider = provider_map.get(
                    self.config.ai_provider,
                    AIProvider.GROQ,  # Default to Groq
                )

                try:
                    if self.config.ai_model:
                        self._ai_analyzer = AIAnalyzerFactory.create(
                            provider, model=self.config.ai_model
                        )
                    else:
                        self._ai_analyzer = AIAnalyzerFactory.create(provider)
                except (ValueError, Exception) as e:
                    # -------------------------------------------------------
                    # If the specified provider fails (no API key, etc.),
                    # try auto-detecting the best available
                    # -------------------------------------------------------
                    print(f"⚠️  Failed to create {self.config.ai_provider} provider: {e}")
                    print("   Trying auto-detect...")
                    self._ai_analyzer = AIAnalyzerFactory.create_default()

            # ---------------------------------------------------------------
            # Generate the answer
            # ---------------------------------------------------------------
            system_prompt = "You are a helpful assistant that answers questions based on provided context."
            response = self._ai_analyzer.analyze(content=prompt, prompt=system_prompt)

            return {
                "content": response.content,
                "provider": response.provider.value,
                "model": response.model,
                "tokens_used": response.tokens_used,
                "success": response.success,
            }

        except Exception as e:
            print(f"⚠️  AI generation failed: {e}")
            return {
                "content": f"[AI generation failed: {e}]\n\nRetrieved context was available but "
                           f"could not generate an answer. Check your AI provider configuration.",
                "provider": "error",
                "model": "none",
            }

    # ==========================================================================
    #  STATUS & UTILITIES
    # ==========================================================================

    def status(self) -> dict:
        """
        Return current pipeline status and statistics.

        -----------------------------------------------------------------------
        Useful for checking what's been indexed and pipeline configuration.

        Example:
            status = pipeline.status()
            print(f"Indexed chunks: {status['total_chunks']}")
            print(f"Embedding model: {status['embedding_model']}")
        -----------------------------------------------------------------------
        """
        return {
            "total_chunks": self._vector_store.count(),
            "embedding_model": self.config.embedding_model,
            "chunking_strategy": self.config.chunking_strategy,
            "chunk_size": self.config.chunk_size,
            "ai_provider": self.config.ai_provider,
            "use_reranker": self.config.use_reranker,
            "vectordb_path": self.config.vectordb_path,
            "collection_name": self.config.collection_name,
        }

    def reset(self) -> None:
        """
        Clear ALL indexed data. Start fresh.

        -----------------------------------------------------------------------
        ⚠️  This permanently deletes all vectors in the collection.
        You'll need to re-ingest documents after calling this.
        -----------------------------------------------------------------------
        """
        print("🗑️  Resetting vector store...")
        self._vector_store.reset()
        print("✅ Vector store cleared. Re-ingest documents to use the pipeline.")
