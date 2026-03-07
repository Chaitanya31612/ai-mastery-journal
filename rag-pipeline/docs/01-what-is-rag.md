# What is RAG? (Retrieval-Augmented Generation)

> **Prerequisites**: Basic understanding of how LLMs work (pre-training, fine-tuning, prompting)
> **Reading time**: ~15 minutes

---

## The Problem: Why LLMs Need Help

You already know from your LLM studies that Large Language Models are trained to predict the next word. They learn language patterns, grammar, and a surprising amount of world knowledge during pre-training. But there are three fundamental limitations:

### 1. Knowledge Cutoff
LLMs only know what was in their training data. If your training data ends in 2024, the model has no idea about events in 2025. Ask "Who won the latest FIFA World Cup?" and you might get an outdated or fabricated answer.

### 2. Hallucination
As you noted in your LLM overview — models learn to generate text, not *factually true* text. They sound confident even when wrong because the training data itself sounds confident. There's no internal "I don't know" mechanism.

### 3. No Access to Private/Custom Data
Even if an LLM is perfectly up-to-date, it can't answer questions about YOUR company's internal docs, YOUR codebase, YOUR medical records. It simply never saw that data during training.

---

## The Solution: Retrieve → Augment → Generate

**RAG** solves all three problems with a brilliantly simple idea:

> Instead of relying on the LLM's internal memory, **find the relevant information first**, then give it to the LLM as context alongside the user's question.

This is exactly what you described in your LLM overview:
> *"Everything that's in the LLM's input sequence is readily available for it to process, while any implicit knowledge it has acquired in pre-training is more difficult and precarious for it to retrieve."*

RAG makes this explicit. Here's the three-step pattern:

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  User Question                                                   │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────┐     ┌──────────┐     ┌────────────┐                │
│  │ RETRIEVE │────▶│ AUGMENT  │────▶│  GENERATE  │                │
│  │          │     │          │     │            │                │
│  │ Search   │     │ Combine  │     │ LLM creates│                │
│  │ relevant │     │ question │     │ answer from│                │
│  │ documents│     │ + context│     │ context    │                │
│  └─────────┘     └──────────┘     └────────────┘                │
│       ▲                                  │                       │
│       │                                  ▼                       │
│  ┌─────────┐                      Grounded Answer                │
│  │ Document │                                                    │
│  │  Store   │                                                    │
│  └─────────┘                                                     │
└──────────────────────────────────────────────────────────────────┘
```

### Step 1: Retrieve
Given a user's question, search through your document store to find the most relevant pieces of text (called "chunks"). This is typically done using **vector similarity search** (covered in detail in `02-vector-embeddings-explained.md`).

### Step 2: Augment
Take the user's original question and combine it with the retrieved context into a prompt. Something like:

```
System: You are a helpful assistant. Answer the user's question based ONLY on
the provided context. If the context doesn't contain the answer, say so.

Context:
[Retrieved chunk 1]
[Retrieved chunk 2]
[Retrieved chunk 3]

User: What is the pre-training phase of LLM development?
```

### Step 3: Generate
Send this augmented prompt to your LLM (Groq, Gemini, Ollama — your AI factory handles this). The model generates an answer that is **grounded** in the actual document content.

---

## The Two Pipelines

A RAG system actually has two separate workflows:

### Ingestion Pipeline (Offline — Done Once Per Document)

```
Documents ──▶ Load ──▶ Chunk ──▶ Embed ──▶ Store in Vector DB

  PDF, MD,      Parse     Split into    Convert to      Save for
  TXT, CSV      content   smaller       numeric         later
  JSON          from      pieces        vectors         retrieval
                files
```

This is the "preparation" step. You process your documents once and store their embeddings in a vector database. When you add new documents, you run them through this pipeline too.

**Each stage has important decisions**:
- **Load**: Different formats need different parsers (PDF extraction, Markdown parsing, etc.)
- **Chunk**: How big should each piece be? Where should you split? (see `03-chunking-strategies.md`)
- **Embed**: Which embedding model? Local or API? (see `02-vector-embeddings-explained.md`)
- **Store**: Which vector database? In-memory or persistent? (see `04-retrieval-mechanisms.md`)

### Query Pipeline (Online — Every Time a User Asks)

```
Question ──▶ Embed ──▶ Search ──▶ Retrieve Top-K ──▶ Augment ──▶ Generate
                        Vector     chunks             prompt       answer
                        DB                            with         using
                                                      context      LLM
```

This happens in real-time. The user asks a question, you embed it using the **same embedding model**, search the vector DB for similar chunks, and pass them to the LLM.

> **Key insight**: The question and the documents must use the **same embedding model** so they exist in the same vector space. You can't embed docs with model A and search with model B.

---

## A Concrete Example

Let's say you have your `HighLevelGeneralOverview.md` indexed in a RAG system.

**User asks**: "Why do LLMs hallucinate?"

**Step 1 — Retrieve**: The system embeds this question and searches the vector DB. It finds these chunks:
- *"the LLM learns only to generate text, not factually true text. Nothing in its training gives the model any indicator of the truth or reliability..."*
- *"text out there on the internet and in books sounds confident, so the LLM of course learns to sound that way, too, even if it is wrong..."*

**Step 2 — Augment**: These chunks are combined with the question into a prompt.

**Step 3 — Generate**: The LLM responds with a grounded answer like:
> "LLMs hallucinate because they are trained to predict the next word, not to produce factually true text. Since most training data sounds confident, the model learns to sound confident too, even when it's wrong. There's no mechanism in the training process that teaches the model to distinguish between true and false information."

This answer is **traceable** — you know exactly which document chunks it came from.

---

## RAG vs Other Approaches

| Approach | How it works | Best for | Limitations |
|----------|-------------|----------|-------------|
| **RAG** | Retrieve relevant docs, add to prompt | Private data, current info, any domain | Retrieval quality depends on chunking/embedding |
| **Fine-tuning** | Retrain model weights on domain data | Teaching model new skills or styles | Expensive, doesn't add factual knowledge reliably |
| **Long context** | Stuff everything into a huge prompt | Small document sets (< 200 pages) | Expensive per query, attention degrades with length |
| **Knowledge graphs** | Structured entity-relationship data | Highly structured domains | Complex to build and maintain |

### When to use RAG:
- ✅ You need answers grounded in specific documents
- ✅ Your data changes frequently
- ✅ You have large document collections
- ✅ You need source attribution for answers
- ✅ You want to use any LLM without retraining

### When NOT to use RAG:
- ❌ You need the model to learn a new skill (use fine-tuning)
- ❌ Your entire knowledge base fits in the context window (just stuff it all in)
- ❌ You need real-time data (use function calling / tools instead)

---

## Key Terminology Quick Reference

| Term | Meaning |
|------|---------|
| **Chunk** | A piece of a document, typically 200-1000 tokens |
| **Embedding** | A numeric vector representing the meaning of text |
| **Vector Store / Vector DB** | Database optimized for storing and searching embeddings |
| **Top-K** | Number of most similar chunks to retrieve (commonly 3-10) |
| **Context Window** | Maximum input size for the LLM (varies by model) |
| **Grounded Response** | An LLM answer based on provided context, not internal memory |
| **Semantic Search** | Finding documents by meaning similarity, not keyword matching |

---

## What's Next?

Now that you understand the big picture, dive into the building blocks:

1. **[02 — Vector Embeddings Explained](./02-vector-embeddings-explained.md)** — How text becomes searchable numbers
2. **[03 — Chunking Strategies](./03-chunking-strategies.md)** — How to split documents effectively
3. **[04 — Retrieval Mechanisms](./04-retrieval-mechanisms.md)** — How to find the right chunks
4. **[05 — RAG Pipeline Architecture](./05-rag-pipeline-architecture.md)** — The complete system design

---

> **Source & Credits**: RAG was introduced by Meta/Facebook AI in 2020 in the paper ["Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"](https://arxiv.org/abs/2005.11401) by Lewis et al. The concept has since become the standard approach for grounding LLMs in external knowledge.
