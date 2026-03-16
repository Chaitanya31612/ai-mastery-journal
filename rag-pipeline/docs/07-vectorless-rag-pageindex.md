# 07 — Vectorless RAG: How PageIndex Works

> PageIndex is a **vectorless, reasoning-based RAG** system. Instead of embedding documents into vectors and searching by similarity, it builds a hierarchical tree index and uses LLMs to **reason** their way to the right information — like a human expert navigating a textbook.

---

## The Core Problem with Vector RAG

Our pipeline (Phase 2) works like this:

```
Question → Embed → Search for similar vectors → Return top chunks
```

This works well for general questions, but has **5 fundamental limitations**:

### 1. Query–Knowledge Space Mismatch

Vector search assumes the most **similar** text is the most **relevant**. But queries express *intent*, not content.

```
Question: "What are the risks to the company?"

  Vector search finds: "The company has taken on significant risk..."
    → Semantically similar, but this is about past actions, not current risks

  A human would navigate to: "Risk Factors" section on page 15
    → Different words entirely, but exactly what was asked
```

### 2. Similarity ≠ Relevance

In domain-specific documents (financial reports, legal contracts), many sections use near-identical language but differ critically in relevance.

```
Annual Report contains:
  - "Revenue increased 15% in Q3" (page 12)
  - "Revenue increased 12% in Q2" (page 8)
  - "Revenue increased 18% in Q1" (page 4)

Question: "What was Q3 revenue growth?"

Vector search: All three chunks are nearly identical in embedding space.
              May return any of them.

Reasoning:    Reads the ToC → finds "Q3 Results" → navigates directly there.
```

### 3. Hard Chunking Breaks Context

Fixed-size chunks (500 chars) split through sentences, tables, and cross-references:

```
Original table (page 42):
  | Asset Type   | 2023    | 2022    |
  | Cash         | $5.2B   | $4.8B   |
  | Securities   | $12.1B  | $11.3B  |
  | Total        | $17.3B  | $16.1B  |

After chunking: The table might be split across 2-3 chunks,
                making any single chunk meaningless.

PageIndex: Retrieves the full section/page containing the table.
```

### 4. No Chat History Awareness

Vector search treats each query independently:
```
User: "What are the company's total assets?"    → Gets page about assets
User: "And what about liabilities?"             → Has no idea we were
                                                    looking at the same report
```

PageIndex maintains conversation context and knows to look in the same section.

### 5. Can't Follow References

```
Document says: "For details, see Appendix G and Table 5.3"

Vector search: The phrase "see Appendix G" has zero semantic
               similarity to the actual content in Appendix G.
               It cannot follow the reference.

PageIndex:     The LLM reads "see Appendix G", navigates the tree
               to the Appendix G node, retrieves its content.
```

---

## How PageIndex Works

### The Two-Step Process

Unlike vector RAG's embed-then-search, PageIndex uses:

1. **Build a tree index** (once, during ingestion)
2. **Reason over the tree** (each query, using LLM)

```
                    STEP 1: INDEX BUILDING
                    ─────────────────────
  PDF / Markdown
       │
       ▼
  ┌─────────────────────────────┐
  │  LLM reads the document     │
  │  and builds a hierarchical  │
  │  JSON tree (like a smart    │
  │  Table of Contents)         │
  └─────────────────────────────┘
       │
       ▼
  tree_index.json


                    STEP 2: REASONING-BASED RETRIEVAL
                    ─────────────────────────────────
  Question
       │
       ▼
  ┌─────────────────────────────┐
  │  LLM reads the tree index   │──▶ "Financial Stability looks relevant"
  │  and REASONS about which    │
  │  section likely has the     │──▶ "Let me check sub-nodes..."
  │  answer                     │
  │                             │──▶ "Node 0007 discusses vulnerabilities"
  │  (iterative tree traversal) │
  └─────────────────────────────┘
       │
       ▼
  Retrieve full section content for Node 0007
       │
       ▼
  LLM generates grounded answer
```

### The Tree Structure

PageIndex transforms a document into a JSON tree:

```json
{
  "title": "Federal Reserve Annual Report 2023",
  "doc_description": "Yearly overview of monetary policy, financial stability...",
  "nodes": [
    {
      "node_id": "0001",
      "title": "Monetary Policy",
      "start_index": 1,           // page 1
      "end_index": 20,            // page 20
      "summary": "Overview of monetary policy decisions...",
      "nodes": [                  // ← sub-sections
        {
          "node_id": "0002",
          "title": "Interest Rate Decisions",
          "start_index": 1,
          "end_index": 10,
          "summary": "Federal funds rate changes..."
        },
        {
          "node_id": "0003",
          "title": "Quantitative Tightening",
          "start_index": 10,
          "end_index": 20,
          "summary": "Balance sheet reduction program..."
        }
      ]
    },
    {
      "node_id": "0006",
      "title": "Financial Stability",
      "start_index": 21,
      "end_index": 35,
      "summary": "Monitoring and managing systemic risks...",
      "nodes": [
        {
          "node_id": "0007",
          "title": "Monitoring Financial Vulnerabilities",
          "start_index": 22,
          "end_index": 28,
          "summary": "Framework for monitoring systemic risks..."
        }
      ]
    }
  ]
}
```

Each node has:
- **`node_id`** — Unique ID to retrieve the actual content
- **`title`** — Section heading
- **`start_index` / `end_index`** — Page range
- **`summary`** — LLM-generated description of what this section covers
- **`nodes`** — Child sections (recursive, creating the tree)

### The Tree Search Algorithm

Inspired by **AlphaGo's tree search** (Monte Carlo Tree Search), PageIndex navigates the tree iteratively:

```
            ┌──────────────────────┐
            │  Read ToC / tree     │ ◀──────────────────────┐
            │  index structure     │                        │
            └──────────┬───────────┘                        │
                       │                                    │
                       ▼                                    │
            ┌──────────────────────┐                        │
            │  LLM selects most    │                        │
            │  promising section   │               Not enough info?
            └──────────┬───────────┘                Return to tree
                       │                                    │
                       ▼                                    │
            ┌──────────────────────┐                        │
            │  Retrieve section    │                        │
            │  content by node_id  │                        │
            └──────────┬───────────┘                        │
                       │                                    │
                       ▼                                    │
            ┌──────────────────────┐         No             │
            │  Enough to answer?   │ ──────────────────────▶│
            └──────────┬───────────┘
                       │ Yes
                       ▼
            ┌──────────────────────┐
            │  Generate answer     │
            │  with full context   │
            └──────────────────────┘
```

**Key difference from vector search**: The LLM *thinks* about where to look next. It doesn't just calculate similarity — it *reasons* about the document structure, reads summaries, and navigates like a human expert would.

---

## The "In-Context Index" Concept

This is the most important idea in PageIndex:

```
  VECTOR RAG:                          PAGEINDEX:
  ──────────                           ──────────
  Index lives OUTSIDE the LLM          Index lives INSIDE the LLM's context
  (in ChromaDB, FAISS, etc.)          (as JSON in the prompt)

  LLM never "sees" the index           LLM reads and navigates the index
  → Can't reason about structure        → Can reason about document structure

  Search is STATISTICAL                 Search is LOGICAL
  (cosine similarity, kNN)             (LLM reasoning, tree traversal)
```

The JSON tree is small enough to fit in the LLM's context window. The LLM reads it, understands the document's structure, and *decides* which section to drill into — exactly like a human flipping through a table of contents.

---

## Using PageIndex

### Requirements

- **OpenAI API key** (uses GPT-4o for tree building and reasoning)
- **PDF or Markdown** input files

### Quick Setup

```bash
# Clone the repo
git clone https://github.com/VectifyAI/PageIndex.git
cd PageIndex

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
echo "CHATGPT_API_KEY=your-key-here" > .env

# Build a tree index from a PDF
python3 run_pageindex.py --pdf_path /path/to/document.pdf

# Or from Markdown
python3 run_pageindex.py --md_path /path/to/document.md
```

### Configuration Options

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | `gpt-4o-2024-11-20` | OpenAI model for tree building |
| `--toc-check-pages` | `20` | Pages to scan for existing ToC |
| `--max-pages-per-node` | `10` | Max pages in a leaf node |
| `--max-tokens-per-node` | `20000` | Max tokens in a leaf node |
| `--if-add-node-summary` | `yes` | Generate summaries per node |

### Alternative: Cloud API

PageIndex is also available as:
- **Chat platform**: [chat.pageindex.ai](https://chat.pageindex.ai)
- **MCP integration**: [pageindex.ai/mcp](https://pageindex.ai/mcp)
- **REST API**: [docs.pageindex.ai/quickstart](https://docs.pageindex.ai/quickstart)

---

## When to Use What

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| Quick Q&A over short docs | **Vector RAG** | Fast, cheap, good enough |
| Long, structured professional docs | **PageIndex** | Better at navigation and reasoning |
| Need to follow cross-references | **PageIndex** | Can "see Appendix G" and navigate there |
| Multi-turn conversations | **PageIndex** | Chat history awareness |
| Budget-conscious / offline | **Vector RAG** | No LLM API calls for retrieval |
| Real-time / low-latency | **Vector RAG** | ~5ms search vs ~2-5s LLM reasoning |
| Financial/legal/regulatory docs | **PageIndex** | 98.7% on FinanceBench vs ~75% vector RAG |

---

## Key Takeaway

> **Vector RAG** searches for *similar* text.
> **PageIndex** *thinks* about where to look.

Both approaches have their place. Vector RAG is fast, cheap, and works offline. PageIndex is more accurate for complex, structured documents but requires LLM API calls for every retrieval step.

The ideal system might combine both — use vector RAG for fast initial filtering and PageIndex-style reasoning for complex queries that need structural understanding.
