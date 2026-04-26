# Agentic RAG System — Task 3

A fully local, privacy-preserving **Conversational Agentic RAG** chatbot built with LangChain, LangGraph, and Ollama. The system retrieves from a 211K-chunk Wikipedia knowledge base, decides autonomously when retrieval is needed, falls back to live web search for current events, and remembers conversation history across turns.

---

## Use Case

**General-knowledge conversational Q&A chatbot** that:
- Answers factual questions from a local Wikipedia knowledge base
- Decides autonomously whether retrieval is needed (agent decision layer)
- Falls back to live web search when local knowledge is insufficient
- Remembers full conversation history across multiple turns
- Critiques and verifies its own answers to reduce hallucinations

---

## System Architecture

```
User question
      │
      ▼
┌─────────────────────┐
│  Node 1 — Retrieve  │◄──── ChromaDB (211K chunks) + bge-reranker-base
│  Semantic search +  │      meta-question? → skip to Generate
│  cross-encoder rank │
└────────┬────────────┘
         │ top-3 chunks
         ▼
┌─────────────────────┐
│  Node 2 — Grade     │  LLM judges: is context relevant?
│  yes / no / memory  │
└──┬──────────────┬───┘
   │ grade=yes    │ grade=no
   │              ▼
   │    ┌──────────────────┐
   │    │ Node 3 — Web     │◄──── Tavily Search API (live web)
   │    │ Search fallback  │
   │    └────────┬─────────┘
   │             │ web context
   ▼             ▼
┌─────────────────────┐
│  Node 4 — Generate  │◄──── llama3.1:8b + conversation history (MemorySaver)
│  LLM answer with    │
│  context + memory   │
└────────┬────────────┘
         │ answer
         ▼
┌─────────────────────┐
│  Node 5 — Critique  │  Self-assess: is answer grounded?
│  good / retry       │
└────────┬────────────┘
         │ good → END / retry → web_search (max 1x)
         ▼
   Final answer
```

---

## Dataset and Source Justification

| Dataset | Source | Size | Purpose |
|---|---|---|---|
| Wikipedia 2023 (English) | `wikimedia/wikipedia` via HuggingFace | 211,190 chunks from 5,000 articles | Primary knowledge base — broad factual coverage, free, standard RAG benchmark corpus |
| CoQA | HuggingFace `datasets` | 108,647 Q&A pairs from 7,199 conversations | Evaluation set — conversational Q&A with follow-ups, matches multi-turn use case |
| Tavily Search API | Live web API | Real-time | Web fallback — handles current events beyond 2023 Wikipedia snapshot |

**Chunking:** RecursiveCharacterTextSplitter — chunk size 512, overlap 64, separators: paragraph → line → sentence → word.

---

## Retrieval Pipeline and Indexing

### Stage 1 — Dense Retrieval
- **Embedding model:** `BAAI/bge-m3` (1024-dim, GPU)
- **Vector store:** ChromaDB (persistent, local)
- **Search:** cosine similarity, top-10 candidates retrieved

### Stage 2 — Cross-Encoder Reranking
- **Reranker:** `BAAI/bge-reranker-base` (CPU — preserves VRAM for LLM)
- Top-10 chunks scored jointly with query, top-3 returned
- Typical top-1 reranker scores: 0.92–0.999 on matched queries

---

## Agent Decision and Reasoning Logic

The agent is a **LangGraph StateGraph** with typed state and conditional routing:

```python
class AgentState(TypedDict):
    messages    : List   # full conversation history (enables memory)
    question    : str    # current user question
    context     : str    # retrieved chunks
    answer      : str    # generated answer
    search_type : str    # 'local' | 'web_needed' | 'web' | 'memory'
    iterations  : int    # loop counter — max 1 retry
```

**Routing logic:**

| After node | Condition | Route to |
|---|---|---|
| Grade | `search_type == 'local'` | Generate |
| Grade | `search_type == 'memory'` (meta-question) | Generate |
| Grade | `search_type == 'web_needed'` | Web Search |
| Critique | `iterations >= 1` | END |
| Critique | answer empty | Web Search |

**Meta-question detection:** If the question contains keywords like "previous", "first", "asked", "said", the Retrieve node skips ChromaDB entirely and sets `search_type='memory'`. The Generate node then answers purely from `messages` history.

---

## Prompt and Control-Flow Strategy

### Grade prompt
```
Does this context contain information to answer the question?
Question: {question}
Context: {context}
Rules:
- Reply 'yes' if context has relevant facts
- Reply 'no' if context is unrelated
- Reply with ONLY one word: yes or no
```

### Generate prompt
```
System: You are a helpful conversational assistant. Answer using the 
provided context. Use conversation history for meta-questions. 
Stay focused on the current question only.

[last 4 messages of history]

Context: {retrieved_chunks}
Question: {question}
Answer from context:
```

### Critique prompt
```
Is this answer grounded in the context?
Question: {question}  Context: {context}  Answer: {answer}
Reply ONLY: good or retry
```

### Control flow by scenario

| Scenario | Path |
|---|---|
| Factual Q — local KB sufficient | Retrieve → Grade(yes) → Generate → Critique → END |
| Current event — not in KB | Retrieve → Grade(no) → Web Search → Generate → Critique → END |
| Follow-up / memory question | Retrieve(skip) → Grade(memory) → Generate(history) → END |
| Multi-hop combining sources | Retrieve → Grade → [Web] → Generate(history+context) → END |

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| LLM | Llama 3.1 8B via Ollama | llama3.1:8b |
| Embedding model | BAAI/bge-m3 (GPU) | sentence-transformers |
| Reranker | BAAI/bge-reranker-base (CPU) | sentence-transformers |
| Vector store | ChromaDB (local, persistent) | 1.5.6 |
| Agent framework | LangGraph | 1.1.6 |
| RAG framework | LangChain | 1.2.15 |
| Web search | Tavily Search API | langchain-tavily |
| ML backend | PyTorch + CUDA | 2.6.0+cu124 |
| Hardware | NVIDIA RTX 3050 6GB, Ubuntu 25.04 | CUDA 12.4 |

---

## Hallucination Mitigation Strategy

Four complementary mechanisms:

1. **Context injection** — LLM always receives retrieved chunks. System prompt explicitly says "answer from context only."
2. **Relevance grading** — If retrieved context is irrelevant (grade=no), the system fetches better context (web) instead of generating from LLM's parametric memory.
3. **Self-critique** — After generation, the Critique node verifies the answer is grounded in the retrieved context. Flags ungrounded answers as "retry."
4. **Source-aware prompting** — Retrieved chunks include `[Source: Wikipedia Title]` metadata, reducing fabricated attributions.

---

## Evaluation Results (from `04_evaluation.ipynb`)

| Category | Queries | Route | Result |
|---|---|---|---|
| Factual single-hop | 5 | local | Hall effect, relativity, photosynthesis — correct ✅ |
| Web fallback | 3 | web | 2025 AI models, Strait of Hormuz — correct ✅ |
| Multi-turn memory | 4 | local/memory | Einstein follow-ups, history recall — correct ✅ |
| Multi-hop | 3 | local+web+memory | Backprop + 2025 architectures synthesis — correct ✅ |
| Hallucination | 3 | various | Zara Qasim: "no info found" (no fabrication) ✅ |

**Retrieval quality (reranker scores):**

| Query | Top-1 | Top-2 | Top-3 |
|---|---|---|---|
| What is machine learning? | 0.999 | 0.995 | 0.693 |
| How does quantum tunneling work? | 0.996 | 0.743 | 0.316 |
| What is the Hall effect? | 0.926 | 0.658 | 0.658 |
| What causes earthquakes? | 0.991 | 0.466 | 0.322 |

**Route distribution:** 8 local · 9 web · 1 memory  
**Average latency:** 74.3s per query (CPU embedding + local LLM)

---

## Failure Cases

| Failure | Example | Cause | Impact |
|---|---|---|---|
| Pronoun ambiguity | "Where was he born?" after Einstein retrieved wrong chunks | bge-m3 embeds pronoun without context | Low — still answered from history |
| Conservative critique | Correct answer marked "retry" | LLM critiques uncertainty language as ungrounded | Low — 1 retry max prevents loops |
| CoQA follow-ups | Context-dependent questions fail standalone | CoQA requires conversation history | Expected — these are multi-turn evals |

---

## Setup Instructions

### Prerequisites
- Ubuntu 22.04+ or WSL2
- NVIDIA GPU with CUDA 12.1+ (or CPU — slower)
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Clone and create environment

```bash
git clone https://github.com/your-username/agentic-rag.git
cd agentic-rag

python3 -m venv rag-env
source rag-env/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch (match your CUDA version)

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install dependencies

```bash
pip install langchain langchain-community langchain-core \
            langchain-ollama langchain-huggingface langchain-chroma \
            langchain-tavily langgraph chromadb \
            sentence-transformers transformers==4.48.0 \
            datasets wikipedia-api arxiv trafilatura \
            tavily-python jupyter ipykernel tqdm pandas numpy \
            python-dotenv FlagEmbedding
```

### 4. Pull the LLM

```bash
ollama pull llama3.1:8b
```

### 5. Register Jupyter kernel

```bash
python -m ipykernel install --user --name=rag-env --display-name "RAG Project"
jupyter notebook
```

---

## Generating the Knowledge Base (chroma_db not included in repo)

The `data/chroma_db/` vector store is ~2-4GB and cannot be uploaded to GitHub. Run the notebooks in order to regenerate it:

### Step 1 — Run `01_ingestion.ipynb`

This notebook:
- Streams 5,000 Wikipedia articles from HuggingFace
- Chunks them into 211,190 passages (512 tokens, 64 overlap)
- Embeds each chunk with `BAAI/bge-m3` on GPU
- Stores everything in ChromaDB at `data/chroma_db/`
- Downloads CoQA evaluation dataset to `data/processed/coqa_eval.csv`

```bash
# Run in Jupyter — select "RAG Project" kernel
# Open: notebooks/01_ingestion.ipynb
# Run All Cells
```

### Step 2 — Run `02_basic_rag.ipynb`

Builds and tests the basic RAG chain with reranker. No data generation — reads from existing `chroma_db`.

### Step 3 — Run `03_agentic_rag.ipynb`

Builds the full LangGraph agent. Requires `chroma_db` from Step 1 and Tavily API key.

### Step 4 — Run `04_evaluation.ipynb`

Runs all 18 evaluation queries and generates `data/processed/eval_results.csv`.

---

## Notebooks

| Notebook | Description |
|---|---|
| `01_ingestion.ipynb` | Data loading, chunking, embedding, ChromaDB indexing |
| `02_basic_rag.ipynb` | Basic RAG chain with retriever, reranker, and LLM |
| `03_agentic_rag.ipynb` | Full LangGraph agent with memory, web fallback, self-critique |
| `04_evaluation.ipynb` | Evaluation on 18 queries across 5 categories |
