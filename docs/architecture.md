# System Architecture

## Diagram

<img width="1600" height="657" alt="image" src="https://github.com/user-attachments/assets/cdcf1b26-ba4c-40fe-9b22-95ae7976533e" />

## Overview

NeuroForge's Intelligent Document Assistant (Sub-track 1C) is built around an **iterative, hybrid-retrieval search agent** that decides for itself when it has enough evidence to answer, rather than retrieving once and stopping. The system has four main parts: an offline ingestion pipeline, the iterative search agent, the application interfaces, and persistent storage.

## 1. Offline Ingestion Pipeline

Runs once (or on corpus update) to prepare the Ashen Era Archive for retrieval.

1. **Corpus Documents** — raw PDF, DOCX, MD, and TXT files from the archive.
2. **DocumentParser** — extracts text (and structure) from each format.
3. **Chunker** — splits parsed documents into retrieval-sized chunks.
4. Chunks are indexed two ways in parallel:
   - **Voyage Document Embeddings** → **ChromaDB Vector Store** (dense/semantic index)
   - **BM25 Keyword Index** (sparse/lexical index)

This dual indexing is what enables hybrid retrieval at query time — dense embeddings catch semantic/paraphrased matches, BM25 catches exact terms, names, and identifiers that embeddings can miss.

## 2. Iterative Search Agent

The core of the sub-track 1C solution — a closed reasoning loop, not a single-pass pipeline.

- **HumanLikeSearchAgent** — the orchestrator. Decides what to search for next and drives the loop.
- **HybridRetriever** — given a query, runs:
  - **BM25 Search** against the BM25 Keyword Index
  - **Dense Vector Search** against the ChromaDB Vector Store
  - Combines both result sets via **Reciprocal Rank Fusion**, producing a single ranked result list that benefits from both retrieval methods.
- **SearchState** — tracks what has been retrieved and reasoned about so far across search iterations (accumulated evidence, prior queries, prior findings).
- **CombinedEvaluator** — reviews the current SearchState and asks: *do we have enough information and confidence to answer?*
  - Evaluates against the decision gate: **"Enough information and confidence ≥ 7?"**
  - **No** → loop back to HumanLikeSearchAgent, which formulates the next search query based on what's still missing, and the cycle repeats.
  - **Yes** (or a stopping limit is reached) → proceed to answer generation.
- **AnswerGenerator** — synthesizes the final answer from the accumulated SearchState evidence once the loop terminates.

This loop is what distinguishes the system from a standard RAG pipeline: retrieval, evaluation, and re-querying happen in a cycle, driven by the agent's own assessment of sufficiency and confidence — mirroring how a human researcher searches, reads, and decides whether to dig further.

## 3. Application Interfaces

Three entry points, all routing into the same search agent:

- **Streamlit UI** (`app.py`) — primary interactive interface for demos.
- **Command Line Interface** (`main.py`) — direct CLI access for testing/scripting.
- **Search Utility** (`scripts/query_search.py`) — used by both UI and CLI as the shared entry point into the agent.

## 4. External AI Services

- **Google Gemini API** — reasoning, evaluation, and answer generation (drives HumanLikeSearchAgent, CombinedEvaluator, and AnswerGenerator).
- **Voyage AI API** — embedding generation for both document indexing and query-time dense search.

## 5. Persistent Storage

- `data/chroma_db` — persisted dense vector index.
- `data/bm25_index.pkl` — persisted sparse keyword index.
- `.cache` — caches intermediate results (e.g. embeddings, evaluator outputs) to reduce redundant API calls across runs.
- `logs/traces` — logs of search traces (queries issued, evidence retrieved, evaluator decisions per hop), used for debugging and for demonstrating the agent's reasoning process to judges.

## Data Flow Summary

```
Corpus → Parse → Chunk → [Embed → ChromaDB] + [Index → BM25]
                                                    │
User Query → UI/CLI/Search Utility → HumanLikeSearchAgent
                                            │
                                     HybridRetriever
                                  (BM25 + Dense + RRF)
                                            │
                                       SearchState
                                            │
                                    CombinedEvaluator
                                     confidence ≥ 7?
                                    /              \
                                  No                Yes / limit reached
                                  │                       │
                          (loop: new query)        AnswerGenerator → User
```

## Why This Design

The hybrid retriever + iterative evaluator loop directly targets the sub-track 1C requirement: a question like *"which other equipment is affected if component Y fails?"* often cannot be answered from a single retrieval pass, because the full chain of facts is spread across multiple documents. By looping — search, evaluate confidence, search again if the confidence gate isn't met — the agent builds up the full evidence chain before answering, rather than answering prematurely from a partial first retrieval.

See [`decisions.md`](decisions.md) for the reasoning behind specific component choices (e.g. why hybrid retrieval over dense-only, why confidence ≥ 7 as the stopping threshold) and [`limitations.md`](limitations.md) for where this design still falls short.
