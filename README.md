# NeuroForge - Intelligent Document Assistant
### SLIIT Codefest 2026 · AI Competition (Powered by IFS)
### Sub-track 1C: Searching the Way a Human Does

An agentic, multi-step retrieval assistant that reasons over the **Ashen Era Archive** - 415 documents, ~1,277 pages of novels, wiki articles, codexes, and in-world ephemera - the way a human researcher does: search, read, decide what's missing, search again, and only then answer.

---

## Table of Contents
- [Overview](#overview)
- [The Challenge](#the-challenge)
- [Our Approach](#our-approach)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Running the Assistant](#running-the-assistant)
- [Evaluation](#evaluation)
- [Limitations & Known Issues](#limitations--known-issues)
- [AI Usage Disclosure](#ai-usage-disclosure)
- [Team](#team)
- [License](#license)

---

## Overview

Real enterprise documentation is scattered, inconsistent, and too large to read in one pass. Some questions can't be answered from a single lookup - they require **iterative reasoning**: search, evaluate what was found, identify the gap, search again, and repeat until there's enough evidence for a complete, well-grounded answer.

**NeuroForge** builds an AI assistant that works this way against the official competition corpus, the **Ashen Era Archive** - a fully invented fantasy franchise (novels, wiki, codexes, ephemera, and scanned images) built specifically so no model's prior knowledge can help. Our system must genuinely reason over the provided documents, resolve conflicting or unreliable sources, and know when it has — and hasn't — gathered enough to answer.

## The Challenge

We chose **Sub-track 1C: Searching the Way a Human Does**:

> Build an assistant that decides where to look and in what order, checks whether it has enough information, and if not, uses what it has learned so far to guide the next search — repeating until it can give a complete answer.

This means our system is judged not on a single retrieval pass, but on its ability to:
1. Plan and re-plan its search strategy based on intermediate findings.
2. Recognize when evidence is incomplete, contradictory, or from an unreliable source.
3. Chain multiple retrieval steps together and justify its reasoning trail.
4. Produce a final answer that is traceable back to the specific documents that support it.

## Our Approach

*(Fill in with your team's actual design — replace this section as your architecture solidifies.)*

- **Retrieval loop**: `[e.g. plan → retrieve → reflect → decide (answer / search again) → repeat]`
- **Stopping criterion**: `[how the agent decides it has "enough" to answer]`
- **Source reliability handling**: `[how conflicting sources, e.g. ballad vs. codex, are weighed or surfaced]`
- **Grounding**: `[how each claim in the final answer is traced back to source documents]`
- **Models used**: `[LLM(s) and embedding model(s), and why]`

## Architecture

```
[Insert / embed your architecture diagram here — see docs/diagrams/]

  User Query
      │
      ▼
  Planner / Reasoning Agent  ──┐
      │                        │  (iterates until sufficient evidence)
      ▼                        │
  Retriever(s)  ───────────────┘
      │
      ▼
  Evidence Synthesis + Source Attribution
      │
      ▼
  Final Answer
```

Full design rationale and decision log: [`docs/architecture.md`](docs/architecture.md) · [`docs/decisions.md`](docs/decisions.md)

## Repository Structure

```
NeuroForge-Codefest2026/
├── .git/
├── README.md
├── docs/
│   ├── architecture.md        # System design and architecture diagram
│   ├── decisions.md           # Key technical decisions and rationale
│   ├── limitations.md         # Known limitations and failure cases
│   └── diagrams/              # Architecture / flow diagrams
├── src/                       # Application source code
├── ai_usage/
│   ├── ai-usage-disclosure.md # Mandatory: tools used, chat logs, decisions
│   ├── skills/
│   ├── claude.md
│   └── context.md
├── configuration-example/     # Example env / config files (no real secrets)
└── submission_report.pdf      # Final submission report (max 5 pages)
```

## Getting Started

### Prerequisites
- `[Python 3.x / Node.js version, etc.]`
- `[Package manager: pip / poetry / npm]`
- API keys for your chosen LLM and embedding providers (see [Configuration](#configuration))

### Installation
```bash
git clone https://github.com/BinaliAssalaarachchi/NeuroForge-Codefest2026.git
cd NeuroForge-Codefest2026

# [e.g.]
python -m venv venv
source venv/bin/activate       # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Configuration

Copy the example config and fill in your own keys — **never commit real keys**:

```bash
cp configuration-example/.env.example .env
```

| Variable | Description |
|---|---|
| `LLM_API_KEY` | API key for the reasoning/generation model |
| `EMBEDDING_API_KEY` | API key for the embedding model |
| `CORPUS_PATH` | Local path to the Ashen Era Archive corpus |

`.env` is listed in `.gitignore` and must never be committed.

## Running the Assistant

```bash
# [e.g.]
python src/main.py --query "Which other equipment is affected if component Y fails?"

# Run against the sample question set
python src/evaluate.py --questions sample_questions.json
```

## Evaluation

We tested against the provided `sample_questions.json` (20 questions) throughout development. Results, sample transcripts, and failure analysis are documented in [`docs/limitations.md`](docs/limitations.md).

## Limitations & Known Issues

See [`docs/limitations.md`](docs/limitations.md) for a full, honest account of what doesn't work yet and why — including approaches we tried and abandoned.

## AI Usage Disclosure

In line with the competition's AI Usage Policy, all AI tools used during development, what they were used for, and which decisions were made by the team (not the AI) are documented in [`ai_usage/ai-usage-disclosure.md`](ai_usage/ai-usage-disclosure.md), alongside exported chat logs.

## Team

**Team Name:** NeuroForge

| Name | Key Contributions |
|---|---|
| `Binali Assalaarachchi` | `Retrieval & Corpus Engineering`|
| `Tharudi Jayasundara` | `Agentic Reasoning & Orchestration`  |
| `Chathuni Piyumali` | `Answer Synthesis, Grounding & Evaluation` |
| `Thiloka Kulathunga` | ` System Integration, Documentation & Submission`  |


*Built for SLIIT Codefest 2026 — AI Competition, powered by IFS.*
