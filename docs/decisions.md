# Key Technical Decisions

This document records the significant design and technical decisions made while building NeuroForge's Intelligent Document Assistant (Sub-track 1C - Searching the Way a Human Does), the alternatives we considered, and why we chose what we chose.

---

## Decision Log

### D1 - Sub-track Selection: 1C (Iterative, Human-Like Search)
**Decision:** We chose Sub-track 1C over 1A (multimodal answers) and 1B (cross-document linking).
**Why:** `[e.g. our strength is in agent orchestration and reasoning, and the Ashen Era Archive's multi-hop questions (e.g. "which equipment is affected if component Y fails?") fit an iterative search agent better than a single-pass retrieval system.]`
**Alternatives considered:** We initially built against the **OpenAI API**, but hit usage/rate limits during development that blocked testing. We switched to **Gemini's free tier**, but found its outputs were sometimes inconsistent - occasionally uncertain or lower-confidence answers - and it also had its own rate limits that interrupted testing. We ultimately settled on **Gemini Pro (Google API)**, which gave more consistent, higher-confidence reasoning for the agent's search-and-evaluate loop, with a more workable rate limit for our team's development pace.

---

### D2 - LLM Provider & Model
**Decision:** Google **Gemini Pro** (via the Google API) is used as the core reasoning and generation model, driving the agent's planning, reflection, and final answer synthesis.
**Why:**
- Strong long-context reasoning, useful for synthesizing evidence gathered across multiple search hops.
- Native function-calling / tool-use support, which we use to let the model trigger retrieval calls as part of its own reasoning loop rather than through a separate hard-coded controller.
- Competitive free-tier quota for a student competition budget, reducing the need for paid fallback during development and testing.

**Alternatives considered:** We evaluated OpenRouter's free-tier open-weight models (Llama, Qwen, DeepSeek) as a no-cost fallback. These were useful for early prototyping and load-testing the retrieval pipeline without burning Gemini quota, but we found Gemini Pro's reasoning was more reliable at the "do I have enough evidence yet?" decision step, which is the core of sub-track 1C — so we standardized on Gemini Pro for the agent's decision-making, keeping an open-weight model as an optional low-cost fallback for less critical sub-calls.

**Trade-offs:** Gemini Pro is slower and has tighter rate limits than the lightweight open models, so multi-hop queries (3+ search iterations) take longer to run. We accepted this latency cost in exchange for more reliable stopping decisions and fewer hallucinated "I have enough information" false-positives.

---

### D3 - Embedding Model & Vector Store
**Decision:** **Voyage AI** embeddings (`voyage-4` family) are used for indexing and querying the corpus. `[Specify: voyage-4-large for indexing, voyage-4-lite for queries — or confirm if you used a single model throughout.]` Vector storage: `[FAISS / Chroma / pgvector — confirm which one you used.]`

**Why:**
- Voyage AI's 200M free-token allowance for new accounts comfortably covers the ~1,277-page corpus with room for iteration and re-indexing during development.
- All `voyage-4`-series models share the same vector space, so we could embed the corpus once with the larger, higher-quality model and embed queries at inference time with the cheaper, faster `voyage-4-lite` model without needing to re-index.
- `[Add: why this vector store — local/offline, no server dependency, fast similarity search, ease of setup for judges reproducing the project from the README, etc.]`

**Alternatives considered:** `[e.g. OpenAI embeddings — rejected due to cost/quota; Gemini's own embedding model — rejected because Voyage's free allowance and dedicated code/long-document embedding quality were stronger for this use case.]`

---

### D4 - Corpus Ingestion & Chunking Strategy
**Decision:** `[Describe how PDFs, DOCX, markdown, plain text, and simulated scans were parsed. Specify chunk size and overlap, e.g. "~500–800 token chunks with 100-token overlap, split along section/paragraph boundaries rather than fixed character counts."]`
**Why:** `[e.g. structure-aware chunking preserves the semantic unit a fact belongs to, which matters when a single passage needs to be retrieved and later cross-referenced against another document; overlap reduces the chance of splitting a fact across a chunk boundary.]`
**Tables & figures:** `[How were codex tables and figure plates handled — extracted as structured text, image-captioned, or embedded separately?]`
**Scanned ephemera:** `[OCR tool used, e.g. Tesseract / Gemini vision, and any known accuracy issues — cross-reference with docs/limitations.md.]`
**Alternatives considered:** Fixed-size character chunking was tried first as a simpler baseline but tended to split tables and multi-sentence facts mid-chunk, hurting retrieval precision — we moved to structure-aware chunking as a result.
**Trade-offs:** Larger chunks preserve more context per retrieval but reduce precision (more irrelevant text retrieved alongside the relevant fact); smaller chunks are more precise but risk losing the surrounding context a fact depends on. We tuned chunk size empirically against the sample question set.

---

### D5 - Agentic Search Loop Design
**Decision:** The system implements a **plan → retrieve → reflect → decide (answer / search again)** loop, orchestrated around Gemini Pro's function-calling: the model is given a retrieval tool and reasons, in-context, about when and what to search next, rather than following a fixed retrieval pipeline.
**Why:** This is the core requirement of sub-track 1C — a human expert doesn't retrieve once and stop, they read, judge sufficiency, and search again if needed. Using the LLM's own reasoning to drive tool calls (instead of a separately hard-coded controller) lets the search strategy adapt per question rather than following one fixed pattern for every query.
**Stopping criterion:** `[Describe precisely — e.g. "the model is prompted at each step to state its current confidence and list any unresolved sub-questions; the loop terminates when it reports no unresolved sub-questions or after a maximum of N hops, whichever comes first."]`
**Alternatives considered:** A fixed-hop pipeline (always retrieve exactly 3 times) was tested first as a simpler baseline. It was rejected because it wasted calls on simple single-hop questions and still under-retrieved on harder multi-hop ones — see `docs/limitations.md`, Attempt 2.
**Trade-offs:** More hops improve recall on genuinely multi-hop questions but increase latency, token cost, and the risk of the agent drifting toward irrelevant material. We capped hops at `[N]` as a safety limit even when the model wants to keep searching.

---

### D6 - Handling Conflicting / Unreliable Sources
**Decision:** `[Describe the actual mechanism — e.g. "document type is tagged at ingestion time (codex / wiki / novel / ephemera), and the agent is instructed to prefer codex/wiki entries as authoritative when sources conflict, while still surfacing the conflicting claim and its source to the user rather than silently discarding it."]`
**Why:** The Ashen Era Archive is explicitly designed with unreliable sources (e.g. a tavern ballad vs. an official codex entry) that don't always agree — a system that picks one silently would misrepresent the archive; a system that surfaces the disagreement is more useful and more honest.
**Alternatives considered:** `[e.g. cross-source corroboration count (trust a claim more if 2+ independent documents agree) — note whether you used this instead of or alongside document-type weighting.]`

---

### D7 - Answer Grounding & Source Attribution
**Decision:** `[Describe your citation mechanism — e.g. "every claim in the final answer is generated alongside a source tag (document ID + page/section), populated from the retrieved chunk metadata used to support that sentence."]`
**Why:** Verifiability is essential both for judging (judges need to check claims against source documents) and for the assistant's real-world usefulness in an enterprise setting, where an ungrounded answer is a liability.
**Alternatives considered:** `[e.g. a separate "verifier" pass that checks the generated answer against retrieved chunks after the fact, vs. inline citation generated during synthesis.]`

---

### D8 - Evaluation Methodology
**Decision:** `[Describe how you tested against sample_questions.json — e.g. "each of the 20 sample questions was run through the full pipeline; answers were manually graded against the expected answer for correctness, and search transcripts were reviewed to confirm the agent's search paths were sensible, not just the final answer."]`
**Why:** Correctness of the final answer alone doesn't verify the sub-track 1C requirement (genuine multi-step reasoning) — we also needed to confirm the agent was actually searching iteratively rather than getting lucky on a single retrieval pass.
**Alternatives considered:** `[e.g. automated semantic-similarity scoring against reference answers — note if you tried this and why manual review was still needed alongside it.]`

---

### D9 - Cost & Rate-Limit Management
**Decision:** We built retry with exponential backoff into all Gemini and Voyage API calls, cached embeddings and retrieval results to avoid redundant calls during repeated test runs, and distributed development load across each team member's own API keys to stay within free-tier daily limits.
**Why:** Both Gemini's and Voyage's free tiers rate-limit aggressively; without backoff and caching, repeated test runs during development would routinely hit 429 errors, and a demo recording without backoff risks failing live.
**Alternatives considered:** `[e.g. a shared single API key — rejected because it would exhaust free-tier daily quota faster across 4 team members developing in parallel.]`

---



