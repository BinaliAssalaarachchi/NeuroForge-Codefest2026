# Limitations  (What Works, What Doesn't, and What We Tried)

This is the honest account of our system's capabilities and gaps, as required by the rubric's **Technical judgment & decisions** criterion. It should be updated continuously during development, not written retroactively before submission.

---

## What Works

- **Hybrid retrieval (BM25 + Dense + RRF).** Combining sparse (BM25) and dense (Voyage embeddings via ChromaDB) search through Reciprocal Rank Fusion reliably outperforms either method alone — BM25 catches exact names, item IDs, and terminology from the archive that embeddings sometimes miss; dense search catches paraphrased or semantically related questions that don't share exact wording with the source text.
- **Iterative search loop.** The HumanLikeSearchAgent → HybridRetriever → CombinedEvaluator loop successfully performs multiple search hops on multi-hop questions (e.g. "which other equipment is affected if component Y fails?"), re-querying based on what SearchState is still missing rather than answering from a single retrieval pass.
- **Confidence-gated stopping.** The CombinedEvaluator's "confidence ≥ 7" gate correctly stops the loop early on simple, single-hop questions (saving API calls and latency) while continuing to search on harder multi-hop questions — `[insert actual figures once measured, e.g. "average 1.4 hops on simple questions vs. 3.2 hops on multi-hop questions in sample_questions.json"]`.
- **Caching and rate-limit resilience.** The `.cache` layer and exponential backoff mean repeated test runs and demo recordings don't fail from Gemini/Voyage free-tier 429 errors.
- **Multi-interface access.** The same search agent works correctly from all three entry points (Streamlit UI, CLI, and the shared `query_search.py` utility), confirming the agent logic is properly decoupled from any single interface.

---

## What Doesn't Work / Partial

- **Fixed confidence threshold.** The "confidence ≥ 7" gate is a single hardcoded threshold applied to every question type. It isn't calibrated per question difficulty, so it can under-search genuinely hard multi-hop questions that *feel* confident to the evaluator too early, and over-search simple questions that hit an unlucky low-confidence read on the first pass.
- **Conflicting source resolution is shallow.** When the archive's sources disagree (e.g. a tavern ballad vs. an official codex entry), the system currently `[describe actual behavior — e.g. "leans on whichever source ranks higher in the fused retrieval results, rather than explicitly reasoning about source reliability"]`. It does not yet have a dedicated reliability-weighting step distinct from retrieval ranking.
- **BM25 index staleness risk.** `data/bm25_index.pkl` and `data/chroma_db` are built once during ingestion; there's no incremental update path, so any corpus change requires a full re-index rather than an incremental one.
- **Search hop ceiling.** `[Confirm and state your actual max-hop cap, e.g. "capped at 5 hops"]` — on the hardest questions requiring evidence from more documents than that, the loop terminates at the cap and answers from incomplete evidence rather than flagging the answer as potentially incomplete.
- **OCR reliability on simulated scans.** `[Confirm — if DocumentParser doesn't yet include OCR support for scanned ephemera images, note this explicitly here: e.g. "The DocumentParser handles PDF/DOCX/MD/TXT but does not currently OCR image-based scanned ephemera, so facts that exist only in scanned form are not indexed."]`

---

## Approaches We Tried and Abandoned

### Attempt 1: OpenAI API, then Gemini free tier
- **What we tried:** Started development on the OpenAI API for reasoning/generation, later switched to Gemini's free tier.
- **Why it failed:** OpenAI usage limits were hit early in development, blocking testing. Gemini's free tier then produced inconsistent answers — sometimes low-confidence or uncertain — and had its own rate limits that interrupted iterative testing.
- **What we learned / changed:** Moved to Gemini Pro via the Google API, which gave more reliable, consistent reasoning for the CombinedEvaluator's confidence-scoring step and a rate limit workable for four team members developing in parallel.

### Attempt 2: Single-pass retrieval (no agent loop)
- **What we tried:** Retrieve top-k chunks once via dense search only, then generate an answer directly.
- **Why it failed:** Multi-hop questions requiring facts from more than one document were answered incompletely or incorrectly, since no single retrieval pass surfaced the full evidence chain.
- **What we learned / changed:** Led directly to the iterative HumanLikeSearchAgent + CombinedEvaluator loop described in `architecture.md` and `decisions.md` (D1, D5).

### Attempt 3: Dense-only retrieval (no BM25)
- **What we tried:** Relied solely on Voyage embeddings + ChromaDB for retrieval, without a keyword index.
- **Why it failed:** Missed exact-match queries for specific names, item/component IDs, and archive-specific terminology that don't embed distinctively from surrounding text.
- **What we learned / changed:** Added the BM25 Keyword Index and combined it with dense search via Reciprocal Rank Fusion (`decisions.md`, D3/D5).

### Attempt 4: Fixed-hop retrieval loop
- **What we tried:** A fixed number of search iterations (e.g. always exactly N hops) regardless of question difficulty.
- **Why it failed:** Wasted API calls and latency on simple questions, while still sometimes under-retrieving on the hardest multi-hop ones.
- **What we learned / changed:** Replaced with the CombinedEvaluator's dynamic confidence-gated stopping criterion.

---

## Known Edge Cases / Failure Modes

- Questions relying on a single, rarely cross-referenced ephemera document can fail to trigger further search, since low corroboration elsewhere gives the evaluator no signal that more searching would help.
- Character/faction name overlaps across novels, wiki, and codex entries occasionally cause the HybridRetriever to surface passages about the wrong entity when names are ambiguous or reused.
- `[Add any other failure modes observed during testing against sample_questions.json.]`

---

## Constraints That Shaped the System

- **Free-tier API limits.** Gemini and Voyage AI free-tier rate limits capped how much large-scale hop testing we could do during development; caching and backoff (D9 in `decisions.md`) mitigate but don't eliminate this.
- **Two-week competition window.** `[Note anything deprioritized due to time — e.g. incremental index updates, per-question-type threshold tuning, dedicated source-reliability scoring.]`

---

## If We Had More Time

- Calibrate the confidence threshold per question type instead of using a single fixed value.
- Add an explicit source-reliability scoring step, separate from retrieval rank, to better resolve conflicting archive sources.
- Add incremental re-indexing so corpus updates don't require a full rebuild of `chroma_db` and `bm25_index.pkl`.
- `[Add any other genuine next steps your team has identified.]`
