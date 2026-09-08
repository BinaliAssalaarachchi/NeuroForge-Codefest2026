import html
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from markdown_it import MarkdownIt

import config
from src.agent.search_loop import HumanLikeSearchAgent


MARKDOWN = MarkdownIt("commonmark", {"html": False, "breaks": True})


st.set_page_config(
    page_title="NeuroForge Ashen Era Archive",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');

    :root {
        --archive-edge: #0b0517;
        --archive-glow: #2b1454;
        --archive-purple: #5b21b6;
        --archive-violet: #7c3aed;
        --archive-cyan: #67e8f9;
        --archive-emerald: #34d399;
        --archive-amber: #f59e0b;
        --archive-lavender: #f0eaff;
    }

    .stApp {
        background:
            radial-gradient(ellipse 68% 42% at 50% 17%, rgba(124, 58, 237, 0.34) 0%, rgba(43, 20, 84, 0.14) 45%, transparent 78%),
            radial-gradient(ellipse 42% 26% at 50% 26%, rgba(103, 232, 249, 0.12) 0%, transparent 70%),
            var(--archive-edge);
        color: var(--archive-lavender);
    }

    .stApp h1, .stApp h2, .stApp h3 {
        font-family: 'Poppins', ui-sans-serif, system-ui, sans-serif;
        letter-spacing: 0.02em;
    }

    .stApp h1, .stApp h2, .stApp h3, .stApp label {
        color: var(--archive-cyan);
    }

    .stApp, .stApp p, .stApp textarea, .stApp button {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stTextArea textarea {
        background: rgba(15, 7, 30, 0.92);
        color: #ffffff;
        border: 1px solid rgba(103, 232, 249, 0.42);
        border-radius: 10px;
    }

    .stButton > button[kind='primary'] {
        background: var(--archive-violet);
        color: white;
        border: 1px solid var(--archive-violet);
        border-radius: 999px;
        font-weight: 700;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.3);
    }

    .stButton > button[kind='primary']:hover {
        background: #8b5cf6;
        color: white;
        border-color: #8b5cf6;
    }

    .final-answer {
        background: rgba(23, 11, 43, 0.96);
        color: #ffffff;
        border-radius: 12px;
        padding: 24px;
        margin: 0.5rem 0 1rem;
        line-height: 1.6;
        font-size: 1.12rem;
        box-shadow: 0 0 34px rgba(124, 58, 237, 0.42);
    }

    .final-answer p {
        margin: 0 0 0.85rem;
    }

    .final-answer p:last-child {
        margin-bottom: 0;
    }

    .final-answer strong, .final-answer em {
        color: #ffffff;
    }

    .citation-badge {
        display: inline-block;
        margin: 0 6px 6px 0;
        padding: 4px 9px;
        border-radius: 999px;
        background: var(--archive-violet);
        color: white;
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 700;
        border: 1px solid var(--archive-violet);
    }

    .citation-badge.unresolved {
        background: var(--archive-violet);
        border-color: var(--archive-violet);
    }

    .citation-badge:hover {
        filter: brightness(1.12);
        color: white;
    }

    div[data-testid='stExpander'] {
        background: rgba(15, 7, 30, 0.9);
        border: 1px solid rgba(103, 232, 249, 0.25);
        border-radius: 10px;
    }

    div[data-testid='stExpander'] summary p {
        font-family: 'Poppins', ui-sans-serif, system-ui, sans-serif;
        color: var(--archive-cyan);
        font-weight: 700;
    }

    .iteration-progress {
        color: var(--archive-cyan);
        font-weight: 700;
        padding: 0.35rem 0;
    }

    div[data-testid='stAlert'] {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid rgba(245, 158, 11, 0.78);
        color: var(--archive-lavender);
    }

    .source-conflict-title {
        color: var(--archive-amber);
        font-family: 'Poppins', ui-sans-serif, system-ui, sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.6rem 0;
    }

    .verified-badge {
        display: inline-block;
        background: var(--archive-emerald);
        color: #052e22;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 800;
        margin: 0.15rem 0 0.8rem;
    }

    .evidence-card {
        background: rgba(23, 11, 43, 0.94);
        border-left: 2px solid var(--archive-cyan);
        border-radius: 0 10px 10px 0;
        padding: 24px;
        margin: 0.7rem 0;
        color: #ffffff;
        line-height: 1.6;
        font-size: 1rem;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _anchor_for_chunk(chunk_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", chunk_id).strip("-")
    return f"evidence-{safe_id}"


def _citation_targets(answer: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    citations = re.findall(r"\[([^\]]+)\]", answer)
    targets: List[Dict[str, str]] = []
    seen = set()

    for citation in citations:
        citation = citation.strip()
        if not citation or citation in seen:
            continue
        seen.add(citation)
        normalized = citation.lower()
        match = None

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id", ""))
            metadata = chunk.get("metadata", {}) or {}
            file_name = str(metadata.get("file_name", chunk.get("file_name", "")))
            page_number = str(metadata.get("page_number", chunk.get("page_number", "")))
            if normalized == chunk_id.lower():
                match = chunk
                break
            if file_name and file_name.lower() in normalized:
                page_match = re.search(r"page\s*(\d+)", normalized)
                if not page_match or page_match.group(1) == page_number:
                    match = chunk
                    break

        targets.append({
            "citation": citation,
            "href": f"#{_anchor_for_chunk(match.get('chunk_id', ''))}" if match else "#evidence-panel",
            "grounded": "true" if match else "false",
        })

    return targets


def _render_citation_badges(answer: str, chunks: List[Dict[str, Any]]) -> None:
    targets = _citation_targets(answer, chunks)
    if not targets:
        return

    st.markdown("**Citations**")
    badges = []
    for target in targets:
        label = html.escape(target["citation"])
        badge_class = "citation-badge" if target["grounded"] == "true" else "citation-badge unresolved"
        badges.append(
            f'<a href="{target["href"]}" class="{badge_class}">[{label}]</a>'
        )
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def _render_evidence(chunks: List[Dict[str, Any]]) -> None:
    with st.expander("Evidence", expanded=True):
        st.markdown('<a id="evidence-panel"></a>', unsafe_allow_html=True)
        if not chunks:
            st.info("No document chunks were retrieved.")
            return

        for index, chunk in enumerate(chunks, start=1):
            chunk_id = str(chunk.get("chunk_id", f"chunk-{index}"))
            metadata = chunk.get("metadata", {}) or {}
            file_name = metadata.get("file_name", chunk.get("file_name", "Unknown source"))
            page_number = metadata.get("page_number", chunk.get("page_number", "Unknown"))
            text = chunk.get("text", chunk.get("raw_content", ""))
            anchor = _anchor_for_chunk(chunk_id)

            st.markdown(f'<a id="{anchor}"></a>', unsafe_allow_html=True)
            st.markdown(f"**{index}. {file_name} — Page {page_number}**")
            st.caption(f"Chunk ID: `{chunk_id}`")
            evidence_text = html.escape(str(text)).replace("\n", "<br>")
            st.markdown(f'<div class="evidence-card">{evidence_text}</div>', unsafe_allow_html=True)


def _render_conflicts(conflicts: List[Dict[str, Any]]) -> None:
    if not conflicts:
        return

    st.warning("Source Conflicts")
    for conflict in conflicts:
        st.markdown(f'<div class="source-conflict-title">{html.escape(str(conflict.get("topic", "Conflict")))}</div>', unsafe_allow_html=True)
        st.markdown(
            f"- {conflict.get('claim_a', '')}  \n"
            f"  Source: `{conflict.get('source_a', '')}`  \n"
            f"- {conflict.get('claim_b', '')}  \n"
            f"  Source: `{conflict.get('source_b', '')}`"
        )


def main() -> None:
    st.title("NeuroForge Ashen Era Archive")
    st.write("Ask a question about the archive and inspect the grounded evidence behind the answer.")

    question = st.text_area(
        "Question",
        placeholder="For example: Which faction won the War of Drowned Light?",
        height=90,
    )
    search_clicked = st.button("Search", type="primary", disabled=not question.strip())

    if not search_clicked:
        return

    progress = st.empty()
    question_id = uuid.uuid4().hex[:8]
    agent = HumanLikeSearchAgent(max_iterations=5)

    def on_iteration(iteration: int, maximum: int) -> None:
        progress.markdown(
            f'<div class="iteration-progress">Search iteration {iteration}/{maximum}...</div>',
            unsafe_allow_html=True,
        )

    try:
        state = agent.run(question.strip(), question_id=question_id, on_iteration=on_iteration)
        progress.markdown(
            f'<div class="iteration-progress">Search complete after {state.iteration} iteration(s).</div>',
            unsafe_allow_html=True,
        )
    except Exception as error:
        progress.empty()
        st.error(f"Search failed: {error}")
        return

    st.markdown("## Final Answer")
    answer_html = MARKDOWN.render(state.final_answer)
    st.markdown(
        f'<div class="final-answer">{answer_html}</div>',
        unsafe_allow_html=True,
    )

    if state.has_enough_info and state.last_confidence_score >= 7 and not state.identified_conflicts:
        st.markdown('<div class="verified-badge">Verified, no conflicts</div>', unsafe_allow_html=True)

    if re.search(r"image/figure plate not covered|text-based ingestion pipeline", state.final_answer, re.I):
        st.warning("This answer requires an image/figure plate not covered by text ingestion.")

    _render_citation_badges(state.final_answer, state.retrieved_chunks)
    _render_conflicts(state.identified_conflicts)
    _render_evidence(state.retrieved_chunks)

    trace_path = config.TRACES_DIR / f"question_{state.question_id}.json"
    with st.expander("Full Trace", expanded=False):
        if trace_path.exists():
            st.json(json.loads(trace_path.read_text(encoding="utf-8")))
        else:
            st.json({
                "question_id": state.question_id,
                "question": state.question,
                "stop_reason": state.stop_reason,
                "iteration_traces": state.trace_logs,
            })


if __name__ == "__main__":
    main()
