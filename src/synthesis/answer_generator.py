import json
import logging
from typing import Optional
from src.agent.state import SearchState
from src.utils.client_gemini import GeminiClient

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Synthesizes grounded final answers with inline citations and explicit conflict surfacing.
    """
    def __init__(self, llm_client: Optional[GeminiClient] = None):
        self.llm_client = llm_client or GeminiClient()

    @staticmethod
    def _is_image_dependent(state: SearchState) -> bool:
        visual_terms = (
            "image", "portrait", "figure plate", "plate", "illustration",
            "depicting", "depicted", "visual", "holding", "engraved",
        )
        context = " ".join([
            state.question,
            " ".join(state.missing_gaps),
            " ".join(str(f.get("fact", "")) for f in state.working_memory),
        ]).lower()
        return any(term in context for term in visual_terms)

    @classmethod
    def _uncertainty_notice(cls, state: SearchState) -> str:
        if state.has_enough_info and state.last_confidence_score >= 7:
            return ""
        if cls._is_image_dependent(state):
            return (
                "Based on available information, I could not fully confirm this. "
                "The requested detail appears to be in an image/figure plate not covered "
                "by the current text-based ingestion pipeline."
            )
        return (
            "Based on available information, I could not fully confirm the requested detail "
            f"within the search limit of {state.max_iterations} iterations."
        )

    def generate_answer(self, state: SearchState) -> str:
        memory_str = json.dumps(state.working_memory, indent=2, ensure_ascii=False)
        conflicts_str = json.dumps(state.identified_conflicts, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are an expert fantasy scholar and chief archivist of the 'Ashen Era Archive'.\n"
            "Your task is to write a clear, accurate, fully grounded answer to the user's question based strictly on "
            "the provided working memory of extracted facts and identified source conflicts.\n\n"
            "STRICT CITATION & CONFLICT RULES:\n"
            "1. INLINE CITATIONS: Every factual claim in your answer MUST include an inline citation in the format `[Filename, Page X]` or `[DocID]`.\n"
            "2. EXPLICIT CONFLICT SURFACING: If there are conflicting accounts or contradictory dates/names across documents, DO NOT silently pick one. "
            "You MUST create a dedicated section titled `### ⚠️ Source Discrepancies & Conflicts` and explain the exact contradictions and which documents state what.\n"
            "3. NO HALLUCINATIONS: Base your answer ONLY on facts present in the working memory."
        )

        user_prompt = f"""
USER QUESTION: "{state.question}"

EXTRACTED FACTS IN WORKING MEMORY:
{memory_str}

IDENTIFIED SOURCE CONFLICTS:
{conflicts_str}

SEARCH STATUS:
- Has enough information: {state.has_enough_info}
- Last confidence score: {state.last_confidence_score}/10
- Stop reason: {state.stop_reason}
- Missing information: {json.dumps(state.missing_gaps, ensure_ascii=False)}

Please generate the final cited answer now.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            answer = self.llm_client.generate(messages, json_mode=False, temperature=0.2)
            notice = self._uncertainty_notice(state)
            return f"{notice}\n\n{answer}" if notice else answer
        except Exception as e:
            logger.error(f"[Answer Generator Error] {e}")
            # Fallback answer synthesis using working memory directly
            lines = [f"### Answer to: {state.question}\n"]
            notice = self._uncertainty_notice(state)
            if notice:
                lines.append(notice + "\n")
            lines.append("Based on retrieved archive records:\n")
            for fact in state.working_memory:
                lines.append(f"- {fact.get('fact', '')} [{fact.get('source', '')}]")
            
            if state.identified_conflicts:
                lines.append("\n### ⚠️ Source Discrepancies & Conflicts\n")
                for c in state.identified_conflicts:
                    lines.append(f"- **{c.get('topic', 'Conflict')}**: {c.get('claim_a', '')} [{c.get('source_a', '')}] VS {c.get('claim_b', '')} [{c.get('source_b', '')}]")
            
            return "\n".join(lines)
