import json
import re
import logging
from typing import List, Dict, Any, Optional
from src.agent.state import SearchState
from src.utils.client_openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

class CombinedEvaluator:
    """
    Combined Evaluator & Conflict Detector in a Single LLM Call per iteration.
    Performs fact extraction, conflict detection, sufficiency evaluation,
    and next-query planning in one structured LLM call.
    """
    def __init__(self, llm_client: Optional[OpenRouterClient] = None):
        self.llm_client = llm_client or OpenRouterClient()

    def evaluate_iteration(
        self,
        state: SearchState,
        newly_read_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        # Build prompt context
        chunks_text_formatted = []
        for c in newly_read_chunks:
            meta = c.get("metadata", {})
            file_name = meta.get("file_name", c.get("file_name", "N/A"))
            page_num = meta.get("page_number", c.get("page_number", 1))
            cid = c.get("chunk_id", f"{file_name}#p{page_num}")
            text_body = c.get("text", "")
            chunks_text_formatted.append(f"--- SOURCE ID: [{cid}] ({file_name}, Page {page_num}) ---\n{text_body}")

        formatted_chunks = "\n\n".join(chunks_text_formatted) if chunks_text_formatted else "(No new chunks retrieved in this iteration)"

        memory_formatted = json.dumps(state.working_memory, indent=2, ensure_ascii=False) if state.working_memory else "[]"
        conflicts_formatted = json.dumps(state.identified_conflicts, indent=2, ensure_ascii=False) if state.identified_conflicts else "[]"
        previous_queries_formatted = json.dumps(state.queries_history, ensure_ascii=False)

        system_prompt = (
            "You are a rigorous Master Archivist analyzing documents from the fantasy 'Ashen Era Archive'.\n"
            "Your task is to evaluate newly retrieved document chunks against the user's question, update working memory, "
            "detect any conflicting facts across documents, and decide if enough information has been gathered.\n\n"
            "YOU MUST RESPOND ONLY WITH A VALID JSON OBJECT matching the schema below. Do not include markdown codeblocks or conversational text outside the JSON."
        )

        user_prompt = f"""
USER QUESTION: "{state.question}"

CURRENT ITERATION: {state.iteration} of {state.max_iterations}
PREVIOUS SEARCH QUERIES: {previous_queries_formatted}

CURRENT WORKING MEMORY (Facts Extracted So Far):
{memory_formatted}

CURRENT IDENTIFIED CONFLICTS:
{conflicts_formatted}

NEWLY RETRIEVED DOCUMENT CHUNKS FOR THIS ITERATION:
{formatted_chunks}

TASK INSTRUCTIONS:
1. Extract new relevant facts from the newly retrieved chunks. State the fact and cite the exact SOURCE ID (e.g., "doc_name.pdf#p3").
2. Check for CONFLICTS/CONTRADICTIONS between newly read chunks and existing working memory (e.g. conflicting dates, names, locations, or numbers). Record conflicting claims with their respective source citations.
3. Judge "Do I have enough information to fully and accurately answer the user question?":
   - Set `has_enough_info` to true ONLY if working memory has conclusive answers.
   - If key details are missing or unclear, set `has_enough_info` to false.
4. If `has_enough_info` is false, formulate 1 to 2 targeted keyword search queries for the next iteration to find the missing information.

REQUIRED JSON RESPONSE SCHEMA:
{{
  "extracted_facts": [
    {{"fact": "Extracted fact description", "source": "filename.pdf#p1#c1"}}
  ],
  "identified_conflicts": [
    {{
      "topic": "Topic of conflict",
      "claim_a": "First claim description",
      "source_a": "filename_a.pdf#p1",
      "claim_b": "Contradictory claim description",
      "source_b": "filename_b.md#p2"
    }}
  ],
  "has_enough_info": boolean,
  "confidence_score": integer (1 to 10),
  "rationale": "Brief rationale for sufficiency judgment",
  "missing_information": ["List of missing facts still needed"],
  "next_suggested_queries": ["Query 1", "Query 2"]
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            raw_response = self.llm_client.generate(messages, json_mode=True, temperature=0.1)
            parsed_result = self._clean_and_parse_json(raw_response)
            return parsed_result
        except Exception as e:
            logger.error(f"[Evaluator Error] LLM evaluation failed: {e}")
            # Fallback evaluation structure if LLM fails or no API keys configured
            return {
                "extracted_facts": [],
                "identified_conflicts": [],
                "has_enough_info": False,
                "confidence_score": 1,
                "rationale": f"Evaluation fallback due to error: {e}",
                "missing_information": ["Unable to parse LLM response"],
                "next_suggested_queries": [state.question]
            }

    @staticmethod
    def _clean_and_parse_json(raw_text: str) -> Dict[str, Any]:
        """
        Clean potential markdown wrappers and parse JSON safely.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Find first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1:
            cleaned = cleaned[start_idx : end_idx + 1]

        return json.loads(cleaned)
