import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import config
from src.agent.state import SearchState
from src.agent.evaluator import CombinedEvaluator
from src.retrieval.hybrid_search import HybridRetriever
from src.synthesis.answer_generator import AnswerGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SearchLoop")

class HumanLikeSearchAgent:
    """
    Self-Directed Iterative Search Agent ('Searching the Way a Human Does').
    Implemented with pure Python control flow without black-box framework abstractions.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        evaluator: Optional[CombinedEvaluator] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        max_iterations: int = 5
    ):
        self.retriever = retriever or HybridRetriever()
        self.evaluator = evaluator or CombinedEvaluator()
        self.answer_generator = answer_generator or AnswerGenerator()
        self.max_iterations = max_iterations

    def run(
        self,
        question: str,
        question_id: Optional[str] = None,
        on_iteration: Optional[Callable[[int, int], None]] = None,
    ) -> SearchState:
        if not question_id:
            question_id = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]

        state = SearchState(
            question_id=question_id,
            question=question,
            max_iterations=self.max_iterations
        )

        logger.info(f"\n==================================================")
        logger.info(f"🔎 Starting Agent Search for QID [{question_id}]: '{question}'")
        logger.info(f"==================================================")

        current_queries = [question]
        stagnant_iterations = 0

        while state.iteration <= state.max_iterations and not state.is_complete:
            if on_iteration:
                try:
                    on_iteration(state.iteration, state.max_iterations)
                except Exception as callback_error:
                    logger.warning("Iteration callback failed: %s", callback_error)
            logger.info(f"\n--- Iteration {state.iteration}/{state.max_iterations} ---")
            logger.info(f"Exec Search Queries: {current_queries}")
            state.queries_history.extend(current_queries)

            # 1. Retrieve chunks using Hybrid RRF Search across all current queries
            iteration_chunks: List[Dict[str, Any]] = []
            for query in current_queries:
                retrieved = self.retriever.search(query, top_k=6)
                for chunk in retrieved:
                    cid = chunk["chunk_id"]
                    if cid not in state.retrieved_chunk_ids:
                        state.retrieved_chunk_ids.add(cid)
                        iteration_chunks.append(chunk)

            state.retrieved_chunks.extend(iteration_chunks)

            logger.info(f"Read {len(iteration_chunks)} NEW chunks (Total unique chunks read so far: {len(state.retrieved_chunk_ids)})")

            # 2. Combined Single-Call Evaluation (Facts + Conflicts + Sufficiency + Next Queries)
            eval_result = self.evaluator.evaluate_iteration(state, iteration_chunks)

            new_facts = eval_result.get("extracted_facts", [])
            new_conflicts = eval_result.get("identified_conflicts", [])
            has_enough = eval_result.get("has_enough_info", False)
            confidence = eval_result.get("confidence_score", 1)
            rationale = eval_result.get("rationale", "")
            next_queries = eval_result.get("next_suggested_queries", [])
            state.has_enough_info = bool(has_enough)
            state.last_confidence_score = int(confidence or 1)
            state.missing_gaps = eval_result.get("missing_information", []) or []

            logger.info(f"Extracted {len(new_facts)} new facts.")
            logger.info(f"Detected {len(new_conflicts)} source conflicts.")
            logger.info(f"Sufficiency Judgment: has_enough={has_enough} (Confidence: {confidence}/10)")
            logger.info(f"Rationale: {rationale}")

            # Update State Memory
            if new_facts:
                state.working_memory.extend(new_facts)
                stagnant_iterations = 0
            else:
                stagnant_iterations += 1

            if new_conflicts:
                state.identified_conflicts.extend(new_conflicts)

            # Record Trace Log for User Constraint 6
            state.add_trace(f"Iteration_{state.iteration}", {
                "queries_executed": current_queries,
                "new_chunks_count": len(iteration_chunks),
                "new_chunks_ids": [c["chunk_id"] for c in iteration_chunks],
                "extracted_facts": new_facts,
                "identified_conflicts": new_conflicts,
                "has_enough_info": has_enough,
                "confidence_score": confidence,
                "rationale": rationale,
                "missing_information": eval_result.get("missing_information", []),
                "next_suggested_queries": next_queries
            })

            # Check Stopping Criteria
            if has_enough and confidence >= 7:
                state.is_complete = True
                state.stop_reason = f"Sufficient information gathered at iteration {state.iteration} (Confidence: {confidence}/10)."
                logger.info(f"✅ Stop Decision: {state.stop_reason}")
                break

            if state.iteration >= state.max_iterations:
                state.is_complete = True
                state.stop_reason = f"Reached maximum allowed iterations ({state.max_iterations})."
                logger.info(f"🛑 Stop Decision: {state.stop_reason}")
                break

            if stagnant_iterations >= 2:
                state.is_complete = True
                state.stop_reason = "Search stagnated (0 new facts retrieved over 2 consecutive iterations)."
                logger.info(f"⚠️ Stop Decision: {state.stop_reason}")
                break

            # Prepare Next Iteration Queries
            if next_queries:
                current_queries = [q for q in next_queries if q not in state.queries_history][:2]
                if not current_queries:
                    current_queries = [f"{question} details"]
            else:
                current_queries = [f"{question} archive facts"]

            state.iteration += 1

        # 3. Generate Final Grounded Answer with Citations & Surfaced Conflicts
        logger.info("\nSynthesizing final answer...")
        # This call is deliberately outside the search loop. Every normal stop
        # path (confidence, max iterations, or stagnation) reaches it.
        try:
            state.final_answer = self.answer_generator.generate_answer(state)
        except Exception as e:
            # Preserve the final-answer guarantee even if a custom generator
            # fails unexpectedly after the loop has stopped.
            logger.exception("Final answer generation failed: %s", e)
            state.final_answer = (
                f"### Answer to: {state.question}\n\n"
                "Based on available information, I could not fully confirm the answer. "
                "The search stopped before conclusive evidence was available."
            )
        
        # 4. Save Full Per-Iteration Trace to Disk (Requirement 6)
        self._save_trace_log(state)

        return state

    def _save_trace_log(self, state: SearchState):
        """
        Save per-question trace log to logs/traces/<question_id>.json
        """
        trace_path = config.TRACES_DIR / f"question_{state.question_id}.json"
        trace_data = {
            "question_id": state.question_id,
            "question": state.question,
            "total_iterations": state.iteration,
            "stop_reason": state.stop_reason,
            "has_enough_info": state.has_enough_info,
            "last_confidence_score": state.last_confidence_score,
            "missing_gaps": state.missing_gaps,
            "unique_chunks_read_count": len(state.retrieved_chunk_ids),
            "retrieved_chunk_ids": list(state.retrieved_chunk_ids),
            "retrieved_chunks": state.retrieved_chunks,
            "working_memory": state.working_memory,
            "identified_conflicts": state.identified_conflicts,
            "final_answer": state.final_answer,
            "iteration_traces": state.trace_logs
        }
        try:
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved full iteration trace log to '{trace_path}'")
        except Exception as e:
            logger.error(f"Failed to save trace log: {e}")
