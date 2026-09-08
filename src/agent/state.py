from dataclasses import dataclass, field
from typing import List, Dict, Any, Set, Optional

@dataclass
class SearchState:
    question_id: str
    question: str
    iteration: int = 1
    max_iterations: int = 5
    queries_history: List[str] = field(default_factory=list)
    retrieved_chunk_ids: Set[str] = field(default_factory=set)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    identified_conflicts: List[Dict[str, Any]] = field(default_factory=list)
    missing_gaps: List[str] = field(default_factory=list)
    has_enough_info: bool = False
    last_confidence_score: int = 1
    is_complete: bool = False
    stop_reason: str = ""
    final_answer: str = ""
    trace_logs: List[Dict[str, Any]] = field(default_factory=list)

    def add_trace(self, step_type: str, data: Dict[str, Any]):
        self.trace_logs.append({
            "iteration": self.iteration,
            "step_type": step_type,
            "data": data
        })
