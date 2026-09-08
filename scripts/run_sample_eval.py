import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.agent.search_loop import HumanLikeSearchAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SampleEval")

def run_evaluation(sample_file: Path, limit: int = 5):
    if not sample_file.exists():
        logger.error(f"Sample questions file not found at '{sample_file}'.")
        return

    with open(sample_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if limit > 0:
        questions = questions[:limit]

    logger.info(f"=== Starting Benchmark Evaluation on {len(questions)} Sample Questions ===")
    
    agent = HumanLikeSearchAgent(max_iterations=5)
    results_summary = []
    start_time = time.time()

    for idx, item in enumerate(questions, start=1):
        qid = item.get("qid", f"sample_{idx}")
        track = item.get("track", "Unknown Track")
        qtext = item.get("question", "")
        trace_path = config.TRACES_DIR / f"question_{qid}.json"

        if trace_path.exists():
            logger.info(f"Skipping QID '{qid}': existing trace found at '{trace_path}'.")
            try:
                with open(trace_path, "r", encoding="utf-8") as trace_file:
                    existing_trace = json.load(trace_file)
                results_summary.append({
                    "qid": qid,
                    "track": track,
                    "question": qtext,
                    "iterations": existing_trace.get("total_iterations", 0),
                    "chunks_read": existing_trace.get("unique_chunks_read_count", 0),
                    "facts_found": len(existing_trace.get("working_memory", [])),
                    "conflicts_surfaced": len(existing_trace.get("identified_conflicts", [])),
                    "stop_reason": existing_trace.get("stop_reason", ""),
                    "final_answer": existing_trace.get("final_answer", ""),
                    "trace_file": str(trace_path),
                    "status": "skipped_existing_trace",
                })
            except Exception as e:
                logger.error(f"Could not load existing trace for QID '{qid}': {e}")
            continue

        logger.info(f"\n==================================================")
        logger.info(f"[{idx}/{len(questions)}] Processing QID '{qid}' ({track})")
        logger.info(f"Question: '{qtext}'")
        logger.info(f"==================================================")

        try:
            state = agent.run(qtext, question_id=qid)
            summary_entry = {
                "qid": qid,
                "track": track,
                "question": qtext,
                "iterations": state.iteration,
                "chunks_read": len(state.retrieved_chunk_ids),
                "facts_found": len(state.working_memory),
                "conflicts_surfaced": len(state.identified_conflicts),
                "stop_reason": state.stop_reason,
                "final_answer": state.final_answer,
                "trace_file": f"logs/traces/question_{qid}.json",
                "status": "executed"
            }
            results_summary.append(summary_entry)
        except Exception as e:
            logger.error(f"Error evaluating QID '{qid}': {e}")

    elapsed = time.time() - start_time
    logger.info(f"\n==================================================")
    logger.info(f"=== Benchmark Evaluation Finished in {elapsed:.2f} seconds ===")
    logger.info(f"Total Questions Processed: {len(results_summary)}")
    if results_summary:
        avg_iters = sum(r["iterations"] for r in results_summary) / len(results_summary)
        avg_chunks = sum(r["chunks_read"] for r in results_summary) / len(results_summary)
        total_conflicts = sum(r["conflicts_surfaced"] for r in results_summary)
        logger.info(f"  - Average Iterations/Question: {avg_iters:.2f}")
        logger.info(f"  - Average Chunks Read/Question: {avg_chunks:.2f}")
        logger.info(f"  - Total Conflicts Surfaced   : {total_conflicts}")
    logger.info(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation benchmark over sample questions")
    parser.add_argument("--sample_file", type=str, default=str(config.CORPUS_DIR / "sample_questions.json"), help="Path to sample questions JSON")
    parser.add_argument("--limit", type=int, default=3, help="Limit number of questions to evaluate (0 for all)")
    args = parser.parse_args()

    run_evaluation(Path(args.sample_file), limit=args.limit)
