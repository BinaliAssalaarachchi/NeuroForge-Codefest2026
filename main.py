import sys
import argparse
from pathlib import Path

# Ensure UTF-8 stdout encoding on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import config
from src.agent.search_loop import HumanLikeSearchAgent

def ask_question(question_str: str, qid: str = "cli_user"):
    agent = HumanLikeSearchAgent(max_iterations=5)
    state = agent.run(question_str, question_id=qid)

    print("\n" + "=" * 80)
    print(f"🎯 FINAL ANSWER FOR: '{state.question}'")
    print("=" * 80)
    print(state.final_answer)
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY:")
    print(f"  - Total Iterations        : {state.iteration}")
    print(f"  - Unique Chunks Read      : {len(state.retrieved_chunk_ids)}")
    print(f"  - Facts Extracted         : {len(state.working_memory)}")
    print(f"  - Discrepancies Surfaced  : {len(state.identified_conflicts)}")
    print(f"  - Stop Rationale          : {state.stop_reason}")
    print(f"  - Full Trace Log          : logs/traces/question_{state.question_id}.json")
    print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="NeuroForge Ashen Era Assistant (SLIIT Codefest 2026)")
    parser.add_argument("--question", type=str, help="Question to ask the assistant")
    parser.add_argument("--qid", type=str, default="cli_q1", help="Optional Question ID for trace log naming")
    args = parser.parse_args()

    if args.question:
        ask_question(args.question, qid=args.qid)
    else:
        print("Welcome to NeuroForge Ashen Era Archive Assistant!")
        print("Type your question below (or 'exit' to quit):\n")
        while True:
            try:
                user_q = input("Question> ").strip()
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Goodbye!")
                    break
                if not user_q:
                    continue
                ask_question(user_q)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break

if __name__ == "__main__":
    main()
