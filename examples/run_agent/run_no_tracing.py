"""
run_no_tracing.py — Run the research agent with NO tracing
===========================================================
"""

import argparse
from agent_core import build_graph, get_init_state, SAMPLE_INPUTS


def run(inputs: list[str], model: str) -> None:
    graph = build_graph()

    for i, question in enumerate(inputs):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(inputs)}] {question}")
        print("=" * 60)
        try:
            result = graph.invoke(get_init_state(question, model))
            print(f"Answer: {result['final_answer'][:300]}")
        except Exception as e:
            print(f"ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run research agent — no tracing")
    parser.add_argument("--limit", type=int, default=3, help="Number of inputs to run")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    inputs = SAMPLE_INPUTS[: args.limit] if args.limit else SAMPLE_INPUTS
    run(inputs, args.model)


if __name__ == "__main__":
    main()
