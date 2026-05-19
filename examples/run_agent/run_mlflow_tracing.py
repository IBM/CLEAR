"""
run_mlflow_tracing.py — Run the research agent with MLflow tracing
===================================================================

Uses mlflow.langchain.autolog() to automatically capture:
  - Every LLM call (model, tokens, latency)
  - LangGraph node transitions
  - Tool calls and results
  - One MLflow trace per graph.invoke()
"""

import os
import json
import uuid
import argparse

import mlflow
from mlflow.client import MlflowClient
from agent_core import build_graph, SAMPLE_INPUTS, get_init_state


def run(inputs: list[str], experiment_name: str, tag: str, model: str):
    # ---- Enable MLflow LangChain auto-tracing ----
    mlflow.set_experiment(experiment_name)
    mlflow.langchain.autolog()

    graph = build_graph()

    for i, question in enumerate(inputs):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(inputs)}] {question}")
        print("=" * 60)

        # start_span creates a Trace (if none exists) + a root span.
        # autolog spans from graph.invoke() nest inside it.
        with mlflow.start_span(
            name=f"research_agent_q{i + 1:03d}",
            span_type="AGENT",
        ) as root_span:
            root_span.set_inputs({"question": question})

            # Now a trace IS active, so update_current_trace works
            mlflow.update_current_trace(tags={
                "agent":       "multi-node-research",
                "input_index": str(i),
                "model":       model,
                "run_group":   tag,
                "intent":      question,
            })

            try:
                init_state = get_init_state(question, model)
                result = graph.invoke(init_state)
                root_span.set_outputs({"answer": result["final_answer"]})
                root_span.set_status("OK")
                print(f"Answer: {result['final_answer'][:300]}")
            except Exception as e:
                root_span.set_status("ERROR")
                print(f"ERROR: {e}")


def export_traces(output_dir: str, experiment_name: str, tag: str):
    """Export only traces from this run (matched by run_group tag)."""
    client = MlflowClient()
    os.makedirs(output_dir, exist_ok=True)

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found")
        return

    traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        filter_string=f"tag.run_group = '{tag}'",
        return_type="list",
        flush=True,
    )
    print(f"\nExporting {len(traces)} traces (tag='{tag}') to {output_dir}/")
    for t in traces:
        trace_id = t.info.request_id
        filepath = os.path.join(output_dir, f"{trace_id}.json")
        with open(filepath, "w") as f:
            json.dump(t.to_dict(), f, indent=2, default=str)
        print(f"  {trace_id}.json")
    print(f"Done! {len(traces)} traces saved.")


def main():
    parser = argparse.ArgumentParser(description="Run research agent — MLflow tracing")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--experiment-name", default="research-agent-mlflow")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5001",
                        help="MLflow tracking URI (default: 'http://127.0.0.1:5001')")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to export trace JSON files (optional)")
    parser.add_argument("--tag", default=f"run_{uuid.uuid4().hex[:8]}")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    print(f"MLflow tracking URI: {args.tracking_uri}")
    print(f"Run tag: {args.tag}")

    inputs = SAMPLE_INPUTS[: args.limit] if args.limit else SAMPLE_INPUTS
    run(inputs, args.experiment_name, args.tag, args.model)

    if args.output_dir:
        export_traces(args.output_dir, args.experiment_name, args.tag)


if __name__ == "__main__":
    main()