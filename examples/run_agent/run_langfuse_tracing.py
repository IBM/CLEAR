"""
run_langfuse_tracing.py — Run the research agent with Langfuse tracing (SDK v3+)
==================================================================================

Uses the Langfuse LangChain CallbackHandler to automatically capture:
  - Every LLM call (model, prompt, completion, tokens, latency)
  - LangGraph node names (planner, researcher, analyst, writer)
  - Tool calls and results
  - One Langfuse trace per graph.invoke()
"""
import os
import json
import uuid
import time
import argparse

from dotenv import load_dotenv
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from agent_core import build_graph, get_init_state, SAMPLE_INPUTS

load_dotenv(override=True)

def run(inputs: list[str], tag: str, model: str, session_id: str):
    langfuse = get_client()
    print(langfuse.auth_check())
    graph = build_graph()

    for i, question in enumerate(inputs):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(inputs)}] {question}")
        print("=" * 60)

        # start_as_current_observation creates a root span → defines the trace.
        # propagate_attributes pushes session/user/tags onto every child span.
        with langfuse.start_as_current_observation(
            as_type="span",
            name=f"research-agent-q{i + 1:03d}",
            input={"question": question},
        ) as root_span:
            with propagate_attributes(
                session_id=session_id,
                user_id="research-agent-runner",
                tags=["multi-node-research", tag],
                metadata={
                    "agent":       "multi-node-research",
                    "input_index": str(i),
                    "model":       model,
                    "run_group":   tag,
                },
            ):
                # Fresh handler per invocation — inherits the current trace context
                langfuse_handler = LangfuseCallbackHandler()

                try:
                    init_state = get_init_state(question, model)
                    result = graph.invoke(
                        init_state,
                        config={"callbacks": [langfuse_handler]},
                    )
                    root_span.update(output={"answer": result["final_answer"]})
                    print(f"Answer: {result['final_answer'][:300]}")
                except Exception as e:
                    root_span.update(output={"error": str(e)})
                    print(f"ERROR: {e}")

        # Flush after each trace to ensure delivery
        langfuse.flush()

    # Final flush to ensure all traces are sent
    print(f"\nFlushing all traces to Langfuse...")
    langfuse.flush()
    print(f"All traces sent to Langfuse (session: {session_id})")


def export_traces(output_dir: str, tag: str):
    """Fetch traces from Langfuse by run_group tag and save as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    langfuse = get_client()

    # Wait for async ingestion to settle
    print("\nWaiting for Langfuse ingestion to settle...")
    time.sleep(5)

    # v3+ API: list traces filtered by tag
    response = langfuse.api.trace.list(tags=[tag], limit=100)
    traces = response.data

    if not traces:
        print(f"No traces found with tag '{tag}'")
        return

    print(f"Exporting {len(traces)} traces (tag='{tag}') to {output_dir}/")

    for t in traces:
        try:
            # Fetch full trace with all observations
            full_trace = langfuse.api.trace.get(t.id)

            # Convert to dict — handles both pydantic v1 (.dict()) and v2 (.model_dump())
            if hasattr(full_trace, "model_dump"):
                trace_dict = full_trace.model_dump()
            elif hasattr(full_trace, "dict"):
                trace_dict = full_trace.dict()
            else:
                trace_dict = full_trace.__dict__

            filepath = os.path.join(output_dir, f"{t.id}.json")
            with open(filepath, "w") as f:
                json.dump(trace_dict, f, indent=2, default=str)
            print(f"  ✓ {t.id}.json")
        except Exception as e:
            print(f"  ✗ {t.id}.json - Error: {e}")

    print(f"Done! {len(traces)} traces saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run research agent — Langfuse tracing")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--tag", default=f"run_{uuid.uuid4().hex[:8]}")
    parser.add_argument("--session-id", default=None,
                        help="Langfuse session ID (groups traces; auto-generated if omitted)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to export trace JSON files (optional)")
    args = parser.parse_args()
    session_id = args.session_id or f"session_{uuid.uuid4().hex[:8]}"
    print(f"Run tag:    {args.tag}")
    print(f"Session ID: {session_id}")

    inputs = SAMPLE_INPUTS[: args.limit] if args.limit else SAMPLE_INPUTS
    run(inputs, args.tag, args.model, session_id)

    if args.output_dir:
        export_traces(args.output_dir, args.tag)


if __name__ == "__main__":
    main()