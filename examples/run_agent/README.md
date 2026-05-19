# Agent Tracing Example — Generate Traces for CLEAR

Run a LangGraph agent with **MLflow** or **Langfuse** tracing and export traces as JSON files compatible with CLEAR evaluation.

No external tool APIs needed — only an OpenAI API key for the LLM.

---

## Files

| File | Purpose |
|------|---------|
| `agent_core.py` | 6-node LangGraph agent with built-in tools, conditional routing, and review loop |
| `run_mlflow_tracing.py` | Run agent with MLflow tracing → export JSON traces |
| `run_langfuse_tracing.py` | Run agent with Langfuse tracing → export JSON traces |
| `run_no_tracing.py` | Run agent without tracing (sanity check) |
| `requirements.txt` | Python dependencies |

---

## Install

```bash
cd examples/run_agent
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

---

## 1. Verify the Agent Works

```bash
python run_no_tracing.py --limit 5
```

---

## 2. Run with MLflow Tracing

Start the MLflow server first:

```bash
# Terminal 1: Start MLflow server
mlflow server --host 127.0.0.1 --port 5001
```

Then run the agent (uses the server at http://127.0.0.1:5001):

```bash
# Terminal 2: Run the agent
python run_mlflow_tracing.py --limit 20 --output-dir my_traces_mlflow
```

Traces are stored on the MLflow server and exported as JSON to `my_traces_mlflow/` View them in the MLflow UI at http://localhost:5001

**Note**: You can use any MLflow tracking URI by passing `--tracking-uri <your-uri>` (e.g., a remote MLflow server, SQLite database `sqlite:///mlflow.db`, or other backend).

---

## 3. Run with Langfuse Tracing

This assumes a valid Langfuse account.

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."  # pragma: allowlist secret
export LANGFUSE_HOST="https://cloud.langfuse.com"

python run_langfuse_tracing.py --limit 5 --output-dir my_traces_langfuse --model gpt-4o-mini
```

---

## Next: Evaluate with CLEAR

Once you have exported traces, feed them into the CLEAR agentic pipeline. See the [Agentic Evaluation README](../../src/clear_eval/agentic/README.md) for instructions.

Pre-exported sample traces (from this agent) and pre-computed CLEAR results are available at `src/clear_eval/sample_data/agentic/` for quick testing.

---

## The Agent

A 6-node LangGraph agent with conditional routing and review loop. See `agent_core.py` for implementation details.

---