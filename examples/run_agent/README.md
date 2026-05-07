# Agent Tracing Example — Generate Traces for CLEAR

Run a LangGraph agent with **MLflow** or **Langfuse** tracing and export traces as JSON files compatible with CLEAR evaluation.

No external tool APIs needed — only an OpenAI API key for the LLM.

---

## Files

| File | Purpose |
|------|---------|
| `agent_core.py` | 4-node LangGraph agent with built-in tools |
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
python run_no_tracing.py --limit 3
```

---

## 2. Run with MLflow Tracing (no external service)

```bash
python run_mlflow_tracing.py --limit 20 --output-dir my_traces_mlflow
```

Traces are exported as JSON to `my_traces_mlflow/`. Optionally view in MLflow UI:
```bash
mlflow ui --port 5001   # http://localhost:5001
```

---

## 3. Run with Langfuse Tracing
This assumes a valid langfuse account. 

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."  # pragma: allowlist secret
export LANGFUSE_HOST="https://cloud.langfuse.com"

python run_langfuse_tracing.py --limit 20 --output-dir my_traces_langfuse --model gpt-4o
```

---

## Next: Evaluate with CLEAR

Once you have exported traces, feed them into the CLEAR agentic pipeline. See the [Agentic Evaluation README](../../src/clear_eval/agentic/README.md) for instructions.

Pre-exported sample traces (from this agent) and pre-computed CLEAR results are available at `src/clear_eval/sample_data/agentic/` for quick testing.

---

## The Agent

A 4-node LangGraph pipeline: **planner → researcher → analyst → writer**

Each node is an LLM with access to 3 deterministic, offline tools:

| Tool | Purpose | Example |
|------|---------|---------|
| `knowledge_lookup` | Query a built-in KB (12 topics) | `knowledge_lookup("mars")` |
| `calculator` | Evaluate Python math expressions | `calculator("13960000 / 2194")` |
| `unit_converter` | Convert between common units | `unit_converter(40, "celsius", "fahrenheit")` |

20 sample inputs exercise: pure lookups, calculations, conversions, multi-step reasoning, and comparisons.

---