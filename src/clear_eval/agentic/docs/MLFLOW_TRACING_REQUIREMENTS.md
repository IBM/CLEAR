# MLflow Tracing Requirements for CLEAR

## Requirements

1. **LLM calls captured as spans** — each LLM API call must appear as a span with `span_type` in `{CHAT_MODEL, MODEL, GENERATION}`.

2. **Named parent spans per graph node** — each LLM call span should have a parent span named after the graph node that made it (e.g. `planner`, `analyst`). *Optional for step analysis — without it, per-component grouping is lost but individual scores are unaffected.*

3. **One trace per agent input** — each trace should contain one complete agent invocation. *Optional for step analysis — without it, trajectory views and path analysis are not meaningful but individual scores are unaffected.*

**Tool schemas:** Use `bind_tools()` so tool definitions are captured in the trace. Describing tools in the system prompt is not visible to autologging.

---

## With `mlflow.langchain.autolog()`

All three requirements satisfied automatically:

```python
mlflow.langchain.autolog()
result = graph.invoke(...)
```

**Caveats:** fragile in async code (`ainvoke`, `astream`), version-sensitive to LangChain/LangGraph updates. Sync code works reliably.

---

## With `mlflow.openai.autolog()`

Only requirement 1 is automatic. For requirements 2 and 3:

**One trace per input**: wrap each invocation:

```python
with mlflow.start_span(name="agent_run", span_type="AGENT"):
    result = graph.invoke(...)
```

**Named node spans**: decorate node functions:

```python
@mlflow.trace(name="planner", span_type="CHAIN")
def planner_node(state):
    ...
```

Node span `span_type` must **not** be `CHAT_MODEL`, `MODEL`, or `GENERATION`. 

Note on ReAct agents: Prebuilt ReAct agents (e.g. create_react_agent) have a single LLM-calling node (agent) whose name is picked up automatically. 
Per-component analysis only becomes meaningful with custom StateGraph agents that have multiple LLM-calling nodes.

---

## Optional: intent and traj_score

These are optional metadata used for trajectory-level evaluation.

**Intent** (the user's original question) — set via trace tags or metadata:

```python
mlflow.update_current_trace(tags={"intent": "What is the population of Tokyo?"})
```

CLEAR searches these fields in order: `intent`, `user_query`, `task`, `goal`, `query`, `question`, `request`, `request_preview`, `input`. It checks trace-level `tags`, `metadata`, `request_metadata`, and top-level trace fields. Falls back to the first user message in the root span's input.

**traj_score** (ground-truth quality label, 0–1) — set the same way:

```python
mlflow.update_current_trace(tags={"traj_score": "0.9"})
```

CLEAR searches: `traj_score`, `score`, `quality`, `rating`, `correctness` across the same locations, plus MLflow assessments.

---
