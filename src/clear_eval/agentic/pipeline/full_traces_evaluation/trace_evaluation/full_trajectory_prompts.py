"""
Full Trajectory Evaluation Prompt Builders & Constants
======================================================

Contains evaluation dimension constants and prompt building functions
for full trajectory evaluation.

Supports two modes:
- Default: nested step-quality + trajectory-level dimensions (14 total)
- Custom: flat user-provided criteria dict
"""

# Step-level quality dimensions (from CLEAR)
STEP_QUALITY_CRITERIA = {
    "Correctness": (
        "The responses and actions produce accurate, logically sound results "
        "for the given task or query."
    ),
    "Completeness": (
        "The responses fully address the user's request. If a response appears "
        "incomplete but is followed by a tool call or action, this is acceptable."
    ),
    "Clarity": (
        "Explanations, reasoning, and any generated code or actions are easy "
        "to read, well-structured, and unambiguous."
    ),
    "Relevance": (
        "Responses stay focused on the task at hand without unnecessary or "
        "off-topic content."
    ),
    "Efficiency": (
        "The solution or action plan is optimized for performance, avoiding "
        "unnecessary complexity or redundant steps."
    ),
    "Robustness": (
        "The solution handles edge cases, potential errors, and unexpected "
        "inputs gracefully."
    ),
    "Best_Practices": (
        "The solution follows accepted conventions, style guidelines, and "
        "maintainable coding / reasoning standards."
    ),
    "Actionability": (
        "Responses provide directly usable steps, code, or API calls without "
        "requiring significant rework."
    ),
    "Transparency": (
        "Reasoning, assumptions, decisions, and intermediate steps are clearly "
        "explained and justified."
    ),
}

# Trajectory-level (holistic) dimensions
TRAJECTORY_CRITERIA = {
    "Objective_Understanding": (
        "How well the agent understood the user's high-level goal from the "
        "start and maintained alignment throughout the trajectory."
    ),
    "Information_Completeness": (
        "Whether the agent gathered all necessary information (via tools, "
        "queries, observations) before acting, and did not leave critical "
        "gaps in its knowledge."
    ),
    "Execution_Quality": (
        "The overall quality of the agent's execution plan — were the right "
        "tools chosen, called in the right order, with correct parameters, "
        "and did the agent recover from errors effectively?"
    ),
    "User_Experience": (
        "How well the trajectory would serve the end-user: clear "
        "communication, appropriate level of detail, no confusing detours, "
        "and timely progress updates."
    ),
    "Final_Deliverable": (
        "The quality and correctness of the agent's final output or answer "
        "relative to the original objective."
    ),
}

# All dimensions combined
ALL_CRITERIA = {**STEP_QUALITY_CRITERIA, **TRAJECTORY_CRITERIA}

# Scoring scale anchors
SCORING_SCALE = """\
Scoring uses a continuous 0 – 1 scale with the following anchors:
  0.00 = completely failed / absent
  0.25 = poor quality, major issues
  0.50 = acceptable but with notable gaps
  0.75 = good quality, minor issues only
  1.00 = excellent, no meaningful issues"""


def build_default_prompt(trajectory_text: str) -> str:
    """
    Build the default full trajectory evaluation prompt with nested
    step-quality (9) and trajectory-level (5) dimensions.

    Args:
        trajectory_text: Formatted and capped trajectory text

    Returns:
        Formatted prompt string
    """
    step_block = "\n".join(
        f"  - **{name}**: {desc}"
        for name, desc in STEP_QUALITY_CRITERIA.items()
    )
    traj_block = "\n".join(
        f"  - **{name}**: {desc}"
        for name, desc in TRAJECTORY_CRITERIA.items()
    )

    step_dims_json = ",\n".join(
        f'    "{dim}": {{"score": "<0.0-1.0>", "justification": "<text>"}}'
        for dim in STEP_QUALITY_CRITERIA
    )
    traj_dims_json = ",\n".join(
        f'    "{dim}": {{"score": "<0.0-1.0>", "justification": "<text>"}}'
        for dim in TRAJECTORY_CRITERIA
    )

    num_dims = len(STEP_QUALITY_CRITERIA) + len(TRAJECTORY_CRITERIA)

    return f"""\
## Trajectory Evaluation Task (CLEAR Framework — Dimension Scoring)

### Scoring Scale

{SCORING_SCALE}

### Step-Level Quality Dimensions (score each 0.0 – 1.0)

{step_block}

### Trajectory-Level Holistic Dimensions (score each 0.0 – 1.0)

{traj_block}

### Full Agent Trajectory

```
{trajectory_text}
```

### Instructions

1. Carefully read the full trajectory above.
2. Score each of the {num_dims} dimensions on a 0.0–1.0 scale with a brief justification.
3. Write your **detailed_feedback** first (4-8 sentences) — this is your \
chain-of-thought reasoning about strengths, weaknesses, and suggestions.
4. Then decide the **overall_score** (0.0–1.0) for the entire trajectory.

### Required Output (valid JSON only, no extra text)

```json
{{
  "step_quality_dimensions": {{
{step_dims_json}
  }},
  "trajectory_dimensions": {{
{traj_dims_json}
  }},
  "detailed_feedback": "<4-8 sentence paragraph>",
  "overall_score": "<0.0-1.0>"
}}
```
"""


def build_custom_criteria_prompt(trajectory_text: str, criteria: dict) -> str:
    """
    Build a full trajectory evaluation prompt with custom flat criteria.

    Args:
        trajectory_text: Formatted and capped trajectory text
        criteria: Flat dict of {dimension_name: description}

    Returns:
        Formatted prompt string
    """
    criteria_block = "\n".join(
        f"  - **{name}**: {desc}"
        for name, desc in criteria.items()
    )

    dims_json = ",\n".join(
        f'    "{dim}": {{"score": "<0.0-1.0>", "justification": "<text>"}}'
        for dim in criteria
    )

    num_dims = len(criteria)

    return f"""\
## Trajectory Evaluation Task (CLEAR Framework — Dimension Scoring)

### Scoring Scale

{SCORING_SCALE}

### Evaluation Dimensions (score each 0.0 – 1.0)

{criteria_block}

### Full Agent Trajectory

```
{trajectory_text}
```

### Instructions

1. Carefully read the full trajectory above.
2. Score each of the {num_dims} dimensions on a 0.0–1.0 scale with a brief justification.
3. Write your **detailed_feedback** first (4-8 sentences) — this is your \
chain-of-thought reasoning about strengths, weaknesses, and suggestions.
4. Then decide the **overall_score** (0.0–1.0) for the entire trajectory.

### Required Output (valid JSON only, no extra text)

```json
{{
  "dimensions": {{
{dims_json}
  }},
  "detailed_feedback": "<4-8 sentence paragraph>",
  "overall_score": "<0.0-1.0>"
}}
```
"""
