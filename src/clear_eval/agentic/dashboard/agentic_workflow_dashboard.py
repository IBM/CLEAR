"""
Agentic Workflow Dashboard - Visualize agent traces and CLEAR analysis results

Built with NiceGUI for a modern, high-quality UI experience.

This dashboard provides:
1. Unified workflow view showing agent nodes and their relationships
2. Node-specific CLEAR analysis (reusing existing analysis logic)
3. Trace-specific exploration with metadata
4. Path analysis with success/failure patterns
5. Temporal analysis of agent behavior
"""

import ast
import json
import math
import signal
import sys
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from nicegui import events, ui

# ─── NiceGUI 3.0+ Compatibility ──────────────────────────────────────────────
# NiceGUI 3.0 introduced a required 'sanitize' parameter for ui.html().
# Since all HTML in this dashboard is trusted (not user input), we use sanitize=False.
_original_ui_html = ui.html

def _html_safe(*args, **kwargs):
    """Wrapper for ui.html that sets sanitize=False by default for NiceGUI 3.0+ compatibility."""
    if 'sanitize' not in kwargs:
        kwargs['sanitize'] = False
    return _original_ui_html(*args, **kwargs)

ui.html = _html_safe

# ─── Color Palette & Theme Constants ─────────────────────────────────────────
COLORS = {
    "primary": "#6366F1",  # Brighter indigo
    "primary_light": "#A5B4FC",
    "primary_dark": "#4338CA",
    "primary_bg": "#EEF2FF",
    "secondary": "#06B6D4",  # Brighter cyan
    "secondary_bg": "#ECFEFF",
    "success": "#10B981",  # Emerald
    "success_bg": "#D1FAE5",
    "success_dark": "#059669",
    "warning": "#F59E0B",  # Amber
    "warning_bg": "#FEF3C7",
    "warning_dark": "#D97706",
    "danger": "#EF4444",  # Red
    "danger_bg": "#FEE2E2",
    "danger_dark": "#DC2626",
    "neutral": "#64748B",
    "neutral_bg": "#F1F5F9",
    "dark": "#0F172A",  # Darker for better contrast
    "text": "#1E293B",  # Darker text for better readability
    "text_secondary": "#475569",  # Medium contrast
    "text_light": "#64748B",  # Light text
    "border": "#E2E8F0",
    "border_light": "#F1F5F9",
    "card_bg": "#FFFFFF",
    "page_bg": "#F8FAFC",
    # Enhanced chart colors with better contrast and distinction
    "chart": [
        "#6366F1",  # Indigo
        "#06B6D4",  # Cyan
        "#10B981",  # Emerald
        "#F59E0B",  # Amber
        "#EF4444",  # Red
        "#8B5CF6",  # Violet
        "#EC4899",  # Pink
        "#14B8A6",  # Teal
        "#F97316",  # Orange
        "#3B82F6",  # Blue
    ],
    "node_default": "#6366F1",
    "node_selected": "#F59E0B",
    "node_success": "#10B981",
    "node_error": "#EF4444",
    "edge": "#94A3B8",
    "edge_active": "#6366F1",
}

# ─── CLEAR Evaluation Criteria Definitions ──────────────────────────────────
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

# Trace-level (holistic) dimensions
TRAJECTORY_CRITERIA = {
    "Objective_Understanding": (
        "How well the agent understood the user's high-level goal from the "
        "start and maintained alignment throughout the trace."
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
        "How well the trace would serve the end-user: clear "
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

# ─── Plotly Template ─────────────────────────────────────────────────────────
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            color=COLORS["text"],
            size=13,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        title=dict(
            font=dict(size=18, color=COLORS["dark"], weight=600),
            x=0.5,
            xanchor="center",
        ),
        xaxis=dict(
            gridcolor="#F1F5F9",
            gridwidth=1,
            zerolinecolor="#E2E8F0",
            zerolinewidth=2,
            linecolor="#CBD5E1",
            linewidth=1,
            tickfont=dict(size=12, color=COLORS["text_secondary"]),
            title=dict(font=dict(size=13, color=COLORS["text"], weight=600)),
        ),
        yaxis=dict(
            gridcolor="#F1F5F9",
            gridwidth=1,
            zerolinecolor="#E2E8F0",
            zerolinewidth=2,
            linecolor="#CBD5E1",
            linewidth=1,
            tickfont=dict(size=12, color=COLORS["text_secondary"]),
            title=dict(font=dict(size=13, color=COLORS["text"], weight=600)),
        ),
        colorway=COLORS["chart"],
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Inter, sans-serif",
            font_color=COLORS["text"],
            bordercolor="#CBD5E1",
            align="left",
        ),
        margin=dict(l=60, r=40, t=60, b=50),
        legend=dict(
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=12, color=COLORS["text"]),
        ),
    )
)
pio.templates["clear_theme"] = PLOTLY_TEMPLATE
pio.templates.default = "clear_theme"

# Constants from original CLEAR dashboard
EXPECTED_COLS = [
    "question_id", "model_input", "response", "score",
    "evaluation_text", "evaluation_summary", "recurring_issues",
    "recurring_issues_str", "ground_truth", "traj_score",
]
NO_ISSUE = "No Issues"
OTHER = "Other Issues"
MAX_NUM_ISSUES = 30


# ─── Pure Data / Analysis Functions (no UI) ──────────────────────────────────

def extract_issues(text, delimiter=";"):
    """Extract issues list from text representation."""
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if isinstance(text, list):
        text = json.dumps(text)
    issues = _extract_issues_from_str(text, delimiter)
    return [NO_ISSUE] if not issues else issues


def _extract_issues_from_str(text, delimiter=";"):
    if pd.isna(text) or not text or text == "[]":
        return []
    try:
        evaluated = ast.literal_eval(text)
        if isinstance(evaluated, list):
            return [str(item).strip() for item in evaluated if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [issue.strip() for issue in str(text).split(delimiter) if issue.strip()]


def get_input_columns(metadata):
    return metadata.get("input_columns", ["question", "documents"])


def load_clear_data_from_bytes(file_bytes, file_name, trajectory_df: pd.DataFrame = None):
    """
    Load CLEAR formatted data from a zip file (bytes).
    
    NEW: If trajectory_df is provided and CLEAR results are missing model_input/response,
    join them from trace data using (task_id, step_in_trace_general).
    """
    with zipfile.ZipFile(BytesIO(file_bytes)) as zf:
        names = set(zf.namelist())
        metadata = None
        csv_file_name = None
        parquet_file_name = None

        for name in names:
            if name.endswith(".csv") and name.startswith("results"):
                csv_file_name = name
            if name.endswith(".parquet") and name.startswith("results"):
                parquet_file_name = name
            if name.endswith(".json") and name.startswith("metadata"):
                metadata = json.load(zf.open(name))

        if csv_file_name is None and parquet_file_name is None:
            return pd.DataFrame(), {}
        if metadata is None:
            metadata = {}

        input_columns = get_input_columns(metadata)
        expected_cols = list(dict.fromkeys(EXPECTED_COLS + input_columns))

        df = pd.DataFrame()
        if csv_file_name is not None:
            with zf.open(csv_file_name) as f:
                df = pd.read_csv(f, usecols=lambda c: c in expected_cols)
        elif parquet_file_name is not None:
            import pyarrow.parquet as pq
            with zf.open(parquet_file_name) as f:
                actual_cols = pq.ParquetFile(f).schema.names
                selected = [c for c in expected_cols if c in actual_cols]
                df = pd.read_parquet(f, engine="pyarrow", columns=selected)

        # NEW: Join with trace data if model_input/response are missing
        if trajectory_df is not None and not trajectory_df.empty:
            missing_cols = []
            if 'model_input' not in df.columns and 'model_input' in trajectory_df.columns:
                missing_cols.append('model_input')
            if 'response' not in df.columns and 'response' in trajectory_df.columns:
                missing_cols.append('response')
            
            if missing_cols:
                # Determine the step column name
                step_col = 'step_in_trace_general' if 'step_in_trace_general' in df.columns else 'step_in_trace'
                traj_step_col = 'step_in_trace_general' if 'step_in_trace_general' in trajectory_df.columns else 'step_in_trace'
                
                if 'task_id' in df.columns and step_col in df.columns:
                    # Select only the columns we need from trace data
                    join_cols = ['task_id', traj_step_col] + missing_cols
                    traj_subset = trajectory_df[join_cols].copy()
                    
                    # Join the data
                    df = df.merge(
                        traj_subset,
                        left_on=['task_id', step_col],
                        right_on=['task_id', traj_step_col],
                        how='left',
                        suffixes=('', '_traj')
                    )
                    
                    # Clean up duplicate columns
                    if traj_step_col != step_col and traj_step_col in df.columns:
                        df = df.drop(columns=[traj_step_col])
                    
                   # print(f"  ✓ Joined {len(missing_cols)} columns from trace data: {', '.join(missing_cols)}")

        if "score" in df.columns:
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df.dropna(subset=["score"], inplace=True)
        if "recurring_issues_str" in df.columns:
            df["discovered_issues"] = df.apply(
                lambda r: ",\n".join(extract_issues(r["recurring_issues_str"])), axis=1
            )
        if "model_input" in df.columns:
            df["model_input_preview"] = df["model_input"].apply(
                lambda x: x[:300] if isinstance(x, str) else x
            )
        if "question_id" in df.columns:
            df.set_index("question_id", inplace=True)

        return df, metadata


def load_data_from_zip(file_bytes, progress_callback=None) -> Tuple[pd.DataFrame, Dict[str, Dict], Dict]:
    """
    Load trace data and agent results from optimized UI results zip.
    
    NEW FORMAT (v4.0):
    - Trace data: Parquet files with model_input & response
    - CLEAR results: CSV/Parquet without model_input & response
    - Dashboard joins them using (task_id, step_in_trace_general)
    - Full trace results: per_traj_results with evaluation scores
    
    Args:
        file_bytes: The zip file bytes
        progress_callback: Optional callback function(current, total, message) for progress updates
    """
    trajectory_df = pd.DataFrame()
    agent_results = {}
    metadata = {}
    traj_eval_results = {}

    with zipfile.ZipFile(BytesIO(file_bytes), "r") as zf:
        # Get all file names and find the trajectory_data.zip (handle nested directories)
        all_files = zf.namelist()
        
        # Find metadata.json (could be at root or in subdirectory)
        metadata_files = [f for f in all_files if f.endswith("metadata.json") and not f.startswith("__MACOSX")]
        if metadata_files:
            with zf.open(metadata_files[0]) as f:
                metadata = json.load(f)
                format_version = metadata.get("format_version", "3.0")
                print(f"📋 Loading data (format version: {format_version})")

        all_traj_data = []

        # Load trace data from trajectory_data.zip (handle nested directories)
        traj_zip_files = [f for f in all_files if f.endswith("trajectory_data.zip") and not f.startswith("__MACOSX")]
        if traj_zip_files:
            traj_zip_path = traj_zip_files[0]
            print(f"📦 Found trace data at: {traj_zip_path}")
            with zf.open(traj_zip_path) as traj_zip_file:
                # Read trace zip into memory once
                traj_zip_bytes = traj_zip_file.read()
                
            with zipfile.ZipFile(BytesIO(traj_zip_bytes), "r") as traj_zf:
                # Load Parquet files (new format v4.0) or CSV files (old format v3.0)
                parquet_files = [n for n in traj_zf.namelist() if n.endswith(".parquet")]
                csv_files = [n for n in traj_zf.namelist() if n.endswith(".csv") and n != "inputs_lookup.csv"]
                
                if parquet_files:
                    print(f"📊 Loading {len(parquet_files)} Parquet trace files...")
                    total_files = len(parquet_files)
                    
                    # Process files in batches to avoid memory issues
                    batch_size = 50
                    for i in range(0, total_files, batch_size):
                        batch_files = parquet_files[i:i+batch_size]
                        batch_data = []
                        
                        for idx, parquet_file in enumerate(batch_files):
                            if progress_callback:
                                progress_callback(i + idx + 1, total_files, f"Loading trace files...")
                            
                            with traj_zf.open(parquet_file) as f:
                                batch_data.append(pd.read_parquet(f))
                        
                        # Concatenate batch and append to main list
                        if batch_data:
                            batch_df = pd.concat(batch_data, ignore_index=True)
                            all_traj_data.append(batch_df)
                            del batch_data  # Free memory
                            
                elif csv_files:
                    print(f"📊 Loading {len(csv_files)} CSV trace files (legacy format)...")
                    total_files = len(csv_files)
                    
                    for idx, csv_file in enumerate(csv_files):
                        if progress_callback:
                            progress_callback(idx + 1, total_files, f"Loading trace files...")
                        
                        with traj_zf.open(csv_file) as f:
                            all_traj_data.append(pd.read_csv(f))
                
                # Handle old format with inputs_lookup.csv (v3.0)
                if "inputs_lookup.csv" in traj_zf.namelist():
                    print("  ℹ️  Found inputs_lookup.csv (legacy format v3.0)")
                    with traj_zf.open("inputs_lookup.csv") as f:
                        inputs_lookup_df = pd.read_csv(f)
                        print(f"  📊 Loaded inputs lookup with {len(inputs_lookup_df)} unique inputs")
                        
                        # Join inputs back to trace data
                        if all_traj_data:
                            if progress_callback:
                                progress_callback(1, 1, "Merging trace data...")
                            
                            trajectory_df = pd.concat(all_traj_data, ignore_index=True)
                            step_col = 'step_in_trace_general' if 'step_in_trace_general' in trajectory_df.columns else 'step_in_trace'
                            
                            if 'task_id' in trajectory_df.columns and step_col in trajectory_df.columns:
                                trajectory_df = trajectory_df.merge(
                                    inputs_lookup_df,
                                    left_on=['task_id', step_col],
                                    right_on=['task_id', 'step_in_trace'],
                                    how='left',
                                    suffixes=('', '_lookup')
                                )
                                if 'step_in_trace' in trajectory_df.columns and step_col != 'step_in_trace':
                                    trajectory_df = trajectory_df.drop(columns=['step_in_trace'])
                            #    print(f"  ✓ Joined {len(trajectory_df)} trace rows with inputs")
                    all_traj_data = []  # Clear since we already concatenated
        else:
            raise ValueError("trajectory_data.zip not found in uploaded file.")

        # Concatenate trace data if not already done
        if all_traj_data:
            if progress_callback:
                progress_callback(1, 1, "Finalizing trace data...")
            
            trajectory_df = pd.concat(all_traj_data, ignore_index=True)
            print(f"✓ Loaded {len(trajectory_df)} trace rows")

        # Add "Name" column if it doesn't exist but "agent_name" does
        if "agent_name" in trajectory_df.columns and "Name" not in trajectory_df.columns:
            trajectory_df["Name"] = trajectory_df["agent_name"]
            print("  ℹ️  Added 'Name' column from 'agent_name'")

        # Store trace data for joining with CLEAR results
        metadata["trajectory_df"] = trajectory_df

        # Support both "agent_results/" and nested agent result directories
        # Filter out __MACOSX files
        agent_files = [
            n for n in all_files
            if n.endswith(".zip") and "agent_results/" in n and not n.startswith("__MACOSX")
        ]
        
        if progress_callback:
            progress_callback(1, 1, "Loading agent results...")
        
        for agent_file in agent_files:
            agent_name = Path(agent_file).stem
            with zf.open(agent_file) as f:
                agent_results[agent_name] = {
                    "zip_bytes": f.read(),
                    "zip_name": Path(agent_file).name,
                }
        
        print(f"✓ Loaded {len(agent_results)} agent result files")

        # Load clear_data CSV files (contain traj_score column)
        clear_data_files = [
            n for n in all_files
            if "clear_data/" in n and n.endswith(".csv") and not n.startswith("__MACOSX")
        ]
        
        task_id_to_traj_score = {}
        if clear_data_files:
            if progress_callback:
                progress_callback(1, 1, "Loading trace scores...")
            
            print(f"📊 Loading {len(clear_data_files)} clear_data CSV files (with traj_score)...")
            for clear_data_file in clear_data_files:
                try:
                    with zf.open(clear_data_file) as f:
                        df = pd.read_csv(f)
                        print(f"  📄 {clear_data_file}: {len(df)} rows, columns: {list(df.columns)[:5]}...")
                        if "task_id" in df.columns and "traj_score" in df.columns:
                            # Extract task_id -> traj_score mapping
                            score_mapping = df[["task_id", "traj_score"]].drop_duplicates(subset=["task_id"])
                            scores_added = 0
                            for _, row in score_mapping.iterrows():
                                task_id = row["task_id"]
                                traj_score = row["traj_score"]
                                if pd.notna(task_id) and pd.notna(traj_score):
                                    task_id_to_traj_score[task_id] = float(traj_score)
                                    scores_added += 1
                            print(f"    ✓ Added {scores_added} task scores from this file")
                        else:
                            print(f"    ⚠️  Missing required columns (task_id or traj_score)")
                except Exception as e:
                    print(f"  ⚠️  Error loading {clear_data_file}: {e}")
                    import traceback
                    traceback.print_exc()
            
            if task_id_to_traj_score:
                print(f"  ✓ Extracted traj_score for {len(task_id_to_traj_score)} tasks from clear_data CSVs")
                metadata["task_id_to_traj_score"] = task_id_to_traj_score
            else:
                print(f"  ⚠️  No traj_scores extracted from clear_data CSVs")

        # Load full trace evaluation results if available
        # Filter out __MACOSX files
        full_traj_files = [
            n for n in all_files
            if "full_traj_results/per_traj_results/" in n and n.endswith("_eval.json") and not n.startswith("__MACOSX")
        ]
        
        if full_traj_files:
            if progress_callback:
                progress_callback(1, 1, "Loading trace evaluations...")
            
            print(f"📊 Loading {len(full_traj_files)} trace evaluation files...")
            for traj_file in full_traj_files:
                try:
                    with zf.open(traj_file) as f:
                        eval_data = json.load(f)
                        traj_name = eval_data.get("trajectory_name", Path(traj_file).stem.replace("_eval", ""))
                        traj_eval_results[traj_name] = eval_data
                except Exception as e:
                    print(f"  ⚠️  Failed to load {traj_file}: {e}")
            
            print(f"✓ Loaded {len(traj_eval_results)} trace evaluations")
            metadata["traj_eval_results"] = traj_eval_results

        # Load rubric evaluation results if available
        # Look for full_traj_results/rubric_eval_results/*.json
        rubric_eval_results = {}
        rubric_files = [
            n for n in all_files
            if "full_traj_results/rubric_eval_results/" in n and n.endswith(".json") and not n.startswith("__MACOSX")
        ]
        
        if rubric_files:
            if progress_callback:
                progress_callback(1, 1, "Loading rubric evaluations...")
            
            print(f"📋 Loading {len(rubric_files)} rubric evaluation files...")
            for rubric_file in rubric_files:
                try:
                    with zf.open(rubric_file) as f:
                        rubric_data = json.load(f)
                        # Extract task_id from data (try task_id, then trajectory_name, then filename)
                        task_id = rubric_data.get("task_id") or rubric_data.get("trajectory_name")
                        if not task_id:
                            # Extract from filename, removing _rubric_eval suffix if present
                            filename = Path(rubric_file).stem
                            task_id = filename.replace("_rubric_eval", "")
                        rubric_eval_results[task_id] = rubric_data
                except Exception as e:
                    print(f"  ⚠️  Failed to load {rubric_file}: {e}")
            
            print(f"✓ Loaded {len(rubric_eval_results)} rubric evaluations")
            metadata["rubric_eval_results"] = rubric_eval_results

        # Load full trace CLEAR results if available
        # Look for full_traj_results/clear_results/*.zip
        full_traj_clear_files = [
            n for n in all_files
            if "full_traj_results/clear_results/" in n and n.endswith(".zip") and not n.startswith("__MACOSX")
        ]
        
        if full_traj_clear_files:
            if progress_callback:
                progress_callback(1, 1, "Loading full trace CLEAR results...")
            
            print(f"📊 Found full trace CLEAR results: {len(full_traj_clear_files)} files")
            # Store the zip bytes for later loading
            for clear_file in full_traj_clear_files:
                with zf.open(clear_file) as f:
                    metadata["full_traj_clear_results"] = {
                        "zip_bytes": f.read(),
                        "zip_name": Path(clear_file).name,
                    }
                    print(f"✓ Loaded full trace CLEAR results: {Path(clear_file).name}")
                    break  # Only load the first one

    return trajectory_df, agent_results, metadata


def build_workflow_graph(traj_df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict]:
    """Build a directed graph representing the agent workflow."""
    G = nx.DiGraph()
    node_stats = defaultdict(
        lambda: {"count": 0, "tasks": set(), "tool_calls": 0, "agent_calls": 0}
    )

    for task_id, task_group in traj_df.groupby("task_id"):
        task_group = task_group.sort_values("step_in_trace_general")
        agents = task_group["Name"].tolist()

        for i, agent in enumerate(agents):
            node_stats[agent]["count"] += 1
            node_stats[agent]["tasks"].add(task_id)

            if i < len(task_group):
                row = task_group.iloc[i]
                if row.get("tool_or_agent") == "tool":
                    node_stats[agent]["tool_calls"] += 1
                else:
                    node_stats[agent]["agent_calls"] += 1

            if i < len(agents) - 1:
                next_agent = agents[i + 1]
                if G.has_edge(agent, next_agent):
                    G[agent][next_agent]["weight"] += 1
                else:
                    G.add_edge(agent, next_agent, weight=1)

            if not G.has_node(agent):
                G.add_node(agent)

    for agent in node_stats:
        node_stats[agent]["unique_tasks"] = len(node_stats[agent]["tasks"])
        node_stats[agent]["tasks"] = list(node_stats[agent]["tasks"])

    return G, dict(node_stats)


def visualize_workflow_graph_d3(
    G: nx.DiGraph, node_stats: Dict, selected_node: str = None
) -> tuple:
    """Create an interactive D3.js force-directed graph with draggable nodes.
    Returns (html_content, script_content) tuple."""
    import json
    import random
    
    # Generate unique container ID
    container_id = f"d3-graph-{random.randint(1000, 9999)}"
    
    # Convert graph to D3-compatible format
    nodes_data = []
    for node in G.nodes():
        stats = node_stats.get(node, {})
        count = stats.get("count", 0)
        unique_tasks = stats.get("unique_tasks", 0)
        
        nodes_data.append({
            "id": node,
            "label": node,
            "count": count,
            "unique_tasks": unique_tasks,
            "tool_calls": stats.get("tool_calls", 0),
            "agent_calls": stats.get("agent_calls", 0),
        })
    
    links_data = []
    for source, target, data in G.edges(data=True):
        links_data.append({
            "source": source,
            "target": target,
            "weight": data.get("weight", 1),
        })
    
    # HTML content (no script tags)
    html_content = f"""
    <div id="{container_id}" style="width:100%; height:800px; background:white; border-radius:16px; position:relative;"></div>
    """
    
    # JavaScript content (to be added via ui.add_body_html)
    script_content = f"""
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
    // Wait for D3 to load and DOM to be ready
    (function initGraph() {{
        if (typeof d3 === 'undefined') {{
            setTimeout(initGraph, 100);
            return;
        }}
        
        const container = document.getElementById('{container_id}');
        if (!container) {{
            setTimeout(initGraph, 100);
            return;
        }}
        const nodes = {json.dumps(nodes_data)};
        const links = {json.dumps(links_data)};
        const selectedNode = {json.dumps(selected_node)};
        
        const width = container.clientWidth;
        const height = 800;
        
        // Clear any existing SVG
        d3.select('#{container_id}').selectAll('*').remove();
        
        // Create SVG
        const svg = d3.select('#{container_id}')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .call(d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {{
                    g.attr('transform', event.transform);
                }})
            );
        
        const g = svg.append('g');
        
        // Define arrow markers
        svg.append('defs').selectAll('marker')
            .data(['arrow', 'arrow-selected'])
            .enter().append('marker')
            .attr('id', d => d)
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', 0)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
            .append('path')
            .attr('d', 'M0,-5L10,0L0,5')
            .attr('fill', d => d === 'arrow-selected' ? '#6366F1' : '#94A3B8');
        
        // Create force simulation
        const simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(d => d.id).distance(150))
            .force('charge', d3.forceManyBody().strength(-800))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(60));
        
        // Create links
        const link = g.append('g')
            .selectAll('path')
            .data(links)
            .enter().append('path')
            .attr('class', 'link')
            .attr('stroke', '#94A3B8')
            .attr('stroke-width', d => Math.min(d.weight / 2, 8))
            .attr('fill', 'none')
            .attr('marker-end', 'url(#arrow)')
            .style('opacity', 0.6);
        
        // Create link labels
        const linkLabel = g.append('g')
            .selectAll('text')
            .data(links)
            .enter().append('text')
            .attr('class', 'link-label')
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .attr('fill', '#475569')
            .attr('text-anchor', 'middle')
            .text(d => d.weight);
        
        // Create nodes
        const node = g.append('g')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .attr('class', 'node')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended)
            )
            .on('click', (event, d) => {{
                // Emit custom event for node click
                window.parent.postMessage({{
                    type: 'node-click',
                    node: d.id
                }}, '*');
            }});
        
        // Add rectangles to nodes
        node.append('rect')
            .attr('width', 140)
            .attr('height', 40)
            .attr('x', -70)
            .attr('y', -20)
            .attr('rx', 8)
            .attr('ry', 8)
            .attr('fill', d => d.id === selectedNode ? '#F59E0B' : '#6B9BD1')
            .attr('stroke', '#fff')
            .attr('stroke-width', 3)
            .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))')
            .style('cursor', 'move');
        
        // Add text to nodes
        node.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '0.35em')
            .attr('font-size', '13px')
            .attr('font-weight', 'bold')
            .attr('fill', '#1E293B')
            .attr('pointer-events', 'none')
            .text(d => d.label.length > 18 ? d.label.substring(0, 15) + '...' : d.label);
        
        // Add hover effects
        node.on('mouseover', function(event, d) {{
            d3.select(this).select('rect')
                .transition().duration(200)
                .attr('stroke-width', 4)
                .style('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))');
            
            // Show tooltip
            const tooltip = d3.select('body').append('div')
                .attr('class', 'node-tooltip')
                .style('position', 'absolute')
                .style('background', 'white')
                .style('padding', '12px')
                .style('border-radius', '8px')
                .style('box-shadow', '0 4px 12px rgba(0,0,0,0.15)')
                .style('pointer-events', 'none')
                .style('font-size', '13px')
                .style('z-index', '1000')
                .html(`
                    <strong>${{d.label}}</strong><br>
                    Total calls: ${{d.count}}<br>
                    Unique tasks: ${{d.unique_tasks}}<br>
                    Agent calls: ${{d.agent_calls}}
                `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        }})
        .on('mouseout', function() {{
            d3.select(this).select('rect')
                .transition().duration(200)
                .attr('stroke-width', 3)
                .style('filter', 'drop-shadow(0 2px 4px rgba(0,0,0,0.2))');
            
            d3.selectAll('.node-tooltip').remove();
        }})
        .on('mousemove', function(event) {{
            d3.selectAll('.node-tooltip')
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        }});
        
        // Update positions on simulation tick
        simulation.on('tick', () => {{
            link.attr('d', d => {{
                const dx = d.target.x - d.source.x;
                const dy = d.target.y - d.source.y;
                const dr = Math.sqrt(dx * dx + dy * dy) * 2;
                return `M${{d.source.x}},${{d.source.y}}A${{dr}},${{dr}} 0 0,1 ${{d.target.x}},${{d.target.y}}`;
            }});
            
            linkLabel
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2);
            
            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        // Drag functions
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            // Keep node fixed after drag
            // d.fx = null;
            // d.fy = null;
        }}
    }})();
    </script>
    """
    
    return html_content, script_content


def visualize_workflow_graph(
    G: nx.DiGraph, node_stats: Dict, selected_node: str = None, layout: str = "shell"
) -> go.Figure:
    """Create an interactive workflow visualization using Plotly with enhanced styling.
    
    Args:
        G: NetworkX DiGraph
        node_stats: Dictionary of node statistics
        selected_node: Node to highlight
        layout: Layout algorithm - "shell", "spring", "kamada_kawai", "spectral", or "circular"
    """
    # Apply selected layout algorithm
    try:
        if layout == "spring":
            pos = nx.spring_layout(G, k=6.0, iterations=250, seed=42, scale=3.5)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G, scale=3.5)
        elif layout == "spectral":
            pos = nx.spectral_layout(G, scale=3.5)
        elif layout == "circular":
            pos = nx.circular_layout(G, scale=3.0)
        else:  # shell (default)
            pos = nx.shell_layout(G, scale=3.0)
    except Exception as e:
        # Fallback to shell layout if chosen layout fails
        print(f"Layout {layout} failed: {e}, falling back to shell")
        pos = nx.shell_layout(G, scale=3.0)

    edge_traces = []
    edge_labels = []
    arrow_traces = []
    processed_edges = set()

    # Enhanced edge colors with better visual hierarchy
    edge_color = "#CBD5E1"  # Softer gray for edges
    edge_active_color = "#818CF8"  # Lighter indigo for active edges
    label_color = "#1E293B"  # Darker for better readability
    label_bg_color = "rgba(255, 255, 255, 0.98)"  # More opaque background

    for edge in G.edges():
        source, target = edge[0], edge[1]
        if (target, source) in processed_edges:
            continue
        has_reverse = G.has_edge(target, source)

        if has_reverse:
            weight_forward = G[source][target]["weight"]
            weight_reverse = G[target][source]["weight"]
            processed_edges.add((source, target))
            processed_edges.add((target, source))

            x0, y0 = pos[source]
            x1, y1 = pos[target]
            dx, dy = x1 - x0, y1 - y0
            length = (dx ** 2 + dy ** 2) ** 0.5

            if length > 0:
                perp_x = -dy / length * 0.05
                perp_y = dx / length * 0.05

                mid_x_fwd = (x0 + x1) / 2 + perp_x
                mid_y_fwd = (y0 + y1) / 2 + perp_y
                edge_traces.append(
                    go.Scatter(
                        x=[x0, mid_x_fwd, x1], y=[y0, mid_y_fwd, y1],
                        mode="lines",
                        line=dict(width=min(weight_forward / 2, 8), color=edge_color, shape="spline"),
                        hoverinfo="skip", showlegend=False, name="",
                    )
                )
                edge_traces.append(
                    go.Scatter(
                        x=[x0, mid_x_fwd, x1], y=[y0, mid_y_fwd, y1],
                        mode="lines",
                        line=dict(width=20, color="rgba(0,0,0,0)", shape="spline"),
                        hovertemplate=f"<b>{source} -> {target}</b><br>Transitions: {weight_forward}<extra></extra>",
                        showlegend=False, name="",
                    )
                )
                lx = (x0 + mid_x_fwd + x1) / 3 + perp_x * 2.0
                ly = (y0 + mid_y_fwd + y1) / 3 + perp_y * 2.0
                edge_labels.append(
                    go.Scatter(
                        x=[lx], y=[ly], mode="text",
                        text=[f"<b>{weight_forward}</b>"],
                        textfont=dict(size=12, color=label_color, family="Inter, sans-serif"),
                        textposition="middle center", hoverinfo="skip", showlegend=False,
                        marker=dict(
                            size=18,
                            color=label_bg_color,
                            line=dict(width=0),
                            symbol="square",
                        ),
                    )
                )
                ax_fwd = mid_x_fwd + 0.4 * (x1 - mid_x_fwd)
                ay_fwd = mid_y_fwd + 0.4 * (y1 - mid_y_fwd)
                da, db = x1 - mid_x_fwd, y1 - mid_y_fwd
                la = (da ** 2 + db ** 2) ** 0.5
                if la > 0:
                    arrow_traces.append(dict(
                        x=ax_fwd, y=ay_fwd,
                        ax=ax_fwd - (da / la) * 0.05, ay=ay_fwd - (db / la) * 0.05,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                        arrowcolor=edge_color, text="",
                    ))

                mid_x_rev = (x1 + x0) / 2 - perp_x
                mid_y_rev = (y1 + y0) / 2 - perp_y
                edge_traces.append(
                    go.Scatter(
                        x=[x1, mid_x_rev, x0], y=[y1, mid_y_rev, y0],
                        mode="lines",
                        line=dict(width=min(weight_reverse / 2, 8), color=edge_color, shape="spline"),
                        hoverinfo="skip", showlegend=False, name="",
                    )
                )
                edge_traces.append(
                    go.Scatter(
                        x=[x1, mid_x_rev, x0], y=[y1, mid_y_rev, y0],
                        mode="lines",
                        line=dict(width=20, color="rgba(0,0,0,0)", shape="spline"),
                        hovertemplate=f"<b>{target} -> {source}</b><br>Transitions: {weight_reverse}<extra></extra>",
                        showlegend=False, name="",
                    )
                )
                lx2 = (x1 + mid_x_rev + x0) / 3 - perp_x * 2.0
                ly2 = (y1 + mid_y_rev + y0) / 3 - perp_y * 2.0
                edge_labels.append(
                    go.Scatter(
                        x=[lx2], y=[ly2], mode="text",
                        text=[f"<b>{weight_reverse}</b>"],
                        textfont=dict(size=12, color=label_color, family="Inter, sans-serif"),
                        textposition="middle center", hoverinfo="skip", showlegend=False,
                        marker=dict(
                            size=18,
                            color=label_bg_color,
                            line=dict(width=0),
                            symbol="square",
                        ),
                    )
                )
                ax_rev = mid_x_rev + 0.4 * (x0 - mid_x_rev)
                ay_rev = mid_y_rev + 0.4 * (y0 - mid_y_rev)
                da2, db2 = x0 - mid_x_rev, y0 - mid_y_rev
                la2 = (da2 ** 2 + db2 ** 2) ** 0.5
                if la2 > 0:
                    arrow_traces.append(dict(
                        x=ax_rev, y=ay_rev,
                        ax=ax_rev - (da2 / la2) * 0.05, ay=ay_rev - (db2 / la2) * 0.05,
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                        arrowcolor=edge_color, text="",
                    ))
        else:
            processed_edges.add((source, target))
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            weight = G[source][target]["weight"]

            dx, dy = x1 - x0, y1 - y0
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length > 0:
                arrow_x = x0 + 0.8 * dx
                arrow_y = y0 + 0.8 * dy
                arrow_traces.append(dict(
                    x=arrow_x, y=arrow_y,
                    ax=arrow_x - (dx / length) * 0.05,
                    ay=arrow_y - (dy / length) * 0.05,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2,
                    arrowcolor=edge_color, text="",
                ))

            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                    line=dict(width=min(weight / 2, 8), color=edge_color),
                    hoverinfo="skip", showlegend=False, name="",
                )
            )
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                    line=dict(width=20, color="rgba(0,0,0,0)"),
                    hovertemplate=f"<b>{source} -> {target}</b><br>Transitions: {weight}<extra></extra>",
                    showlegend=False, name="",
                )
            )
            # Position label with offset to avoid overlap with arrow
            lx = x0 + 0.5 * dx
            ly = y0 + 0.5 * dy + 0.12
            edge_labels.append(
                go.Scatter(
                    x=[lx], y=[ly], mode="text",
                    text=[f"<b>{weight}</b>"],
                    textfont=dict(size=12, color=label_color, family="Inter, sans-serif"),
                    textposition="middle center", hoverinfo="skip", showlegend=False,
                    marker=dict(
                        size=18,
                        color=label_bg_color,
                        line=dict(width=0),
                        symbol="square",
                    ),
                )
            )

    # Nodes - Calculate sizes based on call counts with enhanced visual styling
    node_x, node_y, node_text, node_color, node_size, node_labels, node_line_colors = [], [], [], [], [], [], []
    
    # Get min and max counts for better size scaling
    all_counts = [node_stats.get(node, {}).get("count", 0) for node in G.nodes()]
    min_count = min(all_counts) if all_counts else 0
    max_count = max(all_counts) if all_counts else 1
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        stats = node_stats.get(node, {})
        count = stats.get("count", 0)
        unique_tasks = stats.get("unique_tasks", 0)
        tool_calls = stats.get("tool_calls", 0)
        agent_calls = stats.get("agent_calls", 0)
        
        # Enhanced hover text with better formatting
        node_text.append(
            f"<b style='font-size:14px'>{node}</b><br><br>"
            f"<b>Total Calls:</b> {count}<br>"
            f"<b>Unique Tasks:</b> {unique_tasks}<br>"
            f"<b>Tool Calls:</b> {tool_calls}<br>"
            f"<b>Agent Calls:</b> {agent_calls}"
        )
        
        # Enhanced color scheme based on selection and activity
        if selected_node and node == selected_node:
            # Selected node: warm amber/orange gradient
            node_color.append("#F59E0B")
            node_line_colors.append("#D97706")
        else:
            # Default nodes: cool indigo gradient based on activity
            if max_count > min_count and count > 0:
                normalized = (count - min_count) / (max_count - min_count)
                # Gradient from light to dark indigo based on activity
                if normalized > 0.7:
                    node_color.append("#4F46E5")  # Dark indigo for high activity
                    node_line_colors.append("#4338CA")
                elif normalized > 0.4:
                    node_color.append("#6366F1")  # Medium indigo
                    node_line_colors.append("#4F46E5")
                else:
                    node_color.append("#818CF8")  # Light indigo for low activity
                    node_line_colors.append("#6366F1")
            else:
                node_color.append("#A5B4FC")  # Very light indigo for minimal activity
                node_line_colors.append("#818CF8")
        
        # Scale node size based on call count (30 to 90 range for better visibility)
        if max_count > min_count:
            normalized = (count - min_count) / (max_count - min_count)
            size = 30 + normalized * 60  # Range from 30 to 90
        else:
            size = 50
        node_size.append(size)
        
        # Truncate long node names for display with better formatting
        display_name = node if len(node) <= 22 else node[:19] + "..."
        node_labels.append(display_name)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(
            size=11,
            color="#1E293B",
            family="Inter, sans-serif",
            weight=600
        ),
        hoverinfo="text", hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=3, color=node_line_colors),
            opacity=0.95,
            symbol="circle",
        ),
        showlegend=False,
        customdata=list(G.nodes()),
    )

    fig = go.Figure(data=edge_traces + edge_labels + [node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=40, r=40, t=60),
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=False,
            gridcolor="rgba(241, 245, 249, 0.8)",  # Softer grid
            gridwidth=1,
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showticklabels=False,
            gridcolor="rgba(241, 245, 249, 0.8)",  # Softer grid
            gridwidth=1,
        ),
        height=850,  # Slightly taller for better visibility
        plot_bgcolor="#FAFBFC",  # Very light gray background for subtle depth
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=arrow_traces,
        dragmode="pan",
        modebar=dict(
            orientation="v",
            bgcolor="rgba(255,255,255,0.95)",
            color="#64748B",
            activecolor="#6366F1",
            remove=["select2d", "lasso2d"],
        ),
        # Add subtle title styling
        font=dict(
            family="Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color="#1E293B"
        ),
    )
    
    # Configure interaction modes - allow zoom and pan
    fig.update_xaxes(fixedrange=False)
    fig.update_yaxes(fixedrange=False)
    
    # Additional config to ensure smooth dragging
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d'],  # Remove selection tools
        'scrollZoom': True,  # Enable scroll to zoom
    }
    
    return fig


def extract_trajectory_paths(traj_df: pd.DataFrame, include_partial: bool = True) -> Dict[str, Any]:
    """Extract unique paths from traces.
    
    Args:
        traj_df: DataFrame containing trace data
        include_partial: If True, extract all subsequences; if False, only full traces
    """
    paths = {}
    for task_id, task_group in traj_df.groupby("task_id"):
        task_group = task_group.sort_values("step_in_trace_general")
        agent_sequence = task_group["Name"].tolist()
        
        if include_partial:
            # Extract all possible subsequences (partial paths)
            for start_idx in range(len(agent_sequence)):
                for end_idx in range(start_idx + 1, len(agent_sequence) + 1):
                    partial_path = tuple(agent_sequence[start_idx:end_idx])
                    path_str = " -> ".join(partial_path)
                    path_length = len(partial_path)
                    
                    if path_str not in paths:
                        paths[path_str] = {"path": partial_path, "count": 0, "task_ids": set(), "length": path_length}
                    paths[path_str]["count"] += 1
                    paths[path_str]["task_ids"].add(task_id)
        else:
            # Extract only full trace paths
            full_path = tuple(agent_sequence)
            path_str = " -> ".join(full_path)
            path_length = len(full_path)
            
            if path_str not in paths:
                paths[path_str] = {"path": full_path, "count": 0, "task_ids": set(), "length": path_length}
            paths[path_str]["count"] += 1
            paths[path_str]["task_ids"].add(task_id)
    
    # Convert task_ids sets to lists for JSON serialization
    for path_data in paths.values():
        path_data["task_ids"] = list(path_data["task_ids"])
    
    return paths


def calculate_path_scores(paths: Dict, all_agent_scores: Dict, traj_df: pd.DataFrame = None) -> Dict:
    """Calculate scores for paths. Prefers ground truth traj_score if available, falls back to agent scores."""
    # Check if ground truth trace scores are available
    has_traj_score = traj_df is not None and "traj_score" in traj_df.columns and not traj_df["traj_score"].isnull().all()
    
    for path_str, path_data in paths.items():
        scores = []
        for task_id in path_data["task_ids"]:
            score_found = False
            
            # Try to use ground truth trace score first
            if has_traj_score:
                task_rows = traj_df[traj_df["task_id"] == task_id]
                if not task_rows.empty:
                    traj_score = task_rows["traj_score"].iloc[0]
                    if pd.notna(traj_score):
                        scores.append(float(traj_score))
                        score_found = True
            
            # Fall back to agent scores if no ground truth
            if not score_found:
                for agent_name, agent_data in all_agent_scores.items():
                    if "id_to_score" in agent_data and task_id in agent_data["id_to_score"]:
                        scores.append(agent_data["id_to_score"][task_id])
                        break
        
        if scores:
            path_data["avg_score"] = sum(scores) / len(scores)
            path_data["score_count"] = len(scores)
            # Calculate success rate (scores >= 0.7)
            success_count = sum(1 for s in scores if s >= 0.7)
            path_data["success_rate"] = success_count / len(scores) if scores else 0
        else:
            path_data["avg_score"] = None
            path_data["score_count"] = 0
            path_data["success_rate"] = None
    return paths


def identify_dead_end_nodes(G: nx.DiGraph, traj_df: pd.DataFrame) -> List[Dict]:
    final_nodes = []
    for task_id, task_group in traj_df.groupby("task_id"):
        task_group = task_group.sort_values("step_in_trace_general")
        final_nodes.append(task_group.iloc[-1]["Name"])
    final_node_counts = Counter(final_nodes)
    total_trajectories = len(traj_df["task_id"].unique())
    dead_ends = []
    for node in G.nodes():
        if node in final_node_counts:
            rate = final_node_counts[node] / total_trajectories
            if rate < 0.05:
                dead_ends.append({"node": node, "completion_rate": rate, "completions": final_node_counts[node]})
    return dead_ends


def analyze_agent_positions(traj_df: pd.DataFrame) -> pd.DataFrame:
    agent_positions = defaultdict(list)
    for task_id, task_group in traj_df.groupby("task_id"):
        task_group = task_group.sort_values("step_in_trace_general")
        total_steps = len(task_group)
        for idx, row in enumerate(task_group.iterrows()):
            agent = row[1]["Name"]
            normalized_pos = idx / max(total_steps - 1, 1)
            agent_positions[agent].append({"position": idx, "normalized_position": normalized_pos, "total_steps": total_steps})

    agent_stats = []
    for agent, positions in agent_positions.items():
        agent_stats.append({
            "agent": agent,
            "avg_position": sum(p["position"] for p in positions) / len(positions),
            "avg_normalized_position": sum(p["normalized_position"] for p in positions) / len(positions),
            "appearances": len(positions),
            "early_appearances": sum(1 for p in positions if p["normalized_position"] < 0.33),
            "mid_appearances": sum(1 for p in positions if 0.33 <= p["normalized_position"] < 0.67),
            "late_appearances": sum(1 for p in positions if p["normalized_position"] >= 0.67),
        })
    return pd.DataFrame(agent_stats).sort_values("avg_normalized_position")


def analyze_retry_patterns(traj_df: pd.DataFrame) -> pd.DataFrame:
    retry_data = []
    for task_id, task_group in traj_df.groupby("task_id"):
        task_group = task_group.sort_values("step_in_trace_general")
        agents = task_group["Name"].tolist()
        agent_counts = Counter(agents)
        for agent, count in agent_counts.items():
            if count > 1:
                positions = [i for i, a in enumerate(agents) if a == agent]
                consecutive = all(positions[i + 1] - positions[i] == 1 for i in range(len(positions) - 1))
                retry_data.append({
                    "task_id": task_id, "agent": agent, "retry_count": count - 1,
                    "total_calls": count, "consecutive": consecutive, "positions": positions,
                })
    return pd.DataFrame(retry_data) if retry_data else pd.DataFrame()


def analyze_score_progression(traj_df: pd.DataFrame, all_agent_scores_df: Dict) -> Dict:
    progression_data = defaultdict(list)
    for task_id, task_group in traj_df.groupby("task_id"):
        task_group = task_group.sort_values("step_in_trace_general")
        scores_in_traj = []
        for idx, row in task_group.iterrows():
            agent = row["Name"]
            step_id = row.get("id", "")
            if agent in all_agent_scores_df:
                df_agent = all_agent_scores_df[agent]
                if "id" in df_agent.columns and step_id:
                    step_row = df_agent[df_agent["id"] == step_id]
                    if not step_row.empty and "score" in step_row.columns:
                        score = step_row.iloc[0]["score"]
                        if pd.notna(score):
                            scores_in_traj.append(float(score))
        if len(scores_in_traj) >= 2:
            for i in range(len(scores_in_traj) - 1):
                change = scores_in_traj[i + 1] - scores_in_traj[i]
                progression_data["position"].append(i)
                progression_data["score_change"].append(change)
                progression_data["improving"].append(change > 0)
    return progression_data


def get_issue_analysis(df, max_num_issues=None):
    if "recurring_issues_str" not in df.columns or df["recurring_issues_str"].isnull().all():
        return pd.DataFrame()
    issues_per_row = df["recurring_issues_str"].apply(extract_issues)
    issues_score_df = pd.DataFrame({"issue": issues_per_row, "score": df["score"]})
    issues_score_df_flat = issues_score_df.explode("issue")
    total_stats = len(df)
    issues_stats = issues_score_df_flat.groupby("issue")["score"].agg(["mean", "std"]).round(2)
    issues_stats.index.name = "issue"
    issues_stats["issue_count"] = issues_score_df_flat["issue"].value_counts()
    issues_stats["issue_freq"] = issues_stats.apply(
        lambda r: round(100 * r["issue_count"] / total_stats, 1), axis=1
    )
    return issues_stats


# ─── Global Styles ───────────────────────────────────────────────────────────

GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #F8FAFC;
    line-height: 1.6;
}

.q-page {
    background: #F8FAFC !important;
}

/* Metric card - Enhanced with better shadows and contrast */
.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-align: center;
    min-width: 160px;
}
.metric-card:hover {
    box-shadow: 0 10px 25px rgba(0,0,0,0.1), 0 4px 10px rgba(0,0,0,0.06);
    transform: translateY(-2px);
    border-color: #CBD5E1;
}
.metric-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}

/* Section header - Better contrast */
.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
    letter-spacing: -0.02em;
}
.section-subtitle {
    font-size: 14px;
    color: #475569;
    margin: 4px 0 0 0;
    line-height: 1.5;
}

/* Dashboard header - More vibrant gradient */
.dashboard-header {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%);
    color: white;
    padding: 40px 48px;
    border-radius: 24px;
    margin-bottom: 32px;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.2);
}
.dashboard-header h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 700;
    color: white;
    letter-spacing: -0.02em;
}
.dashboard-header p {
    margin: 12px 0 0;
    opacity: 0.95;
    font-size: 16px;
    color: rgba(255,255,255,0.95);
    line-height: 1.6;
}

/* Agent badge - Better contrast */
.agent-badge {
    display: inline-block;
    background: linear-gradient(135deg, #EEF2FF, #E0E7FF);
    color: #3730A3;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px;
    border: 1px solid #C7D2FE;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.agent-arrow {
    display: inline-block;
    color: #64748B;
    font-size: 18px;
    margin: 0 6px;
    vertical-align: middle;
}

/* Score badges - Enhanced contrast */
.score-good {
    background: #D1FAE5;
    color: #065F46;
    padding: 6px 14px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #A7F3D0;
}
.score-mid {
    background: #FEF3C7;
    color: #92400E;
    padding: 6px 14px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #FDE68A;
}
.score-bad {
    background: #FEE2E2;
    color: #991B1B;
    padding: 6px 14px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #FECACA;
}

/* Step cards */
.step-number {
    background: linear-gradient(135deg, #6366F1, #4F46E5);
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
}
.step-type-badge {
    background: #F1F5F9;
    color: #475569;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #E2E8F0;
}

/* Intent card */
.intent-card {
    background: #EEF2FF;
    border-left: 4px solid #6366F1;
    padding: 16px 20px;
    border-radius: 0 12px 12px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Custom card - Enhanced with better spacing */
.custom-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
}

/* Sidebar styles - Better contrast */
.sidebar-stat {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
}
.sidebar-stat-label {
    color: #CBD5E1;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.sidebar-stat-value {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin-top: 4px;
}

/* Issues table - Better readability */
.issues-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 14px;
    background: white;
    border-radius: 12px;
    overflow: hidden;
}
.issues-table th {
    padding: 14px 16px;
    text-align: left;
    background: #F8FAFC;
    border-bottom: 2px solid #E2E8F0;
    font-weight: 700;
    color: #1E293B;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.issues-table td {
    padding: 14px 16px;
    border-bottom: 1px solid #F1F5F9;
    vertical-align: middle;
    color: #334155;
}
.issues-table tr:hover {
    background: #F8FAFC;
}
.issues-table tbody tr:last-child td {
    border-bottom: none;
}
.severity-bar {
    height: 10px;
    border-radius: 5px;
    transition: width 0.3s ease;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Tabs - Enhanced */
.q-tab {
    text-transform: none !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* Divider */
.section-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #CBD5E1, transparent);
    margin: 28px 0;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 100px 0;
}
.empty-state .icon {
    font-size: 72px;
    margin-bottom: 20px;
    opacity: 0.5;
}
.empty-state h2 {
    color: #0F172A;
    font-weight: 700;
    margin: 0 0 12px;
    font-size: 24px;
}
.empty-state p {
    color: #475569;
    font-size: 16px;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Scrollbar - More visible */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* Chart containers - Better backgrounds and spacing */
.plotly-graph-div {
    background: white !important;
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Data presentation containers */
.q-card {
    margin-bottom: 20px !important;
}

/* Better spacing for rows and columns */
.q-gutter-md > * {
    margin-bottom: 16px !important;
}

/* Section spacing */
.section-container {
    margin-bottom: 32px;
    padding: 24px;
    background: white;
    border-radius: 16px;
    border: 1px solid #E2E8F0;
}

/* Filter section specific styling */
.filter-section {
    background: #F8FAFC;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid #E2E8F0;
}

/* Ensure proper spacing between UI elements */
.nicegui-content > * {
    margin-bottom: 16px;
}

/* Tab panel content spacing */
.q-tab-panel {
    padding: 24px !important;
}
"""


# ─── UI Helper Components ────────────────────────────────────────────────────

def render_metric_card(label: str, value):
    """Render a beautiful metric card."""
    ui.html(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """)


def render_section_header(title: str, subtitle: str = ""):
    """Render a styled section header."""
    html = f'<div class="section-title">{title}</div>'
    if subtitle:
        html += f'<div class="section-subtitle">{subtitle}</div>'
    ui.html(html)


def render_divider():
    ui.html('<div class="section-divider"></div>')


def score_badge_html(score: float) -> str:
    if score >= 0.7:
        cls = "score-good"
    elif score >= 0.4:
        cls = "score-mid"
    else:
        cls = "score-bad"
    return f'<span class="{cls}">{score:.2f}</span>'


def score_to_hex(score: float) -> str:
    """Convert a 0-1 score to a blue gradient color (darker blue for lower scores, lighter/brighter blue for higher scores)."""
    # Use a blue gradient: from indigo (low) -> blue (mid) -> sky blue (high)
    if score <= 0.5:
        # Low scores: transition from indigo (#6366F1) to blue (#3B82F6)
        # Indigo RGB: (99, 102, 241), Blue RGB: (59, 130, 246)
        ratio = score / 0.5
        r = int(99 + (59 - 99) * ratio)
        g = int(102 + (130 - 102) * ratio)
        b = int(241 + (246 - 241) * ratio)
    else:
        # High scores: transition from blue (#3B82F6) to sky blue (#0EA5E9)
        # Blue RGB: (59, 130, 246), Sky Blue RGB: (14, 165, 233)
        ratio = (score - 0.5) / 0.5
        r = int(59 + (14 - 59) * ratio)
        g = int(130 + (165 - 130) * ratio)
        b = int(246 + (233 - 246) * ratio)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_dimension_definition(dimension_name: str) -> str:
    """Get the definition for a dimension, handling various name formats."""
    # Try exact match first
    if dimension_name in ALL_CRITERIA:
        return ALL_CRITERIA[dimension_name]
    
    # Try with underscores replaced by spaces
    normalized = dimension_name.replace(" ", "_")
    if normalized in ALL_CRITERIA:
        return ALL_CRITERIA[normalized]
    
    # Try removing prefixes
    for prefix in ["Step Quality: ", "Trace: "]:
        if dimension_name.startswith(prefix):
            clean_name = dimension_name[len(prefix):]
            if clean_name in ALL_CRITERIA:
                return ALL_CRITERIA[clean_name]
            normalized_clean = clean_name.replace(" ", "_")
            if normalized_clean in ALL_CRITERIA:
                return ALL_CRITERIA[normalized_clean]
    
    return "No definition available for this dimension."


def render_dimension_with_tooltip(dimension_name: str, score: float, n_samples: int, score_color: str, icon: str = "✓"):
    """Render a dimension card with an expandable tooltip showing its definition."""
    definition = get_dimension_definition(dimension_name)
    
    with ui.card().classes("w-full").style(f"border-left: 5px solid {score_color}; padding: 16px; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 10px;"):
        with ui.row().classes("w-full items-start justify-between"):
            with ui.column().classes("flex-1"):
                with ui.row().classes("items-center gap-2"):
                    ui.html(f'<div style="font-size: 13px; color: #64748B; font-weight: 600;">{dimension_name}</div>')
                    # Info icon with tooltip
                    with ui.button(icon="info", on_click=lambda: None).props("flat dense round size=xs").classes("text-blue-500"):
                        ui.tooltip(definition).classes("bg-slate-800 text-white text-sm max-w-md")
                ui.html(f'<div style="font-size: 28px; font-weight: 800; color: #0F172A; margin-top: 6px;">{score:.2f}</div>')
                ui.html(f'<div style="font-size: 11px; color: #94A3B8; margin-top: 4px; font-weight: 500;">n={n_samples}</div>')
            ui.html(f'<div style="font-size: 36px; opacity: 0.15;">{icon}</div>')


# ─── Dashboard State ─────────────────────────────────────────────────────────

class DashboardState:
    """Holds all dashboard state for a single user session."""

    def __init__(self):
        self.traj_df: pd.DataFrame = pd.DataFrame()
        self.agent_results: Dict[str, Dict] = {}
        self.metadata: Dict = {}
        self.statistics: Dict = {}
        self.all_agent_scores: Dict = {}
        self.all_agent_scores_df: Dict[str, pd.DataFrame] = {}
        self.inputs_lookup_df: pd.DataFrame = pd.DataFrame()  # For deduplicated format
        self.full_traj_clear_results: Dict = {}  # For full trace CLEAR results
        self.loaded = False

    def load(self, file_bytes: bytes, progress_callback=None):
        """Load all data from the uploaded ZIP."""
        self.traj_df, self.agent_results, self.metadata = load_data_from_zip(file_bytes, progress_callback)
        
        # Extract inputs_lookup if it was loaded (stored in metadata temporarily)
        if "inputs_lookup_df" in self.metadata:
            self.inputs_lookup_df = self.metadata.pop("inputs_lookup_df")

        # Extract full trace CLEAR results if available
        if "full_traj_clear_results" in self.metadata:
            self.full_traj_clear_results = self.metadata.get("full_traj_clear_results", {})

        if self.traj_df.empty:
            return False

        self.statistics = {
            "unique_tasks": self.traj_df["task_id"].nunique() if "task_id" in self.traj_df.columns else 0,
            "total_rows": len(self.traj_df),
            "unique_agents": self.traj_df["Name"].nunique() if "Name" in self.traj_df.columns else 0,
        }

        # Pre-load agent scores
        self.all_agent_scores = {}
        self.all_agent_scores_df = {}
        # Get trajectory_df from metadata for joining
        trajectory_df = self.metadata.get("trajectory_df")
        
        if progress_callback:
            progress_callback(0, len(self.agent_results), "Loading agent results...")
        
        # Track task_id to traj_score mapping from agent results
        task_id_to_traj_score = {}
        
        for idx, (agent_name, agent_data) in enumerate(self.agent_results.items()):
            try:
                if progress_callback:
                    progress_callback(idx + 1, len(self.agent_results), f"Loading {agent_name}...")
                
                if "zip_bytes" in agent_data:
                    file_bytes_agent = agent_data["zip_bytes"]
                    zip_name = agent_data.get("zip_name", f"{agent_name}.zip")
                    df, _ = load_clear_data_from_bytes(file_bytes_agent, zip_name, trajectory_df)
                    if not df.empty:
                        self.all_agent_scores_df[agent_name] = df
                        if "score" in df.columns and "task_id" in df.columns:
                            id_to_score = dict(zip(df["task_id"], df["score"]))
                            self.all_agent_scores[agent_name] = {
                                "id_to_score": id_to_score,
                                "avg_score": df["score"].mean(),
                            }
                            
                            # Extract traj_score if this is EvaluationResult (trace-level scores)
                            if agent_name == "EvaluationResult" and "traj_score" in df.columns:
                                traj_score_mapping = dict(zip(df["task_id"], df["traj_score"]))
                                task_id_to_traj_score.update(traj_score_mapping)
                                print(f"  📊 Found traj_score in {agent_name}: {len(traj_score_mapping)} task scores")
            except Exception as e:
                print(f"  ⚠️  Error loading {agent_name}: {e}")
                pass

        # Populate traj_score from clear_data CSVs if available (primary source)
        if "task_id_to_traj_score" in self.metadata and "task_id" in self.traj_df.columns:
            task_id_to_traj_score_from_clear_data = self.metadata["task_id_to_traj_score"]
            if task_id_to_traj_score_from_clear_data:
                print(f"📊 Populating traj_score from clear_data CSVs ({len(task_id_to_traj_score_from_clear_data)} tasks)...")
                self.traj_df["traj_score"] = self.traj_df["task_id"].map(task_id_to_traj_score_from_clear_data)
                # Fill any remaining NaN values with 0
                self.traj_df["traj_score"] = self.traj_df["traj_score"].fillna(0.0)
                non_zero_count = (self.traj_df["traj_score"] > 0).sum()
                total_mapped = self.traj_df["traj_score"].notna().sum()
                print(f"  ✓ Mapped {len(task_id_to_traj_score_from_clear_data)} scores to trace data")
                print(f"  ✓ Updated traj_score for {total_mapped} trace rows ({non_zero_count} with non-zero scores)")
        
        # Fallback: Populate traj_score from agent results CSVs if not already populated
        elif task_id_to_traj_score and "task_id" in self.traj_df.columns:
            print(f"📊 Populating traj_score from agent result CSVs ({len(task_id_to_traj_score)} tasks)...")
            self.traj_df["traj_score"] = self.traj_df["task_id"].map(task_id_to_traj_score)
            # Fill any remaining NaN values with 0
            self.traj_df["traj_score"] = self.traj_df["traj_score"].fillna(0.0)
            non_zero_count = (self.traj_df["traj_score"] > 0).sum()
            total_mapped = self.traj_df["traj_score"].notna().sum()
            print(f"  ✓ Mapped {len(task_id_to_traj_score)} scores to trace data")
            print(f"  ✓ Updated traj_score for {total_mapped} trace rows ({non_zero_count} with non-zero scores)")

        # Fallback: Populate traj_score from trace evaluation results if not already populated
        if ("traj_score" not in self.traj_df.columns or self.traj_df["traj_score"].isnull().all()) and \
           "traj_eval_results" in self.metadata and self.metadata["traj_eval_results"]:
            print(f"📊 Populating traj_score from {len(self.metadata['traj_eval_results'])} trace evaluations...")
            task_id_to_score = {}
            for traj_name, eval_data in self.metadata["traj_eval_results"].items():
                # Try different score keys
                score = eval_data.get("overall_score") or eval_data.get("score")
                if score is not None:
                    task_id_to_score[traj_name] = float(score)
            
            if task_id_to_score and "task_id" in self.traj_df.columns:
                # Map scores to trace dataframe
                self.traj_df["traj_score"] = self.traj_df["task_id"].map(task_id_to_score)
                # Fill any remaining NaN values with 0
                self.traj_df["traj_score"] = self.traj_df["traj_score"].fillna(0.0)
                non_zero_count = (self.traj_df["traj_score"] > 0).sum()
                total_mapped = self.traj_df["traj_score"].notna().sum()
                print(f"  ✓ Mapped {len(task_id_to_score)} scores to trace data")
                print(f"  ✓ Updated traj_score for {total_mapped} trace rows ({non_zero_count} with non-zero scores)")

        self.loaded = True
        return True


# ─── Main Dashboard Page ─────────────────────────────────────────────────────

@ui.page("/")
def main_page():
    # Per-user state
    state = DashboardState()

    # -- Inject global CSS --
    ui.add_head_html(f"<style>{GLOBAL_CSS}</style>")
    ui.add_head_html('<link rel="preconnect" href="https://fonts.googleapis.com">')
    ui.add_head_html('<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">')

    # -- References for dynamic containers --
    main_content_ref = None
    sidebar_summary_ref = None

    # ── Sidebar (Left Drawer) ────────────────────────────────────────
    with ui.left_drawer(value=True).classes("bg-[#1E293B]").style("width: 300px; padding: 0;"):
        with ui.column().classes("w-full items-center p-4 gap-0"):
            ui.html("""
                <div style="text-align:center; padding:16px 0 8px;">
                    <div style="font-size:18px; font-weight:700; color:white; margin-top:4px;">Agentic CLEAR</div>
                    <div style="font-size:12px; color:#94A3B8; margin-top:2px;">Workflow Analysis Dashboard</div>
                </div>
            """)

        ui.separator().classes("bg-[#334155]")

        with ui.column().classes("w-full p-4 gap-3"):
            ui.html('<div style="font-size:13px; font-weight:600; color:#E2E8F0; letter-spacing:0.05em; text-transform:uppercase;">Data Upload</div>')

            # Upload widget with increased max file size (500MB for large trace data)
            upload_el = ui.upload(
                label="Upload ui_results.zip",
                auto_upload=True,
                on_upload=lambda e: handle_upload(e),
                max_file_size=500 * 1024 * 1024,  # 500MB limit
            ).props('accept=.zip color=deep-purple-9 flat bordered').classes("w-full").style(
                "border: 1px dashed rgba(255,255,255,0.2); border-radius: 10px;"
            )

        # Sidebar summary container
        sidebar_summary_container = ui.column().classes("w-full p-4 gap-2")

        with sidebar_summary_container:
            sidebar_summary_ref = ui.column().classes("w-full gap-2")
            with sidebar_summary_ref:
                ui.html("""
                    <div style="text-align:center; padding:24px 0; color:#94A3B8;">
                        <div style="font-size:40px; margin-bottom:8px;">&#128194;</div>
                        <div style="font-size:13px;">Upload a <code style="color:#818CF8;">ui_results.zip</code><br>to get started</div>
                    </div>
                """)

    # ── Main Content Area ────────────────────────────────────────────
    # Zoom state
    zoom_level = {"value": 100}
    
    with ui.column().classes("w-full max-w-[1400px] mx-auto p-8 gap-0").style("position: relative;") as main_container:
        # Zoom controls (fixed position in top-right)
        with ui.row().classes("gap-2").style(
            "position: fixed; top: 20px; right: 20px; z-index: 1000; "
            "background: white; padding: 8px 12px; border-radius: 8px; "
            "box-shadow: 0 2px 8px rgba(0,0,0,0.15);"
        ):
            ui.label("Zoom:").classes("text-sm font-semibold self-center")
            ui.button(
                icon="remove",
                on_click=lambda: update_zoom(zoom_level["value"] - 10)
            ).props("flat dense round size=sm").classes("text-gray-600")
            zoom_display = ui.label(f"{zoom_level['value']}%").classes("text-sm font-mono self-center w-12 text-center")
            ui.button(
                icon="add",
                on_click=lambda: update_zoom(zoom_level["value"] + 10)
            ).props("flat dense round size=sm").classes("text-gray-600")
            ui.button(
                "Reset",
                on_click=lambda: update_zoom(100)
            ).props("flat dense size=sm").classes("text-gray-600")
        
        # Main content container with zoom transform (includes header)
        main_content_container = ui.column().classes("w-full gap-6").style(
            f"transform: scale({zoom_level['value'] / 100}); "
            "transform-origin: top center; "
            "transition: transform 0.2s ease;"
        )
        
        with main_content_container:
            # Header inside zoomable container
            ui.html("""
                <div class="dashboard-header">
                    <h1>Agentic Workflow Analysis</h1>
                    <p>Explore agent traces, analyze performance, and discover patterns in multi-agent systems</p>
                </div>
            """)
        
        def update_zoom(new_zoom):
            # Clamp zoom between 50% and 200%
            new_zoom = max(50, min(200, new_zoom))
            zoom_level["value"] = new_zoom
            zoom_display.text = f"{new_zoom}%"
            # When zooming out, increase width to compensate and maintain proportions
            width_compensation = 100 / new_zoom * 100  # Inverse of zoom
            main_content_container.style(
                f"transform: scale({new_zoom / 100}); "
                "transform-origin: top center; "
                "transition: transform 0.2s ease; "
                f"width: {width_compensation}%;"
            )

        with main_content_container:
            main_content_ref = ui.column().classes("w-full gap-6")
            with main_content_ref:
                ui.html("""
                    <div class="empty-state">
                        <div class="icon">&#128202;</div>
                        <h2>No Data Loaded</h2>
                        <p>Upload a <code>ui_results.zip</code> file in the sidebar to begin exploring your agent workflows.</p>
                    </div>
                """)

    # ── Upload Handler ───────────────────────────────────────────────
    async def handle_upload(e: events.UploadEventArguments):
        nonlocal state
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Show loading notification with progress
        progress_notification = ui.notification(
            "Reading file...",
            type="ongoing",
            position="top",
            timeout=None,
            close_button=False
        )
        
        try:
            file_bytes = await e.file.read()
            
            # Progress callback to update notification
            def update_progress(current, total, message):
                if total > 0:
                    percent = int((current / total) * 100)
                    progress_notification.message = f"{message} ({current}/{total}) - {percent}%"
                else:
                    progress_notification.message = message
            
            # Run the blocking load operation in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                success = await loop.run_in_executor(
                    executor,
                    state.load,
                    file_bytes,
                    update_progress
                )
            
            # Close progress notification
            progress_notification.dismiss()
            
        except Exception as ex:
            progress_notification.dismiss()
            ui.notify(f"Error loading ZIP: {ex}", type="negative", timeout=5000)
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return

        if not success:
            ui.notify("No trace data found in ZIP file", type="warning")
            return

        ui.notify(
            f"✓ Loaded {len(state.agent_results)} agents, "
            f"{state.statistics.get('unique_tasks', 0)} traces, "
            f"{state.statistics.get('total_rows', 0)} records",
            type="positive",
            timeout=3000
        )

        # Update sidebar summary
        sidebar_summary_ref.clear()
        with sidebar_summary_ref:
            ui.separator().classes("bg-[#334155]")
            ui.html('<div style="font-size:13px; font-weight:600; color:#E2E8F0; letter-spacing:0.05em; text-transform:uppercase; margin-bottom:4px;">Loaded Data</div>')
            ui.html(f"""
                <div class="sidebar-stat">
                    <div class="sidebar-stat-label">Agents</div>
                    <div class="sidebar-stat-value">{state.statistics.get("unique_agents", 0)}</div>
                    <div class="sidebar-stat-label" style="margin-top:8px;">Traces</div>
                    <div class="sidebar-stat-value">{state.statistics.get("unique_tasks", 0)}</div>
                    <div class="sidebar-stat-label" style="margin-top:8px;">Records</div>
                    <div class="sidebar-stat-value">{state.statistics.get("total_rows", 0)}</div>
                </div>
            """)

        # Rebuild main content
        main_content_ref.clear()
        with main_content_ref:
            build_dashboard_tabs(state)

    # ── Build Dashboard ──────────────────────────────────────────────
    def build_dashboard_tabs(state: DashboardState):
        """Build all 5 dashboard tabs."""
        # Shared state for cross-tab communication
        selected_agent_ref = {"value": None}
        
        with ui.tabs().classes("w-full").props("dense active-color=deep-purple-6 indicator-color=deep-purple-6") as tabs:
            tab_workflow = ui.tab("Workflow View").props("icon=analytics no-caps")
            tab_node = ui.tab("Node Analysis").props("icon=search no-caps")
            tab_traj = ui.tab("Trace Explorer").props("icon=explore no-caps")
            tab_path = ui.tab("Predictive Patterns").props("icon=route no-caps")
            tab_temporal = ui.tab("Temporal Analysis").props("icon=schedule no-caps")
            tab_prediction = ui.tab("Score Prediction").props("icon=insights no-caps")

        with ui.tab_panels(tabs, value=tab_workflow).classes("w-full") as tab_panels:
            with ui.tab_panel(tab_workflow).classes("p-6"):
                build_workflow_tab(state, tabs, tab_node, selected_agent_ref)

            with ui.tab_panel(tab_node).classes("p-6"):
                build_node_analysis_tab(state, selected_agent_ref)

            with ui.tab_panel(tab_traj).classes("p-6"):
                build_trajectory_tab(state)

            with ui.tab_panel(tab_path).classes("p-6"):
                build_path_analysis_tab(state)

            with ui.tab_panel(tab_temporal).classes("p-6"):
                build_temporal_tab(state)

            with ui.tab_panel(tab_prediction).classes("p-6"):
                build_score_prediction_tab(state)

    # ────────────────────────────────────────────────────────────────
    # TAB 1: Workflow View
    # ────────────────────────────────────────────────────────────────
    def build_workflow_tab(state: DashboardState, tabs=None, tab_node=None, selected_agent_ref=None):
        G, node_stats = build_workflow_graph(state.traj_df)

        # Graph Section
        render_section_header("Agent Workflow Graph", "Click on a node to view its analysis or select an agent to highlight")

        agent_options = ["None"] + list(G.nodes())
        selected_node_val = {"value": None}
        layout_choice = {"value": "shell"}
        graph_container = ui.column().classes("w-full")

        def update_graph():
            graph_container.clear()
            with graph_container:
                # Use enhanced Plotly visualization
                fig = visualize_workflow_graph(G, node_stats, selected_node_val["value"], layout_choice["value"])
                plot = ui.plotly(fig).classes("w-full").style("border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);")
                
                # Add click handler to navigate to node analysis tab
                if tabs and tab_node and selected_agent_ref is not None:
                    def on_graph_click(e):
                        # Get clicked point data
                        if e.args and 'points' in e.args and len(e.args['points']) > 0:
                            point = e.args['points'][0]
                            # Check if it's a node (has text attribute)
                            if 'text' in point and point['text']:
                                clicked_node = point['text']
                                # Store selected agent for node analysis tab
                                selected_agent_ref["value"] = clicked_node
                                # Switch to node analysis tab
                                tabs.set_value(tab_node)
                                ui.notify(f"Navigating to analysis for: {clicked_node}", type="info")
                    
                    plot.on('plotly_click', on_graph_click)

        with ui.row().classes("w-full items-center gap-4 mb-4"):
            ui.select(
                options={
                    "shell": "Shell (Concentric) - Default",
                    "spring": "Spring (Force-directed)",
                    "kamada_kawai": "Kamada-Kawai (Balanced)",
                    "spectral": "Spectral (Eigenvalue)",
                    "circular": "Circular"
                },
                value="shell",
                label="Layout Algorithm",
                on_change=lambda e: (layout_choice.update({"value": e.value}), update_graph())
            ).classes("w-64").props("outlined dense")
            
            select = ui.select(
                agent_options, value="None", label="Highlight Agent"
            ).classes("flex-grow").props("outlined dense")

            def on_agent_change(e):
                # NiceGUI select sends the new value in e.args
                val = e.args
                
                # Extract the actual agent name from the event
                # NiceGUI can send either a string or a dict with 'value' and 'label'
                if isinstance(val, dict):
                    # Use 'label' which contains the agent name
                    val = val.get('label', val.get('value'))
                
                # Convert to string to ensure consistency
                val = str(val) if val is not None else None
                
                # Handle the value
                if val == "None" or val is None:
                    selected_node_val["value"] = None
                    update_graph()
                    return
                
                # val should be the agent name string
                selected_node_val["value"] = val
                update_graph()
                show_node_details(val)

            select.on("update:model-value", on_agent_change)

        update_graph()

        # Node Usage Histogram
        render_divider()
        render_section_header("Node Usage Frequency", "Number of times each node was called across all traces")
        if node_stats:
            nodes_sorted = sorted(node_stats.items(), key=lambda x: x[1]["count"], reverse=True)
            node_names = [n for n, _ in nodes_sorted]
            node_counts = [s["count"] for _, s in nodes_sorted]
            node_unique_tasks = [s["unique_tasks"] for _, s in nodes_sorted]

            hist_fig = go.Figure()
            hist_fig.add_trace(go.Bar(
                x=node_names,
                y=node_counts,
                name="Total Calls",
                marker_color=[COLORS.get("primary", "#6366F1")] * len(node_names),
                customdata=list(zip(node_unique_tasks, node_counts)),
                hovertemplate="<b>%{x}</b><br>Total Calls: %{y}<br>Unique Tasks: %{customdata[0]}<extra></extra>",
                text=node_counts,
                textposition="outside",
            ))
            hist_fig.update_layout(
                xaxis_title="Node",
                yaxis_title="Total Calls",
                plot_bgcolor="white",
                paper_bgcolor="white",
                margin=dict(l=60, r=20, t=80, b=120),
                height=420,
                xaxis=dict(tickangle=-35, tickfont=dict(size=11)),
                yaxis=dict(
                    gridcolor="#E2E8F0",
                    title_standoff=20,
                    range=[0, max(node_counts) * 1.18] if node_counts else None,
                ),
                bargap=0.25,
                uniformtext=dict(minsize=9, mode="hide"),
            )
            ui.plotly(hist_fig).classes("w-full").style(
                "border-radius: 12px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.08);"
            )

        render_divider()

        # Selected node details container
        node_details_container = ui.column().classes("w-full gap-2")

        def show_node_details(node_name):
            node_details_container.clear()
            with node_details_container:
                render_divider()
                render_section_header(f"Agent Details: {node_name}", "Statistics and connections for the selected agent")
                
                # Debug: Check if node exists in stats
               # if node_name not in node_stats:
               #     ui.html(f'<div style="text-align:center; padding:40px; color:#F59E0B;">Agent "{node_name}" not found in node_stats. Available agents: {list(node_stats.keys())}</div>')
                #    return
                
                stats = node_stats[node_name]

                with ui.row().classes("w-full gap-4 justify-center"):
                    render_metric_card("Total Calls", stats.get("count", 0))
                    render_metric_card("Unique Tasks", stats.get("unique_tasks", 0))
                    render_metric_card("Tool Calls", stats.get("tool_calls", 0))
                    render_metric_card("Agent Calls", stats.get("agent_calls", 0))

                with ui.row().classes("w-full gap-6"):
                    with ui.column().classes("flex-1"):
                        ui.label("Incoming Connections").classes("text-sm font-semibold text-slate-600")
                        incoming = [(pred, G[pred][node_name]["weight"]) for pred in G.predecessors(node_name)]
                        if incoming:
                            for pred, weight in sorted(incoming, key=lambda x: x[1], reverse=True):
                                ui.html(f'<span class="agent-badge">{pred}</span> <span class="agent-arrow">&#8594;</span> <strong>{weight}</strong> transitions')
                        else:
                            ui.label("None").classes("text-slate-400 italic")

                    with ui.column().classes("flex-1"):
                        ui.label("Outgoing Connections").classes("text-sm font-semibold text-slate-600")
                        outgoing = [(succ, G[node_name][succ]["weight"]) for succ in G.successors(node_name)]
                        if outgoing:
                            for succ, weight in sorted(outgoing, key=lambda x: x[1], reverse=True):
                                ui.html(f'<span class="agent-arrow">&#8594;</span> <span class="agent-badge">{succ}</span> <strong>{weight}</strong> transitions')
                        else:
                            ui.label("None").classes("text-slate-400 italic")

        render_divider()

        # Average Trace Evaluation Scores (from traj_eval_results) - MOVED ABOVE ISSUES
        traj_eval_results = state.metadata.get("traj_eval_results", {})
        
        if traj_eval_results:
            render_section_header("📊 Average Trace Evaluation", "Average scores across all evaluated traces")
            
            # Calculate averages across all traces
            all_overall_scores = []
            all_dimension_scores = defaultdict(list)
            
            # Debug: print structure of first trace evaluation
            # if traj_eval_results:
            #     first_traj = next(iter(traj_eval_results.values()))
            #     print(f"🔍 Debug - First trace eval keys: {first_traj.keys()}")
            #     if "parsed_evaluation" in first_traj:
            #         print(f"🔍 Debug - parsed_evaluation keys: {first_traj['parsed_evaluation'].keys()}")
            #     if "dimension_scores" in first_traj:
            #         print(f"🔍 Debug - dimension_scores: {first_traj['dimension_scores']}")
            #
            for traj_name, eval_data in traj_eval_results.items():
                # Collect overall scores
                overall_score = eval_data.get("overall_score")
                if overall_score is not None:
                    try:
                        all_overall_scores.append(float(overall_score))
                    except (ValueError, TypeError):
                        print(f"⚠️  Warning: Could not convert overall_score to float: {overall_score}")
                
                # Collect dimension scores - check both dimension_scores and parsed_evaluation
                dimension_scores = eval_data.get("dimension_scores", {})
                parsed_eval = eval_data.get("parsed_evaluation", {})
                
                # Try to get dimensions from parsed_evaluation first (preferred structure)
                step_quality = parsed_eval.get("step_quality_dimensions", {})
                trajectory_dims = parsed_eval.get("trajectory_dimensions", {})
                
                # Collect step quality dimension scores
                for criterion, details in step_quality.items():
                    score = details.get("score")
                    if score is not None:
                        try:
                            all_dimension_scores[f"Step Quality: {criterion}"].append(float(score))
                        except (ValueError, TypeError):
                            print(f"⚠️  Warning: Could not convert score to float for {criterion}: {score}")
                
                # Collect trace dimension scores
                for criterion, details in trajectory_dims.items():
                    score = details.get("score")
                    if score is not None:
                        try:
                            all_dimension_scores[f"Trace: {criterion}"].append(float(score))
                        except (ValueError, TypeError):
                            print(f"⚠️  Warning: Could not convert score to float for {criterion}: {score}")
                
                # Fallback: if no dimensions found in parsed_evaluation, try dimension_scores directly
                if not step_quality and not trajectory_dims and dimension_scores:
                    for criterion, score in dimension_scores.items():
                        if score is not None:
                            try:
                                all_dimension_scores[criterion].append(float(score))
                            except (ValueError, TypeError):
                                print(f"⚠️  Warning: Could not convert score to float for {criterion}: {score}")
            
            # Calculate and display averages
            if all_overall_scores:
                avg_overall = np.mean(all_overall_scores)
                
                # Display average overall score in an expandable card with improved colors
                with ui.expansion("🎯 Average Overall Score", icon="analytics", value=True).classes("w-full").style("background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%); border: 2px solid #0EA5E9; margin-bottom: 16px; border-radius: 12px;"):
                    with ui.card().classes("w-full").style("background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%); color: white; padding: 24px; border-radius: 12px; box-shadow: 0 10px 25px rgba(14, 165, 233, 0.3);"):
                        ui.html(f"""
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div>
                                    <div style="font-size: 14px; opacity: 0.95; margin-bottom: 6px; letter-spacing: 0.5px;">Average Overall Score</div>
                                    <div style="font-size: 52px; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">{avg_overall:.2f}</div>
                                    <div style="font-size: 13px; opacity: 0.85; margin-top: 6px;">Based on {len(all_overall_scores)} traces</div>
                                </div>
                                <div style="font-size: 72px; opacity: 0.25;">📊</div>
                            </div>
                        """)
                
                # Display average dimension scores in an expandable section with improved colors
                if all_dimension_scores:
                    print(f"📊 Displaying {len(all_dimension_scores)} dimension types with scores")
                    with ui.expansion("📈 Average Dimension Scores", icon="bar_chart", value=False).classes("w-full").style("background: linear-gradient(135deg, #E0F2FE 0%, #BAE6FD 100%); border: 2px solid #0EA5E9; margin-bottom: 16px; border-radius: 12px;"):
                        # Separate step quality and trace dimensions
                        step_quality_dims = {k: v for k, v in all_dimension_scores.items() if k.startswith("Step Quality:")}
                        trajectory_level_dims = {k: v for k, v in all_dimension_scores.items() if k.startswith("Trace:")}
                        other_dims = {k: v for k, v in all_dimension_scores.items() if not k.startswith("Step Quality:") and not k.startswith("Trace:")}
                        
                        print(f"  Step quality dims: {len(step_quality_dims)}, Trace dims: {len(trajectory_level_dims)}, Other dims: {len(other_dims)}")
                        
                        if step_quality_dims:
                            ui.html('<div style="font-weight: 700; color: #0F172A; margin: 16px 0 12px 0; font-size: 16px; letter-spacing: 0.3px;">Step Quality Dimensions</div>')
                            
                            # Create a grid for dimension scores
                            with ui.grid(columns=2).classes("w-full gap-4"):
                                for dimension, scores in sorted(step_quality_dims.items()):
                                    avg_score = np.mean(scores)
                                    score_color = score_to_hex(float(avg_score))
                                    dimension_name = dimension.replace("Step Quality: ", "")
                                    render_dimension_with_tooltip(dimension_name, float(avg_score), len(scores), score_color, "✓")
                        
                        if trajectory_level_dims:
                            ui.html('<div style="font-weight: 700; color: #0F172A; margin: 24px 0 12px 0; font-size: 16px; letter-spacing: 0.3px;">Trace-Level Dimensions</div>')
                            
                            # Create a grid for dimension scores
                            with ui.grid(columns=2).classes("w-full gap-4"):
                                for dimension, scores in sorted(trajectory_level_dims.items()):
                                    avg_score = np.mean(scores)
                                    score_color = score_to_hex(float(avg_score))
                                    dimension_name = dimension.replace("Trace: ", "")
                                    render_dimension_with_tooltip(dimension_name, float(avg_score), len(scores), score_color, "📈")
                        
                        # Display other dimensions if they exist (fallback for different data structures)
                        if other_dims:
                            ui.html('<div style="font-weight: 700; color: #0F172A; margin: 24px 0 12px 0; font-size: 16px; letter-spacing: 0.3px;">Evaluation Dimensions</div>')
                            
                            # Create a grid for dimension scores
                            with ui.grid(columns=2).classes("w-full gap-4"):
                                for dimension, scores in sorted(other_dims.items()):
                                    avg_score = np.mean(scores)
                                    score_color = score_to_hex(float(avg_score))
                                    render_dimension_with_tooltip(dimension, float(avg_score), len(scores), score_color, "📊")
            
            render_divider()

        # Recurring Issues Table (ONLY from full trace CLEAR results)
        all_issues_list = []
        full_traj_df = pd.DataFrame()  # Initialize to empty to avoid unbound errors
        
        # Use full trace CLEAR results if available
        if state.full_traj_clear_results:
            try:
                print(f"📊 Loading full trace CLEAR results for workflow issues...")
                file_bytes = state.full_traj_clear_results["zip_bytes"]
                zip_name = state.full_traj_clear_results.get("zip_name", "full_traj_clear_results.zip")
                trajectory_df = state.metadata.get("trajectory_df")
                full_traj_df, _ = load_clear_data_from_bytes(file_bytes, zip_name, trajectory_df)
                
                print(f"  Loaded {len(full_traj_df)} rows from full trace CLEAR results")
                print(f"  Columns: {list(full_traj_df.columns)}")
                
                if not full_traj_df.empty and "recurring_issues_str" in full_traj_df.columns:
                    for issues_str in full_traj_df["recurring_issues_str"].dropna():
                        issues = extract_issues(issues_str)
                        all_issues_list.extend(issues)
                    print(f"  Extracted {len(all_issues_list)} issues from full trace data")
                else:
                    print(f"  No recurring_issues_str column found or dataframe is empty")
            except Exception as e:
                print(f"❌ Could not load full trace CLEAR results for issues: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("ℹ️  No full trace CLEAR results available")
        
        # Only display if we have full trace data
        if all_issues_list:
            print(f"✓ Displaying {len(all_issues_list)} issues in workflow tab")
            render_section_header("Recurring Issues Across Workflow", "Most common issues identified in full trace evaluations")
            
            # Count issues and their frequency
            issue_counter = Counter(all_issues_list)
            issues_data = []
            
            # Total rows (traces) for frequency calculation
            total_rows = len(full_traj_df) if not full_traj_df.empty else len(all_issues_list)
            
            for issue, count in issue_counter.most_common():
                # Compute mean score for severity if score column available
                if not full_traj_df.empty and "score" in full_traj_df.columns and "recurring_issues_str" in full_traj_df.columns:
                    issue_rows = full_traj_df[full_traj_df["recurring_issues_str"].apply(
                        lambda x: issue in extract_issues(x) if pd.notna(x) else False
                    )]
                    mean_score = float(issue_rows["score"].mean()) if not issue_rows.empty else 0.5
                else:
                    mean_score = 0.5
                freq = round(100 * count / total_rows, 1) if total_rows > 0 else 0
                severity = round(1 - mean_score, 2)
                issues_data.append({
                    "issue": issue, "count": count,
                    "freq": freq, "severity": severity,
                    "mean_score": mean_score,
                })

            # Separate NO_ISSUE from other issues
            display_issues = [d for d in issues_data if d["issue"] not in [NO_ISSUE, OTHER]]
            no_issue_data = [d for d in issues_data if d["issue"] == NO_ISSUE]

            if display_issues or no_issue_data:
                max_freq = max(d["freq"] for d in display_issues) if display_issues else 1

                def freq_badge_color(freq, max_f):
                    """Return a color for the frequency: red (high) → amber → green (low) scale."""
                    t = min(freq / max_f, 1.0) if max_f > 0 else 0
                    if t >= 0.66:
                        return "#FEE2E2", "#DC2626"   # red (high freq)
                    elif t >= 0.33:
                        return "#FEF9C3", "#D97706"   # amber
                    else:
                        return "#D1FAE5", "#16A34A"   # green (low freq)

                table_html = (
                    '<table class="issues-table" style="'
                    'width:100%; border-collapse:collapse; font-size:14px;'
                    'box-shadow:0 2px 8px rgba(0,0,0,0.07); border-radius:10px; overflow:hidden;">'
                    '<thead><tr style="background:#4F46E5; color:#fff;'
                    'font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">'
                    '<th style="padding:11px 16px; text-align:left; font-weight:600;">Issue</th>'
                    '<th style="padding:11px 16px; text-align:center; font-weight:600;">Count</th>'
                    '<th style="padding:11px 16px; text-align:center; font-weight:600;">Frequency</th>'
                    '<th style="padding:11px 16px; text-align:center; font-weight:600;">Severity</th>'
                    '</tr></thead><tbody>'
                )

                for i, d in enumerate(display_issues):
                    row_bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
                    freq_bg, freq_fg = freq_badge_color(d["freq"], max_freq)
                    sev = d["severity"]
                    if sev >= 0.7:
                        sev_bg, sev_fg = "#FEE2E2", "#991B1B"
                    elif sev >= 0.4:
                        sev_bg, sev_fg = "#FEF9C3", "#854D0E"
                    else:
                        sev_bg, sev_fg = "#D1FAE5", "#065F46"
                    table_html += (
                        f'<tr style="background:{row_bg}; border-bottom:1px solid #E2E8F0;">'
                        f'<td style="padding:9px 16px; color:#1E293B;">{d["issue"]}</td>'
                        f'<td style="padding:9px 16px; text-align:center; font-weight:600; color:#334155;">{d["count"]}</td>'
                        f'<td style="padding:9px 16px; text-align:center; font-weight:700; color:{freq_fg};">{d["freq"]}%</td>'
                        f'<td style="padding:9px 16px; text-align:center;">'
                        f'<span style="background:{sev_bg}; color:{sev_fg}; padding:2px 10px;'
                        f'border-radius:999px; font-weight:600; font-size:13px;">{sev:.2f}</span></td>'
                        f'</tr>'
                    )

                if no_issue_data:
                    d = no_issue_data[0]
                    table_html += (
                        f'<tr style="background:#F0FDF4; border-top:2px solid #BBF7D0;">'
                        f'<td style="padding:9px 16px; color:#15803D; font-weight:600;">✅ No Issues</td>'
                        f'<td style="padding:9px 16px; text-align:center; font-weight:600; color:#334155;">{d["count"]}</td>'
                        f'<td style="padding:9px 16px; text-align:center; font-weight:700; color:#065F46;">{d["freq"]}%</td>'
                        f'<td style="padding:9px 16px; text-align:center;">'
                        f'<span style="background:#D1FAE5; color:#065F46; padding:2px 10px;'
                        f'border-radius:999px; font-weight:600; font-size:13px;">0.00</span></td>'
                        f'</tr>'
                    )

                table_html += "</tbody></table>"
                ui.html(table_html)
            else:
                ui.html('<div style="text-align:center; padding:20px; color:#94A3B8;">All issues are generic ("Other Issues")</div>')

            render_divider()

    # ────────────────────────────────────────────────────────────────
    # TAB 2: Node Analysis
    # ────────────────────────────────────────────────────────────────
    def build_node_analysis_tab(state: DashboardState, selected_agent_ref=None):
        render_section_header("Node-Specific CLEAR Analysis", "Select an agent to view its evaluation results")

        # Check if full trace CLEAR results are available
        has_full_traj = bool(state.full_traj_clear_results)
        
        # Build agent options list - add "Full Traj" if available
        available_agents = list(state.agent_results.keys())
        agent_options = []
        if has_full_traj:
            agent_options.append("Full Trace")
        agent_options.extend(available_agents)
        
        if not agent_options:
            ui.html('<div style="text-align:center; padding:40px; color:#64748B;"><p>No CLEAR analysis results found.</p><p style="margin-top:10px; color:#94A3B8; font-size:14px;">The uploaded ZIP file contains trace data but no agent_results folder with CLEAR analysis ZIP files.</p></div>')
            return

        # Determine initial agent - "Full Traj" by default if available, otherwise use selected_agent_ref or first agent
        initial_agent = agent_options[0]  # Default to first option (Full Traj if available)
        if selected_agent_ref and selected_agent_ref.get("value") and selected_agent_ref["value"] in agent_options:
            initial_agent = selected_agent_ref["value"]
        
        # Agent selector
        agent_select = ui.select(
            agent_options, value=initial_agent, label="Select Agent"
        ).classes("w-full max-w-md mb-4").props("outlined dense")
        
        analysis_container = ui.column().classes("w-full gap-4")
        
        # Watch for changes from graph clicks
        def check_and_update_from_ref():
            if selected_agent_ref and selected_agent_ref.get("value"):
                new_agent = selected_agent_ref["value"]
                if new_agent in available_agents and agent_select.value != new_agent:
                    agent_select.value = new_agent
                    show_agent_analysis(new_agent)
        
        # Check on tab visibility
        ui.timer(0.5, check_and_update_from_ref)

        def show_agent_analysis(agent_name):
            analysis_container.clear()
            with analysis_container:
                # Load data based on selection
                df = None
                meta = {}
                
                # Handle "Full Traj" option
                if agent_name == "Full Trace":
                    if not state.full_traj_clear_results:
                        ui.notify("No full trace CLEAR results available", type="warning")
                        return
                    
                    try:
                        file_bytes = state.full_traj_clear_results["zip_bytes"]
                        zip_name = state.full_traj_clear_results.get("zip_name", "full_traj_clear_results.zip")
                        trajectory_df = state.metadata.get("trajectory_df")
                        df, meta = load_clear_data_from_bytes(file_bytes, zip_name, trajectory_df)
                        
                        if df.empty:
                            ui.label("No data loaded from full trace CLEAR results").classes("text-slate-500")
                            return
                        
                    except Exception as e:
                        ui.notify(f"Error loading full trace CLEAR results: {str(e)}", type="negative")
                        return
                
                elif agent_name not in state.agent_results:
                    ui.notify(f"No CLEAR analysis results for {agent_name}", type="warning")
                    return
                else:
                    try:
                        agent_data = state.agent_results[agent_name]
                        file_bytes = agent_data["zip_bytes"]
                        zip_name = agent_data.get("zip_name", f"{agent_name}.zip")
                        # Pass trajectory_df for joining if needed
                        trajectory_df = state.metadata.get("trajectory_df")
                        df, meta = load_clear_data_from_bytes(file_bytes, zip_name, trajectory_df)
                        
                        if df.empty:
                            ui.label("No data loaded from CLEAR results").classes("text-slate-500")
                            return
                    
                    except Exception as e:
                        ui.notify(f"Error loading agent results: {str(e)}", type="negative")
                        return

                # Common rendering logic for both Full Traj and agent-specific results
                try:
                    # Statistics header with gradient banner
                    if agent_name == "Full Trace":
                        render_section_header("Full Trace CLEAR Analysis", "Overall evaluation results across all traces")
                    else:
                        render_section_header(f"{agent_name} Statistics", "CLEAR evaluation results for this specific agent")
                    
                    # Calculate metrics
                    total_evals = len(df)
                    avg_score = f"{df['score'].mean():.2f}" if "score" in df.columns else "N/A"
                    # Count unique issues (excluding NO_ISSUE and OTHER)
                    unique_issues_count = 0
                    if "recurring_issues_str" in df.columns:
                        all_unique_issues = set()
                        for issues_str in df["recurring_issues_str"].dropna():
                            issues = extract_issues(issues_str)
                            for issue in issues:
                                if issue not in [NO_ISSUE, OTHER]:
                                    all_unique_issues.add(issue)
                        unique_issues_count = len(all_unique_issues)
                    unique_tasks = df["task_id"].nunique() if "task_id" in df.columns else 0
                    
                    # Gradient banner for metrics
                    ui.html(f'''
                        <div style="
                            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                            padding: 24px;
                            border-radius: 12px;
                            margin-bottom: 24px;
                            box-shadow: 0 10px 25px rgba(240, 147, 251, 0.3);
                        ">
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;">
                                <div style="text-align: center; color: white;">
                                    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📊 Evaluations</div>
                                    <div style="font-size: 32px; font-weight: 700;">{total_evals}</div>
                                </div>
                                <div style="text-align: center; color: white;">
                                    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">⭐ Avg Score</div>
                                    <div style="font-size: 32px; font-weight: 700;">{avg_score}</div>
                                </div>
                                <div style="text-align: center; color: white;">
                                    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">⚠️ Unique Issues</div>
                                    <div style="font-size: 32px; font-weight: 700;">{unique_issues_count}</div>
                                </div>
                                <div style="text-align: center; color: white;">
                                    <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">🎯 Tasks</div>
                                    <div style="font-size: 32px; font-weight: 700;">{unique_tasks}</div>
                                </div>
                            </div>
                        </div>
                    ''')

                    render_divider()

                    # Score Distribution
                    if "score" in df.columns and not df["score"].empty:
                        with ui.expansion("Score Distribution", icon="bar_chart").classes("w-full"):
                            render_section_header("Score Distribution", "Distribution of evaluation scores for this agent")
                            fig = px.histogram(
                                df, x="score", nbins=20,
                                color_discrete_sequence=[COLORS["primary"]],
                                labels={"score": "Score", "count": "Count"},
                            )
                            fig.update_layout(showlegend=False, height=300, title="")
                            ui.plotly(fig).classes("w-full")

                    render_divider()

                    # Issues Frequency (shown before filters)
                    all_issues_list = []
                    if "recurring_issues_str" in df.columns:
                        render_section_header("Discovered Issues", "Recurring problems identified in this agent's outputs")

                        for issues_str in df["recurring_issues_str"].dropna():
                            all_issues_list.extend(extract_issues(issues_str))

                        if all_issues_list:
                            issue_counts = Counter(all_issues_list)
                            issues_data = []
                            for issue, count in issue_counts.most_common():
                                issue_rows = df[df["recurring_issues_str"].apply(
                                    lambda x: issue in extract_issues(x) if pd.notna(x) else False
                                )]
                                mean_score = issue_rows["score"].mean() if "score" in issue_rows.columns and not issue_rows["score"].empty else 0.5
                                freq = round(100 * count / len(df), 1)
                                severity = round(1 - mean_score, 2)
                                issues_data.append({
                                    "issue": issue, "count": count,
                                    "freq": freq, "severity": severity,
                                    "mean_score": mean_score,
                                })

                            # Build HTML table
                            # Separate NO_ISSUE from other issues
                            display_issues = [d for d in issues_data if d["issue"] not in [NO_ISSUE, OTHER]]
                            no_issue_data = [d for d in issues_data if d["issue"] == NO_ISSUE]
                            
                            if display_issues or no_issue_data:
                                max_freq = max(d["freq"] for d in display_issues) if display_issues else 1

                                table_html = (
                                    '<table class="issues-table" style="'
                                    'width:100%; border-collapse:collapse; font-size:14px;'
                                    'box-shadow:0 2px 8px rgba(0,0,0,0.07); border-radius:10px; overflow:hidden;">'
                                    '<thead><tr style="background:#4F46E5; color:#fff;'
                                    'font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">'
                                    '<th style="padding:11px 16px; text-align:left; font-weight:600;">Issue</th>'
                                    '<th style="padding:11px 16px; text-align:center; font-weight:600;">Count</th>'
                                    '<th style="padding:11px 16px; text-align:center; font-weight:600;">Frequency</th>'
                                    '<th style="padding:11px 16px; text-align:center; font-weight:600;">Severity</th>'
                                    '</tr></thead><tbody>'
                                )

                                # Add regular issues
                                for i, d in enumerate(display_issues):
                                    row_bg = "#FFFFFF" if i % 2 == 0 else "#F8FAFC"
                                    t = min(d["freq"] / max_freq, 1.0) if max_freq > 0 else 0
                                    if t >= 0.66:
                                        freq_fg = "#DC2626"   # red (high freq)
                                    elif t >= 0.33:
                                        freq_fg = "#D97706"   # amber
                                    else:
                                        freq_fg = "#16A34A"   # green (low freq)
                                    sev = d["severity"]
                                    if sev >= 0.7:
                                        sev_bg, sev_fg = "#FEE2E2", "#991B1B"
                                    elif sev >= 0.4:
                                        sev_bg, sev_fg = "#FEF9C3", "#854D0E"
                                    else:
                                        sev_bg, sev_fg = "#D1FAE5", "#065F46"
                                    table_html += (
                                        f'<tr style="background:{row_bg}; border-bottom:1px solid #E2E8F0;">'
                                        f'<td style="padding:9px 16px; color:#1E293B;">{d["issue"]}</td>'
                                        f'<td style="padding:9px 16px; text-align:center; font-weight:600; color:#334155;">{d["count"]}</td>'
                                        f'<td style="padding:9px 16px; text-align:center; font-weight:700; color:{freq_fg};">{d["freq"]}%</td>'
                                        f'<td style="padding:9px 16px; text-align:center;">'
                                        f'<span style="background:{sev_bg}; color:{sev_fg}; padding:2px 10px;'
                                        f'border-radius:999px; font-weight:600; font-size:13px;">{sev:.2f}</span></td>'
                                        f'</tr>'
                                    )

                                # Add "No Issues" row at the end
                                if no_issue_data:
                                    d = no_issue_data[0]
                                    table_html += (
                                        f'<tr style="background:#F0FDF4; border-top:2px solid #BBF7D0;">'
                                        f'<td style="padding:9px 16px; color:#15803D; font-weight:600;">✅ No Issues</td>'
                                        f'<td style="padding:9px 16px; text-align:center; font-weight:600; color:#334155;">{d["count"]}</td>'
                                        f'<td style="padding:9px 16px; text-align:center; font-weight:700; color:#065F46;">{d["freq"]}%</td>'
                                        f'<td style="padding:9px 16px; text-align:center;">'
                                        f'<span style="background:#D1FAE5; color:#065F46; padding:2px 10px;'
                                        f'border-radius:999px; font-weight:600; font-size:13px;">0.00</span></td>'
                                        f'</tr>'
                                    )

                                table_html += "</tbody></table>"
                                ui.html(table_html)
                            else:
                                ui.html('<div style="text-align:center; padding:20px; color:#94A3B8;">All issues are generic ("Other Issues")</div>')
                        else:
                            ui.html('<div style="text-align:center; padding:20px; color:#10B981;">No issues discovered in this agent\'s outputs</div>')

                    render_divider()

                    # Filters Section - moved after issues list
                    render_section_header("🔍 Filter Data", "Filter evaluation records by issues and score")
                    
                    # Get all unique issues for filter options
                    all_issues_for_filter = []
                    if all_issues_list:
                        issue_counts_temp = Counter(all_issues_list)
                        all_issues_for_filter = [issue for issue in issue_counts_temp.keys() if issue not in [NO_ISSUE, OTHER]]
                    
                    with ui.column().classes("w-full gap-6"):
                        ui.html('<div style="color:#475569; font-size:15px; margin-bottom:8px; font-weight:500;">Select issues using AND/OR/NOT logic, then click Apply Filter</div>')
                        
                        with ui.row().classes("w-full gap-6"):
                            with ui.column().classes("flex-1 gap-3"):
                                ui.label("Include ANY of (OR)").classes("text-sm font-bold text-slate-700").style("margin-bottom: 8px;")
                                include_filter = ui.select(
                                    all_issues_for_filter, multiple=True, label="At least one of these"
                                ).classes("w-full").props("outlined use-chips")
                            
                            with ui.column().classes("flex-1 gap-3"):
                                ui.label("Must ALSO have (AND)").classes("text-sm font-bold text-slate-700").style("margin-bottom: 8px;")
                                must_have_filter = ui.select(
                                    all_issues_for_filter, multiple=True, label="All of these"
                                ).classes("w-full").props("outlined use-chips")
                            
                            with ui.column().classes("flex-1 gap-3"):
                                ui.label("Exclude ANY of (NOT)").classes("text-sm font-bold text-slate-700").style("margin-bottom: 8px;")
                                exclude_filter = ui.select(
                                    all_issues_for_filter, multiple=True, label="None of these"
                                ).classes("w-full").props("outlined use-chips")
                        
                        only_checkbox = ui.checkbox("Only selected issues (no other issues)", value=False).classes("mt-4")
                        
                        ui.label("Score Range").classes("text-sm font-bold text-slate-700 mt-6").style("margin-bottom: 12px;")
                        if "score" in df.columns:
                            score_filter = ui.range(
                                min=0.0, max=1.0, step=0.05, value={"min": 0.0, "max": 1.0}
                            ).classes("w-full").props("label-always color=deep-purple-6")
                        else:
                            score_filter = None
                            ui.label("Score data not available").classes("text-slate-400 italic text-sm")
                        
                        with ui.row().classes("gap-3 mt-6"):
                            apply_filter_btn = ui.button(
                                "✅ Apply Filter",
                                icon="filter_alt"
                            ).props("color=positive no-caps").style("padding: 8px 20px;")
                            
                            clear_filter_btn = ui.button(
                                "🧹 Clear Filters",
                                icon="clear"
                            ).props("color=grey no-caps").style("padding: 8px 20px;")

                    render_divider()

                    # Issue Distribution Comparison Container
                    distribution_container = ui.column().classes("w-full gap-4")
                    
                    # Data Explorer Container
                    data_explorer_container = ui.column().classes("w-full gap-4")
                    
                    # Store filtered data
                    filtered_data_ref = {"df": df}
                    
                    def apply_issue_filters():
                        """Apply AND/OR/NOT logic to filter by issues"""
                        include = include_filter.value or []
                        must_have = must_have_filter.value or []
                        exclude = exclude_filter.value or []
                        only = only_checkbox.value
                        score_range = score_filter.value if score_filter else {"min": 0.0, "max": 1.0}
                        
                        def issue_filter_func(text_issues_str):
                            issues = extract_issues(text_issues_str)
                            
                            # OR logic
                            if include:
                                if not any(i in issues for i in include):
                                    return False
                            
                            # AND logic
                            if must_have:
                                if not all(i in issues for i in must_have):
                                    return False
                            
                            # NOT logic
                            if exclude:
                                if any(i in issues for i in exclude):
                                    return False
                            
                            # "Only" logic
                            if only:
                                allowed = set(include + must_have)
                                if any(i not in allowed for i in issues if i not in [NO_ISSUE, OTHER]):
                                    return False
                            
                            return True
                        
                        # Apply issue filters
                        if "recurring_issues_str" in df.columns:
                            filtered_df = df[df["recurring_issues_str"].apply(issue_filter_func)]
                        else:
                            filtered_df = df.copy()
                        
                        # Apply score filter
                        if "score" in filtered_df.columns:
                            filtered_df = filtered_df[
                                (filtered_df["score"] >= score_range["min"]) &
                                (filtered_df["score"] <= score_range["max"])
                            ]
                        
                        return filtered_df
                    
                    def update_visualizations():
                        distribution_container.clear()
                        with distribution_container:
                            filtered_df = apply_issue_filters()
                            filtered_data_ref["df"] = filtered_df
                            
                            render_section_header("Issue Distribution Comparison", "Comparing full dataset vs filtered subset")
                            
                            # Get issue frequencies
                            full_issues = []
                            for issues_str in df["recurring_issues_str"].dropna():
                                full_issues.extend(extract_issues(issues_str))
                            full_issue_counts = Counter(full_issues)
                            
                            filtered_issues = []
                            for issues_str in filtered_df["recurring_issues_str"].dropna():
                                filtered_issues.extend(extract_issues(issues_str))
                            filtered_issue_counts = Counter(filtered_issues)
                            
                            # Remove NO_ISSUE and OTHER
                            for issue_type in [NO_ISSUE, OTHER]:
                                full_issue_counts.pop(issue_type, None)
                                filtered_issue_counts.pop(issue_type, None)
                            
                            if not full_issue_counts:
                                ui.html('<div style="text-align:center; padding:20px; color:#94A3B8;">No issues to display</div>')
                            else:
                                # Create comparison data — sort ascending so top issue appears at top of chart
                                all_issue_names = sorted(full_issue_counts.keys(), key=lambda x: full_issue_counts[x], reverse=False)[-20:]
                                full_counts = [full_issue_counts.get(issue, 0) for issue in all_issue_names]
                                filtered_counts = [filtered_issue_counts.get(issue, 0) for issue in all_issue_names]

                                # Wrap long labels for the y-axis (max ~40 chars per line)
                                def wrap_label(text, max_chars=40):
                                    if len(text) <= max_chars:
                                        return text
                                    words = text.split()
                                    lines, line = [], []
                                    for word in words:
                                        if sum(len(w) for w in line) + len(line) + len(word) > max_chars and line:
                                            lines.append(" ".join(line))
                                            line = [word]
                                        else:
                                            line.append(word)
                                    if line:
                                        lines.append(" ".join(line))
                                    return "<br>".join(lines)

                                wrapped_labels = [wrap_label(name) for name in all_issue_names]

                                # Horizontal grouped bar chart
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    name="Full Dataset",
                                    y=wrapped_labels,
                                    x=full_counts,
                                    orientation="h",
                                    customdata=all_issue_names,
                                    hovertemplate="<b>%{customdata}</b><br>Full Dataset: %{x}<extra></extra>",
                                    marker=dict(color=COLORS["primary"], opacity=0.85),
                                    text=full_counts,
                                    textposition="outside",
                                    textangle=0,
                                    textfont=dict(size=12, color=COLORS["text"]),
                                    cliponaxis=False,
                                ))
                                fig.add_trace(go.Bar(
                                    name="Filtered Subset",
                                    y=wrapped_labels,
                                    x=filtered_counts,
                                    orientation="h",
                                    customdata=all_issue_names,
                                    hovertemplate="<b>%{customdata}</b><br>Filtered Subset: %{x}<extra></extra>",
                                    marker=dict(color=COLORS["secondary"], opacity=0.85),
                                    text=filtered_counts,
                                    textposition="outside",
                                    textangle=0,
                                    textfont=dict(size=12, color=COLORS["text"]),
                                    cliponaxis=False,
                                ))

                                n_issues = len(all_issue_names)
                                bar_height = 34  # px per bar pair
                                chart_height = max(400, n_issues * bar_height * 2 + 100)
                                max_count = max(full_counts + [1])

                                fig.update_layout(
                                    barmode="group",
                                    height=chart_height,
                                    xaxis_title="Count",
                                    yaxis_title="",
                                    showlegend=True,
                                    legend=dict(
                                        orientation="h",
                                        x=0.5, xanchor="center",
                                        y=1.02, yanchor="bottom",
                                        font=dict(size=12),
                                    ),
                                    margin=dict(l=340, r=80, t=50, b=60),
                                    plot_bgcolor="white",
                                    paper_bgcolor="white",
                                    xaxis=dict(
                                        gridcolor="#F1F5F9",
                                        zerolinecolor="#E2E8F0",
                                        tickfont=dict(size=12),
                                        range=[0, max_count * 1.15],
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(size=12, color=COLORS["text"]),
                                        automargin=True,
                                        ticklabelposition="outside left",
                                        ticklabeloverflow="allow",
                                    ),
                                    bargap=0.3,
                                    bargroupgap=0.05,
                                    uniformtext=dict(minsize=10, mode="hide"),
                                )
                                ui.plotly(fig).classes("w-full")
                            
                            ui.html(f'<div style="color:#64748B; font-size:14px; margin-top:10px;">Showing <strong>{len(filtered_df)}</strong> of {len(df)} records after filtering</div>')
                    
                    def update_data_explorer():
                        data_explorer_container.clear()
                        with data_explorer_container:
                            render_section_header("Data Explorer", "Browse individual evaluation records (filtered)")
                            
                            filtered_df = filtered_data_ref["df"]
                            if not filtered_df.empty:
                                display_cols = [c for c in ["intent", "model_input_preview", "response", "score", "evaluation_summary", "recurring_issues_str"] if c in filtered_df.columns and filtered_df[c].dropna().astype(str).str.strip().ne("").any()]
                                display_df = filtered_df[display_cols].head(100).copy()

                                if "score" in display_df.columns:
                                    display_df["score"] = display_df["score"].round(2)
                                for col in display_df.select_dtypes(include=["object"]).columns:
                                    display_df[col] = display_df[col].apply(
                                        lambda x: str(x)[:200] + "..." if isinstance(x, str) and len(str(x)) > 200 else x
                                    )

                                rows = display_df.reset_index().to_dict("records")
                                columns = [{"name": c, "label": c.replace("_", " ").title(), "field": c, "sortable": True, "align": "left"} for c in display_df.reset_index().columns]

                                table = ui.table(
                                    columns=columns, rows=rows, row_key="question_id",
                                    pagination={"rowsPerPage": 15, "sortBy": "score"},
                                ).classes("w-full").props("flat bordered dense")

                                detail_container = ui.column().classes("w-full gap-2")

                                def on_row_click(e):
                                    row_data = e.args[1]
                                    detail_container.clear()
                                    with detail_container:
                                        render_divider()
                                        render_section_header("Record Details")
                                        idx = row_data.get("question_id", row_data.get("index", ""))
                                        if idx in filtered_df.index:
                                            orig_row = filtered_df.loc[idx]
                                        else:
                                            ui.label("Could not find original record").classes("text-slate-500")
                                            return

                                        with ui.card().classes("w-full custom-card"):
                                            ui.label(f"Record: {idx}").classes("text-lg font-semibold text-slate-700")

                                            _exclude_detail = {"Name", "step_in_trace_node", "id", "agent_or_tool"}
                                            input_columns = get_input_columns(meta)
                                            for column in input_columns:
                                                if column in _exclude_detail or column not in orig_row:
                                                    continue
                                                val = orig_row[column]
                                                if val is not None and not (isinstance(val, float) and pd.isna(val)) and str(val).strip():
                                                    with ui.expansion(column.replace("_", " ").title()).classes("w-full"):
                                                        ui.html(f'<pre style="white-space:pre-wrap; font-size:12px; color:#334155; max-height:300px; overflow-y:auto;">{str(val)}</pre>')

                                            with ui.expansion("Model Input (Prompt)").classes("w-full"):
                                                ui.html(f'<pre style="white-space:pre-wrap; font-size:12px; color:#334155; max-height:300px; overflow-y:auto;">{str(orig_row.get("model_input", "N/A"))}</pre>')

                                            with ui.expansion("Response").classes("w-full"):
                                                ui.html(f'<pre style="white-space:pre-wrap; font-size:12px; color:#334155; max-height:300px; overflow-y:auto;">{str(orig_row.get("response", "N/A"))}</pre>')

                                            with ui.expansion("Full Evaluation Text").classes("w-full"):
                                                ui.html(f'<pre style="white-space:pre-wrap; font-size:12px; color:#334155; max-height:300px; overflow-y:auto;">{str(orig_row.get("evaluation_text", "N/A"))}</pre>')

                                            eval_summary = orig_row.get("evaluation_summary")
                                            if eval_summary and not pd.isna(eval_summary):
                                                ui.label("Evaluation Summary").classes("font-semibold text-slate-600 mt-2")
                                                ui.label(str(eval_summary)).classes("text-sm text-slate-500")

                                            score_val = orig_row.get("score", "N/A")
                                            ui.html(f'<div class="mt-2"><strong>Score:</strong> {score_badge_html(float(score_val)) if isinstance(score_val, (int, float)) else score_val}</div>')

                                            recurring = orig_row.get("recurring_issues_str")
                                            if recurring and not pd.isna(recurring):
                                                issues = extract_issues(recurring)
                                                ui.label("Recurring Issues").classes("font-semibold text-slate-600 mt-2")
                                                for iss in issues:
                                                    ui.html(f'<span class="agent-badge" style="margin:2px;">{iss}</span>')

                                table.on("rowClick", on_row_click)
                            else:
                                ui.html('<div style="text-align:center; padding:20px; color:#F59E0B;">No records match the selected filters</div>')
                    
                    def on_apply_filters():
                        update_visualizations()
                        update_data_explorer()
                        ui.notify(f"Filters applied: {len(filtered_data_ref['df'])} records match", type="positive")
                    
                    def on_clear_filters():
                        include_filter.value = []
                        must_have_filter.value = []
                        exclude_filter.value = []
                        only_checkbox.value = False
                        if score_filter:
                            score_filter.value = {"min": 0.0, "max": 1.0}
                        filtered_data_ref["df"] = df
                        update_visualizations()
                        update_data_explorer()
                        ui.notify("Filters cleared", type="info")
                    
                    apply_filter_btn.on_click(on_apply_filters)
                    clear_filter_btn.on_click(on_clear_filters)
                    
                    # Initial render
                    update_visualizations()
                    update_data_explorer()

                except Exception as ex:
                    ui.label(f"Error loading CLEAR analysis: {ex}").classes("text-red-500")

        def on_select_change():
            show_agent_analysis(agent_select.value)
            # Clear the selected_agent_ref after use
            if selected_agent_ref:
                selected_agent_ref["value"] = None

        agent_select.on_value_change(on_select_change)
        show_agent_analysis(initial_agent)

    # ────────────────────────────────────────────────────────────────
    # Helper functions for trace evaluation display
    # ────────────────────────────────────────────────────────────────
    def render_standard_evaluation(traj_eval):
        """Render the standard dimension-based trace evaluation."""
        # Overall score and detailed feedback (open by default)
        overall_score = traj_eval.get("overall_score", 0)
        detailed_feedback = traj_eval.get("detailed_feedback", "No feedback available")
        
        with ui.card().classes("w-full").style("background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; margin-bottom: 16px;"):
            ui.html(f"""
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">Overall Score</div>
                        <div style="font-size: 48px; font-weight: 700;">{overall_score:.2f}</div>
                    </div>
                    <div style="font-size: 64px; opacity: 0.3;">🎯</div>
                </div>
            """)
        
        # Detailed feedback (open by default)
        with ui.expansion("💬 Detailed Feedback", icon="feedback", value=True).classes("w-full").style("background: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 16px;"):
            ui.html(f"""
                <div style="padding: 12px; background: white; border-radius: 8px; line-height: 1.6; color: #334155;">
                    {detailed_feedback}
                </div>
            """)
        
        # Dimension scores (expandable)
        dimension_scores = traj_eval.get("dimension_scores", {})
        parsed_eval = traj_eval.get("parsed_evaluation", {})
        
        if dimension_scores:
            with ui.expansion("📈 Evaluation Criteria Scores", icon="analytics", value=False).classes("w-full").style("background: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 16px;"):
                # Separate step quality and trace dimensions
                step_quality = parsed_eval.get("step_quality_dimensions", {})
                trajectory_dims = parsed_eval.get("trajectory_dimensions", {})
                
                if step_quality:
                    ui.html('<div style="font-weight: 600; color: #1E293B; margin: 12px 0 8px 0; font-size: 15px;">Step Quality Dimensions</div>')
                    for criterion, details in step_quality.items():
                        score = float(details.get("score", 0))
                        justification = details.get("justification", "No justification provided")
                        score_color = score_to_hex(score)
                        definition = get_dimension_definition(criterion)
                        
                        with ui.row().classes("w-full items-center gap-2 mb-2"):
                            with ui.expansion(f"{criterion}: {score:.2f}", icon="check_circle").classes("flex-1").style(f"border-left: 4px solid {score_color};"):
                                ui.html(f"""
                                    <div style="padding: 8px; background: #FAFAFA; border-radius: 6px; color: #475569; font-size: 13px; line-height: 1.5;">
                                        {justification}
                                    </div>
                                """)
                            with ui.button(icon="info", on_click=lambda: None).props("flat dense round size=sm").classes("text-blue-500"):
                                ui.tooltip(definition).classes("bg-slate-800 text-white text-sm max-w-md")
                
                if trajectory_dims:
                    ui.html('<div style="font-weight: 600; color: #1E293B; margin: 20px 0 8px 0; font-size: 15px;">Trace-Level Dimensions</div>')
                    for criterion, details in trajectory_dims.items():
                        score = float(details.get("score", 0))
                        justification = details.get("justification", "No justification provided")
                        score_color = score_to_hex(score)
                        definition = get_dimension_definition(criterion)
                        
                        with ui.row().classes("w-full items-center gap-2 mb-2"):
                            with ui.expansion(f"{criterion}: {score:.2f}", icon="check_circle").classes("flex-1").style(f"border-left: 4px solid {score_color};"):
                                ui.html(f"""
                                    <div style="padding: 8px; background: #FAFAFA; border-radius: 6px; color: #475569; font-size: 13px; line-height: 1.5;">
                                        {justification}
                                    </div>
                                """)
                            with ui.button(icon="info", on_click=lambda: None).props("flat dense round size=sm").classes("text-blue-500"):
                                ui.tooltip(definition).classes("bg-slate-800 text-white text-sm max-w-md")

    def render_rubric_evaluation(rubric_eval):
        """Render the rubric-based trace evaluation."""
        # Overall score card
        score = rubric_eval.get("score", 0)
        fulfilled_count = rubric_eval.get("fulfilled_count", 0)
        num_rubrics = rubric_eval.get("num_rubrics", 0)
        summary = rubric_eval.get("summary", "No summary available")
        
        with ui.card().classes("w-full").style("background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; margin-bottom: 16px;"):
            ui.html(f"""
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">Rubric Score</div>
                        <div style="font-size: 48px; font-weight: 700;">{score:.2f}</div>
                        <div style="font-size: 14px; opacity: 0.9; margin-top: 8px;">
                            {fulfilled_count} of {num_rubrics} rubrics fulfilled
                        </div>
                    </div>
                    <div style="font-size: 64px; opacity: 0.3;">📋</div>
                </div>
            """)
        
        # Summary (open by default)
        with ui.expansion("📝 Evaluation Summary", icon="summarize", value=True).classes("w-full").style("background: #F8FAFC; border: 1px solid #E2E8F0; margin-bottom: 16px;"):
            ui.html(f"""
                <div style="padding: 12px; background: white; border-radius: 8px; line-height: 1.6; color: #334155;">
                    {summary}
                </div>
            """)
        
        # Individual rubrics
        rubrics = rubric_eval.get("rubrics", [])
        rubric_results = rubric_eval.get("rubric_results", {})
        
        if rubrics:
            with ui.expansion("📊 Individual Rubric Results", icon="checklist", value=True).classes("w-full").style("background: #F8FAFC; border: 1px solid #E2E8F0;"):
                for rubric in rubrics:
                    rubric_id = rubric.get("id", "")
                    description = rubric.get("description", "No description")
                    criterion = rubric.get("criterion", "No criterion")
                    
                    # Get the result for this rubric
                    result = rubric_results.get(rubric_id, {})
                    fulfilled = result.get("fulfilled", 0)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    
                    # Color based on fulfillment
                    if fulfilled == 1:
                        border_color = "#10B981"  # Green
                        icon = "check_circle"
                        status_text = "✓ Fulfilled"
                        status_color = "#10B981"
                    else:
                        border_color = "#EF4444"  # Red
                        icon = "cancel"
                        status_text = "✗ Not Fulfilled"
                        status_color = "#EF4444"
                    
                    with ui.card().classes("w-full mb-3").style(f"border-left: 4px solid {border_color}; background: white;"):
                        # Rubric header
                        ui.html(f"""
                            <div style="padding: 12px; border-bottom: 1px solid #E2E8F0;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <span style="font-weight: 600; color: #1E293B; font-size: 15px;">{rubric_id}</span>
                                    <span style="color: {status_color}; font-weight: 600; font-size: 14px;">{status_text}</span>
                                </div>
                                <div style="color: #475569; font-size: 14px; line-height: 1.5; margin-bottom: 8px;">
                                    <strong>Description:</strong> {description}
                                </div>
                                <div style="color: #64748B; font-size: 13px; line-height: 1.5; font-style: italic;">
                                    <strong>Criterion:</strong> {criterion}
                                </div>
                            </div>
                        """)
                        
                        # Reasoning (expandable)
                        with ui.expansion("💭 Evaluation Reasoning", icon="psychology", value=False).classes("w-full"):
                            ui.html(f"""
                                <div style="padding: 8px; background: #FAFAFA; border-radius: 6px; color: #475569; font-size: 13px; line-height: 1.5;">
                                    {reasoning}
                                </div>
                            """)

    # ────────────────────────────────────────────────────────────────
    # TAB 3: Trace Explorer
    # ────────────────────────────────────────────────────────────────
    def build_trajectory_tab(state: DashboardState):
        render_section_header("Trace Explorer", "Browse and search individual agent traces")

        available_tasks = sorted(state.traj_df["task_id"].unique())
        all_agents = sorted(state.traj_df["Name"].unique())

        # Search and Filters in a card with better styling
        with ui.card().classes("w-full").style("background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; margin-bottom: 20px;"):
            ui.html('<div style="font-size: 18px; font-weight: 600; color: #1e293b; margin-bottom: 16px;">🔍 Search & Filter Traces</div>')
            
            # Search bar - prominent position
            with ui.row().classes("w-full gap-4 mb-4"):
                search_input = ui.input(
                    label="Search by Task ID or Intent",
                    placeholder="Start typing to search..."
                ).classes("flex-grow").props("outlined dense clearable bg-color=white")
                search_input.props('prepend-inner-icon="search"')
                
                ui.button("Clear All Filters", icon="clear_all", on_click=lambda: reset_filters()).props("outline color=grey-7 no-caps").classes("self-center")
            
            # Advanced Filters in expansion
            with ui.expansion("🎛️ Advanced Filters", icon="tune", value=False).classes("w-full").style("background: white; border-radius: 8px;"):
                all_lengths = []
                for task_id in available_tasks:
                    task_data = state.traj_df[state.traj_df["task_id"] == task_id]
                    all_lengths.append(len(task_data))
                min_len, max_len = min(all_lengths) if all_lengths else 1, max(all_lengths) if all_lengths else 10

                with ui.grid(columns=3).classes("w-full gap-4 p-4"):
                    # Length filter
                    with ui.column().classes("gap-2"):
                        ui.html('<div style="font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 8px;">📏 Trace Length</div>')
                        length_slider = ui.range(
                            min=min_len, max=max_len, value={"min": min_len, "max": max_len}
                        ).props("label-always snap color=cyan-6").classes("w-full")
                        ui.html(f'<div style="font-size: 11px; color: #64748b; margin-top: 4px;">Range: {min_len} - {max_len} steps</div>')

                    # Agent filter
                    with ui.column().classes("gap-2"):
                        ui.html('<div style="font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 8px;">👥 Contains Agents</div>')
                        agent_filter = ui.select(
                            all_agents, with_input=True, multiple=True, label="Select agents"
                        ).classes("w-full").props("outlined dense use-chips color=cyan-6")
                        ui.html(f'<div style="font-size: 11px; color: #64748b; margin-top: 4px;">{len(all_agents)} agents available</div>')

                    # Score filter
                    with ui.column().classes("gap-2"):
                        ui.html('<div style="font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 8px;">⭐ Score Range (Standard Evaluation)</div>')
                        has_traj_eval_scores = bool(state.metadata.get("traj_eval_results"))
                        has_traj_score = "traj_score" in state.traj_df.columns and state.traj_df["traj_score"].notna().any()
                        if has_traj_eval_scores or has_traj_score or state.all_agent_scores:
                            score_slider = ui.range(
                                min=0.0, max=1.0, step=0.05, value={"min": 0.0, "max": 1.0}
                            ).props("label-always snap color=cyan-6").classes("w-full")
                            ui.html('<div style="font-size: 11px; color: #64748b; margin-top: 4px;">0.0 - 1.0 (overall_score from Standard Evaluation)</div>')
                        else:
                            score_slider = None
                            ui.html('<div style="font-size: 12px; color: #94a3b8; font-style: italic; padding: 12px; background: #f1f5f9; border-radius: 6px;">Score data not available</div>')
                
                # Apply button
                with ui.row().classes("w-full justify-end mt-4"):
                    ui.button("Apply Filters", on_click=lambda: refresh_list(), icon="filter_alt").props("color=cyan-6 no-caps unelevated").classes("px-6")
        
        def reset_filters():
            """Reset all filters to default values"""
            search_input.value = ""
            length_slider.value = {"min": min_len, "max": max_len}
            agent_filter.value = []
            if score_slider:
                score_slider.value = {"min": 0.0, "max": 1.0}
            refresh_list()

        # Trace list container
        traj_list_container = ui.column().classes("w-full gap-4 mt-4")
        detail_container = ui.column().classes("w-full gap-2")

        def get_filtered_tasks():
            filtered = []
            length_range = length_slider.value
            selected_agents = agent_filter.value or []
            score_range = score_slider.value if score_slider else None
            search_term = search_input.value or ""

            # Build a task_id -> overall_score lookup from traj_eval_results
            # This matches the score shown in the Standard Evaluation tab
            traj_eval_results = state.metadata.get("traj_eval_results", {})
            traj_eval_score_lookup = {}
            for traj_name, eval_data in traj_eval_results.items():
                overall_score = eval_data.get("overall_score")
                if overall_score is not None:
                    # traj_name may be "task_id" or "task_id_variant"
                    traj_eval_score_lookup[traj_name] = float(overall_score)

            for task_id in available_tasks:
                task_data = state.traj_df[state.traj_df["task_id"] == task_id]
                task_length = len(task_data)

                if task_length < length_range["min"] or task_length > length_range["max"]:
                    continue
                if selected_agents:
                    task_agents = set(task_data["Name"].unique())
                    if not any(a in task_agents for a in selected_agents):
                        continue
                if score_range:
                    # Use overall_score from traj_eval_results — same score shown in Standard Evaluation tab
                    task_score = traj_eval_score_lookup.get(str(task_id))
                    if task_score is None:
                        # traj_name may have a suffix like "task_id_variant", find by prefix
                        for traj_name, score in traj_eval_score_lookup.items():
                            if traj_name.startswith(str(task_id)):
                                task_score = score
                                break
                    # Fallback to traj_score from traj_df
                    if task_score is None and "traj_score" in state.traj_df.columns:
                        task_rows = state.traj_df[state.traj_df["task_id"] == task_id]
                        if not task_rows.empty:
                            val = task_rows["traj_score"].iloc[0]
                            if pd.notna(val):
                                task_score = float(val)
                    if task_score is not None:
                        if task_score < score_range["min"] or task_score > score_range["max"]:
                            continue
                if search_term:
                    if search_term.lower() not in str(task_id).lower():
                        if "intent" in task_data.columns:
                            intent_val = task_data["intent"].iloc[0] if not task_data.empty else ""
                            if not intent_val or search_term.lower() not in str(intent_val).lower():
                                continue
                        else:
                            continue

                filtered.append(task_id)
            return filtered

        def show_trajectory_details(task_id):
            detail_container.clear()
            with detail_container:
                task_data = state.traj_df[state.traj_df["task_id"] == task_id].sort_values("step_in_trace_general")
                if task_data.empty:
                    ui.label("No data found for this trace").classes("text-slate-500")
                    return

                render_divider()

                # Gather scores first (needed for metrics)
                trajectory_scores = []
                scores_by_step = {}
                for idx, row in task_data.iterrows():
                    agent_name = row["Name"]
                    step_num = row.get("step_in_trace_general", idx)
                    task_id_val = str(row.get("task_id", ""))
                    score_val = None
                    eval_summary_val = None

                    # Construct the composite id used in CLEAR results: "{task_id}_{step_in_trace_general}"
                    composite_id = f"{task_id_val}_{step_num}" if task_id_val else ""

                    if agent_name in state.all_agent_scores_df:
                        df_agent = state.all_agent_scores_df[agent_name]

                        match = pd.DataFrame()
                        # Strategy 1: match by composite id column
                        if "id" in df_agent.columns and composite_id:
                            match = df_agent[df_agent["id"] == composite_id]
                        # Strategy 2: match by task_id + step_in_trace_general columns
                        if match.empty and "task_id" in df_agent.columns:
                            step_col = "step_in_trace_general" if "step_in_trace_general" in df_agent.columns else "step_in_trace"
                            if step_col in df_agent.columns and task_id_val:
                                match = df_agent[(df_agent["task_id"].astype(str) == task_id_val) & (df_agent[step_col] == step_num)]
                        # Strategy 3: match by index (question_id) if it equals composite_id
                        if match.empty and composite_id and composite_id in df_agent.index:
                            match = df_agent.loc[[composite_id]]

                        if not match.empty:
                            if "score" in match.columns:
                                s = match.iloc[0]["score"]
                                if pd.notna(s):
                                    try:
                                        score_val = float(s)
                                    except (ValueError, TypeError):
                                        pass
                            if "evaluation_summary" in match.columns:
                                es = match.iloc[0]["evaluation_summary"]
                                if pd.notna(es) and str(es).strip():
                                    eval_summary_val = str(es).strip()

                    if score_val is not None:
                        trajectory_scores.append(score_val)
                        scores_by_step[step_num] = {"agent": agent_name, "score": score_val, "eval_summary": eval_summary_val}

                # Intent - Display at the top
                intent = task_data.iloc[0].get("intent", "N/A")
                if intent and intent != "N/A":
                    ui.html(f"""
                        <div class="intent-card" style="margin: 12px 0;">
                            <strong style="color:#4338CA;">User Intent:</strong>
                            <span style="color:#334155;">{intent}</span>
                        </div>
                    """)

                # Metrics - Display at the top with gradient banner
                unique_agents = task_data["Name"].nunique()
                steps_count = len(task_data)
                min_score = f"{min(trajectory_scores):.2f}" if trajectory_scores else "N/A"
                avg_score = f"{sum(trajectory_scores) / len(trajectory_scores):.2f}" if trajectory_scores else "N/A"
                
                ui.html(f'''
                    <div style="
                        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 24px;
                        border-radius: 12px;
                        margin-bottom: 24px;
                        box-shadow: 0 10px 25px rgba(79, 172, 254, 0.3);
                    ">
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;">
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📍 Steps</div>
                                <div style="font-size: 32px; font-weight: 700;">{steps_count}</div>
                            </div>
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">👥 Agents</div>
                                <div style="font-size: 32px; font-weight: 700;">{int(unique_agents)}</div>
                            </div>
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📉 Min Score</div>
                                <div style="font-size: 32px; font-weight: 700;">{min_score}</div>
                            </div>
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📊 Avg Score</div>
                                <div style="font-size: 32px; font-weight: 700;">{avg_score}</div>
                            </div>
                        </div>
                    </div>
                ''')

                # Agent flow - Display at the top, closed by default
                with ui.expansion("Agent Flow (Steps and Agents)", icon="account_tree", value=False).classes("w-full"):
                    agent_flow = task_data["Name"].tolist()
                    badges = []
                    for i, agent in enumerate(agent_flow):
                        badges.append(f'<span class="agent-badge">{agent}</span>')
                        if i < len(agent_flow) - 1:
                            badges.append('<span class="agent-arrow">&#8594;</span>')
                    ui.html("".join(badges))

                render_divider()

                # Check for trace evaluation results
                traj_eval_results = state.metadata.get("traj_eval_results", {})
                rubric_eval_results = state.metadata.get("rubric_eval_results", {})
                traj_eval = None
                rubric_eval = None
                
                # Try to find evaluation for this trace
                # The trace name format is typically: task_id_variant (e.g., "1a79e37_2")
                for traj_name, eval_data in traj_eval_results.items():
                    if traj_name.startswith(str(task_id)):
                        traj_eval = eval_data
                        break
                
                # Try to find rubric evaluation for this trace
                rubric_eval = rubric_eval_results.get(str(task_id))
                if not rubric_eval:
                    # Try without converting to string
                    rubric_eval = rubric_eval_results.get(task_id)
                
                # Display trace evaluation scores if available
                if traj_eval or rubric_eval:
                    render_section_header("📊 Trace Evaluation", "Overall assessment of this trace's quality")
                    
                    # Create nested tabs for different evaluation types
                    with ui.tabs().classes("w-full") as eval_tabs:
                        tab_standard = ui.tab("Standard Evaluation", icon="analytics")
                        tab_rubric = ui.tab("Rubric Evaluation", icon="rule")
                    
                    with ui.tab_panels(eval_tabs, value=tab_standard).classes("w-full"):
                        # Tab 1: Standard Evaluation (existing dimension-based evaluation)
                        with ui.tab_panel(tab_standard):
                            if traj_eval:
                                render_standard_evaluation(traj_eval)
                            else:
                                ui.html('<div style="text-align:center; padding:40px; color:#94A3B8;">No standard evaluation available for this trace</div>')
                        
                        # Tab 2: Rubric Evaluation (new rubric-based evaluation)
                        with ui.tab_panel(tab_rubric):
                            if rubric_eval:
                                render_rubric_evaluation(rubric_eval)
                            else:
                                ui.html('<div style="text-align:center; padding:40px; color:#94A3B8;">No rubric evaluation available for this trace</div>')
                    
                    render_divider()

                # Score progression
                if scores_by_step:
                    render_section_header("Score Progression", "How scores evolve across trace steps")
                    steps = sorted(scores_by_step.keys())
                    scores = [scores_by_step[s]["score"] for s in steps]
                    agents_list = [scores_by_step[s]["agent"] for s in steps]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=steps, y=scores, mode="lines+markers", name="Score",
                        line=dict(color=COLORS["primary"], width=2.5),
                        marker=dict(size=10, color=scores, colorscale="RdYlGn", showscale=True,
                                    colorbar=dict(title="Score", thickness=12)),
                        text=agents_list,
                        hovertemplate="<b>Step %{x}</b><br>Agent: %{text}<br>Score: %{y:.2f}<extra></extra>",
                    ))
                    fig.update_layout(
                        xaxis_title="Step Number", yaxis_title="Score",
                        yaxis=dict(range=[0, 1]), hovermode="closest", height=350,
                    )
                    ui.plotly(fig).classes("w-full")

                render_divider()
                render_section_header("Step-by-Step Execution", "Expand each step to see inputs and outputs")

                for idx, row in task_data.iterrows():
                    step_num = int(row["step_in_trace_general"])
                    agent_name = row["Name"]
                    # Build score label for expansion header
                    step_score = scores_by_step.get(step_num, {}).get("score")
                    if step_score is not None:
                        if step_score >= 0.7:
                            score_color = "#16A34A"
                            score_text = f"{step_score:.2f}"
                        elif step_score >= 0.4:
                            score_color = "#D97706"
                            score_text = f"{step_score:.2f}"
                        else:
                            score_color = "#DC2626"
                            score_text = f"{step_score:.2f}"
                    else:
                        score_color = "#94A3B8"
                        score_text = "N/A"

                    with ui.expansion(
                        f"Step {step_num}: {agent_name}  |  Score: {score_text}",
                        icon="play_circle_outline",
                    ).classes("w-full").style(f"border-left: 4px solid {score_color}; margin-bottom: 4px;"):

                        with ui.row().classes("w-full gap-4"):
                            with ui.column().classes("flex-1"):
                                ui.label("Input").classes("text-sm font-semibold text-slate-600")
                                _inp = row.get("model_input", None)
                                input_text = str(_inp) if _inp is not None and not (isinstance(_inp, float) and pd.isna(_inp)) else "N/A"
                                ui.html(f'<pre style="white-space:pre-wrap; font-size:12px; color:#334155; max-height:250px; overflow-y:auto; background:#F8FAFC; padding:12px; border-radius:8px; border:1px solid #E2E8F0;">{input_text[:1000]}{"..." if len(input_text) > 1000 else ""}</pre>')

                            with ui.column().classes("flex-1"):
                                ui.label("Output").classes("text-sm font-semibold text-slate-600")
                                _out = row.get("response", None)
                                output_text = str(_out) if _out is not None and not (isinstance(_out, float) and pd.isna(_out)) else "N/A"
                                ui.html(f'<pre style="white-space:pre-wrap; font-size:12px; color:#334155; max-height:250px; overflow-y:auto; background:#F8FAFC; padding:12px; border-radius:8px; border:1px solid #E2E8F0;">{output_text[:1000]}{"..." if len(output_text) > 1000 else ""}</pre>')

                        # Evaluation summary shown directly below input/output (not collapsible)
                        step_info = scores_by_step.get(step_num, {})
                        eval_summary = step_info.get("eval_summary")
                        if eval_summary:
                            ui.html(f'''
                                <div style="
                                    margin-top: 10px;
                                    padding: 12px;
                                    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
                                    border-radius: 8px;
                                    border-left: 4px solid #F59E0B;
                                    color: #78350F;
                                    font-size: 13px;
                                    line-height: 1.6;
                                ">
                                    <strong style="display:block; margin-bottom:4px;">📝 Evaluation Summary</strong>
                                    {eval_summary}
                                </div>
                            ''')

        def refresh_list():
            filtered_tasks = get_filtered_tasks()
            traj_list_container.clear()
            with traj_list_container:
                ui.html(f'<div style="color:#64748B; font-size:14px;">Showing <strong>{len(filtered_tasks)}</strong> of {len(available_tasks)} traces</div>')

                if not filtered_tasks:
                    ui.html('<div style="text-align:center; padding:30px; color:#F59E0B;">No traces match the selected filters</div>')
                    return

                # Build trace options with task_id and intent
                traj_options = {}
                for t in filtered_tasks:
                    task_data = state.traj_df[state.traj_df["task_id"] == t]
                    if not task_data.empty and "intent" in task_data.columns:
                        intent = task_data["intent"].iloc[0]
                        if intent and intent != "N/A" and isinstance(intent, str):
                            traj_options[t] = f"{t} - {intent[:80]}{'...' if len(str(intent)) > 80 else ''}"
                        else:
                            traj_options[t] = f"{t}"
                    else:
                        traj_options[t] = f"{t}"
                
                traj_select = ui.select(
                    traj_options,
                    value=filtered_tasks[0],
                    label="Select Trace",
                ).classes("w-full max-w-lg").props("outlined dense")

                def on_traj_change():
                    show_trajectory_details(traj_select.value)

                traj_select.on_value_change(on_traj_change)
                show_trajectory_details(filtered_tasks[0])

        # Trigger initial list
        refresh_list()

    # ────────────────────────────────────────────────────────────────
    # TAB 4: Path Analysis
    # ────────────────────────────────────────────────────────────────
    def build_path_analysis_tab(state: DashboardState):
        render_section_header("Predictive Patterns", "Analyze predictive patterns in agent traces")

        # Merge traj_score from agent scores into trace dataframe if not already present
        if "traj_score" not in state.traj_df.columns or state.traj_df["traj_score"].isnull().all():
            # Create a mapping of task_id to traj_score from agent score dataframes
            task_id_to_traj_score = {}
            if state.all_agent_scores_df:
                for agent_name, agent_df in state.all_agent_scores_df.items():
                    if "traj_score" in agent_df.columns and "task_id" in agent_df.columns:
                        for _, row in agent_df.iterrows():
                            task_id = row.get("task_id")
                            traj_score = row.get("traj_score")
                            if pd.notna(task_id) and pd.notna(traj_score):
                                # Use first non-null traj_score found for each task_id
                                if task_id not in task_id_to_traj_score:
                                    task_id_to_traj_score[task_id] = float(traj_score)
            
            # Add traj_score column to trace dataframe
            if task_id_to_traj_score:
                state.traj_df["traj_score"] = state.traj_df["task_id"].map(task_id_to_traj_score)

        # Add path filters - defaults set to 3-7 length and 15 min frequency
        with ui.column().classes("w-full gap-2 mb-4"):
            with ui.row().classes("w-full gap-4 items-center"):
                ui.label("Path Length:").classes("text-sm font-semibold text-slate-600")
                min_path_len = ui.number(value=3, min=1, max=200, step=1, label="Min").classes("w-24").props("outlined dense")
                ui.label("to").classes("text-sm text-slate-600")
                max_path_len = ui.number(value=7, min=1, max=200, step=1, label="Max").classes("w-24").props("outlined dense")
                
                ui.label("Min Frequency:").classes("text-sm font-semibold text-slate-600 ml-4")
                min_freq = ui.number(value=15, min=1, max=100, step=1).classes("w-24").props("outlined dense")
                
                ui.button("Apply Filters", on_click=lambda: update_paths(), icon="filter_alt").props("color=deep-purple-6 no-caps")
            
            # Add explanatory note
            ui.html('''
                <div style="font-size: 12px; color: #64748B; padding: 8px; background: #F8FAFC; border-radius: 6px; border-left: 3px solid #6366F1;">
                    <strong>💡 Tip:</strong> Analyzing <strong>partial paths</strong> (agent subsequences) to find patterns predictive of success or failure.
                    Adjust length (3-7 recommended) and frequency (15+ recommended) to find statistically significant patterns.
                </div>
            ''')
        
        paths_container = ui.column().classes("w-full")
        
        def update_paths():
            paths_container.clear()
            with paths_container:
                # Get filter values
                min_len = int(min_path_len.value)
                max_len = int(max_path_len.value)
                min_frequency = int(min_freq.value)
                
                # Create list of traces and their labels
                traces = []
                labels = []
                
                # Group by task_id to get one trace per task
                for task_id in state.traj_df["task_id"].unique():
                    task_data = state.traj_df[state.traj_df["task_id"] == task_id].sort_values("step_in_trace_general")
                    
                    # Extract agent sequence
                    agent_sequence = task_data["Name"].tolist()
                    traces.append(agent_sequence)
                    
                    # Get label (1=success, 0=failure) - use traj_score if available
                    if "traj_score" in task_data.columns and task_data["traj_score"].notna().any():
                        traj_score = task_data["traj_score"].iloc[0]
                        # Consider score > 0.5 as success
                        labels.append(1 if traj_score > 0.5 else 0)
                    else:
                        # Fallback: try to get from agent scores
                        if state.all_agent_scores:
                            # Get average score across agents for this task
                            task_scores = []
                            for agent_data in state.all_agent_scores.values():
                                if "id_to_score" in agent_data and task_id in agent_data["id_to_score"]:
                                    task_scores.append(agent_data["id_to_score"][task_id])
                            if task_scores:
                                avg_score = sum(task_scores) / len(task_scores)
                                labels.append(1 if avg_score > 0.5 else 0)
                            else:
                                labels.append(0)  # Default to failure if no score
                        else:
                            labels.append(0)  # Default to failure if no scores available
                
                render_path_analysis_content(traces, labels, min_len, max_len, min_frequency, state)
        
        def render_path_analysis_content(traces, labels, min_len, max_len, min_frequency, state):
                # Import the pattern analysis function
                from clear_eval.agentic.dashboard.path_analysis import find_predictive_patterns
                
                # Display summary metrics with enhanced styling
                total_trajs = len(traces)
                success_count = sum(labels)
                failure_count = total_trajs - success_count
                success_rate = (success_count / total_trajs * 100) if total_trajs > 0 else 0
                
                # Visual summary banner
                ui.html(f'''
                    <div style="background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%); padding: 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; gap: 20px;">
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📊 Total Traces</div>
                                <div style="font-size: 32px; font-weight: 700;">{total_trajs}</div>
                            </div>
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">✅ Success Rate</div>
                                <div style="font-size: 32px; font-weight: 700;">{success_rate:.1f}%</div>
                            </div>
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">🎯 Successes</div>
                                <div style="font-size: 32px; font-weight: 700; color: #10B981;">{success_count}</div>
                            </div>
                            <div style="text-align: center; color: white;">
                                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">⚠️ Failures</div>
                                <div style="font-size: 32px; font-weight: 700; color: #EF4444;">{failure_count}</div>
                            </div>
                        </div>
                    </div>
                ''')
                
                # Call find_predictive_patterns with the filter values
                if total_trajs > 0:
                    try:
                        patterns_df = find_predictive_patterns(
                            trajectories=traces,
                            labels=labels,
                            min_pattern_len=min_len,
                            max_pattern_len=max_len,
                            min_occurrences=min_frequency,
                            p_value_threshold=0.05
                        )
                        
                        if patterns_df is not None and len(patterns_df) > 0:
                            # Separate success and failure patterns
                            success_patterns = patterns_df[patterns_df['effect'] > 0].sort_values('effect', ascending=False)
                            failure_patterns = patterns_df[patterns_df['effect'] < 0].sort_values('effect')
                            
                            # Display Success Patterns
                            render_section_header(
                                "🎯 Success-Predictive Patterns",
                                f"Agent sequences that significantly predict success (p < 0.05, min freq: {min_frequency})"
                            )
                            
                            if len(success_patterns) > 0:
                                ui.html(f'''
                                    <div style="padding: 12px; background: #D1FAE5; border-left: 4px solid #10B981; border-radius: 6px; margin-bottom: 16px;">
                                        <strong>✅ Found {len(success_patterns)} significant success patterns</strong><br>
                                        <span style="font-size: 13px; color: #065F46;">
                                            These agent sequences are statistically associated with higher success rates.
                                            Effect shows the increase in success probability when the pattern is present.
                                        </span>
                                    </div>
                                ''')
                                
                                # Add expandable tips
                                with ui.expansion("ℹ️ Column Explanations", icon="help_outline").classes("w-full mb-4").props("dense"):
                                    ui.html('''
                                        <div style="padding: 8px; font-size: 13px; line-height: 1.6;">
                                            <div style="margin-bottom: 12px;">
                                                <strong style="color: #059669;">📊 Success Rate (with):</strong>
                                                <div style="margin-left: 20px; color: #374151;">
                                                    Success rate when this pattern appears in the trace.
                                                    Shows the percentage of successful tasks that contain this specific agent sequence.
                                                </div>
                                            </div>
                                            <div style="margin-bottom: 12px;">
                                                <strong style="color: #059669;">📈 Effect Size:</strong>
                                                <div style="margin-left: 20px; color: #374151;">
                                                    Impact on success rate - the increase in success probability when this pattern is present.
                                                    Calculated as: (Success Rate with Pattern) - (Baseline Success Rate).
                                                    Higher values indicate stronger predictive power.
                                                </div>
                                            </div>
                                            <div>
                                                <strong style="color: #059669;">🎯 Predictive Score:</strong>
                                                <div style="margin-left: 20px; color: #374151;">
                                                    Combined metric considering effect size, frequency, and statistical significance.
                                                    Higher scores indicate more reliable and impactful patterns.
                                                </div>
                                            </div>
                                        </div>
                                    ''')
                                
                                # Convert to display format with visual enhancements
                                success_rows = []
                                for idx, row in enumerate(success_patterns.head(15).iterrows(), 1):
                                    _, row = row
                                    effect_pct = row['effect']*100
                                    # Visual indicator based on effect size
                                    if effect_pct > 30:
                                        indicator = "🔥"
                                    elif effect_pct > 20:
                                        indicator = "⭐"
                                    else:
                                        indicator = "✓"
                                    
                                    success_rows.append({
                                        "Rank": f"#{idx}",
                                        "Impact": indicator,
                                        "Pattern": row['pattern'],
                                        "Length": row['length'],
                                        "Freq": row['occurrences'],
                                        "Success w/": f"{row['success_rate_with']*100:.1f}%",
                                     #   "Success w/o": f"{row['success_rate_without']*100:.1f}%",
                                        "Effect": f"+{effect_pct:.1f}%",
                                        "Predictive Score": row["predictive_score"]
                                     #   "Lift": f"{row['lift']:.2f}x",
                                     #   "p-value": f"{row['p_value']:.4f}"
                                    })
                                
                                success_cols = [
                                    {"name": "Rank", "label": "Rank", "field": "Rank", "sortable": False, "align": "center", "style": "width: 60px"},
                                    {"name": "Impact", "label": "💫", "field": "Impact", "sortable": False, "align": "center", "style": "width: 50px"},
                                    {"name": "Pattern", "label": "Agent Sequence", "field": "Pattern", "sortable": True, "align": "left", "style": "font-family: monospace; font-size: 13px; max-width: 400px; overflow-x: auto; white-space: nowrap;"},
                                    {"name": "Length", "label": "Len", "field": "Length", "sortable": True, "align": "center", "style": "width: 60px"},
                                    {"name": "Freq", "label": "Freq", "field": "Freq", "sortable": True, "align": "center", "style": "width: 70px"},
                                    {"name": "Success w/", "label": "Success Rate (with)", "field": "Success w/", "sortable": True, "align": "center"},
                               #     {"name": "Success w/o", "label": "Success Rate (w/o)", "field": "Success w/o", "sortable": True, "align": "center"},
                                    {"name": "Effect", "label": "Effect Size", "field": "Effect", "sortable": True, "align": "center", "style": "width: 100px"},
                                   # {"name": "Predictive Score", "label": "Predictive Score", "field": "Predictive Score", "sortable": True,
                                    # "align": "center", "style": "width: 130px"},
                               #     {"name": "Lift", "label": "Lift", "field": "Lift", "sortable": True, "align": "center", "style": "width: 80px"},
                               #     {"name": "p-value", "label": "p-value", "field": "p-value", "sortable": True, "align": "center", "style": "width: 90px"}
                                ]
                                success_table = ui.table(
                                    columns=success_cols,
                                    rows=success_rows,
                                    row_key="Pattern",
                                    pagination={"rowsPerPage": 10}
                                ).classes("w-full shadow-sm").props("flat bordered").style("border-radius: 8px; overflow: hidden;")
                                
                                # Add custom CSS for scrollable pattern cells
                                ui.add_head_html('''
                                    <style>
                                    .q-table td[data-label="Agent Sequence"] {
                                        max-width: 400px !important;
                                        overflow-x: auto !important;
                                        white-space: nowrap !important;
                                        display: block !important;
                                    }
                                    .q-table td[data-label="Agent Sequence"]::-webkit-scrollbar {
                                        height: 6px;
                                    }
                                    .q-table td[data-label="Agent Sequence"]::-webkit-scrollbar-thumb {
                                        background: #CBD5E1;
                                        border-radius: 3px;
                                    }
                                    .q-table td[data-label="Agent Sequence"]::-webkit-scrollbar-track {
                                        background: #F1F5F9;
                                    }
                                    </style>
                                ''')
                            else:
                                ui.html('''
                                    <div style="text-align:center; padding:20px; color:#64748B;">
                                        No statistically significant success patterns found with current filters.
                                        Try adjusting the min frequency or length range.
                                    </div>
                                ''')
                            
                            render_divider()
                            
                            # Display Failure Patterns
                            render_section_header(
                                "⚠️ Failure-Predictive Patterns",
                                f"Agent sequences that significantly predict failure (p < 0.05, min freq: {min_frequency})"
                            )
                            
                            if len(failure_patterns) > 0:
                                ui.html(f'''
                                    <div style="padding: 12px; background: #FEE2E2; border-left: 4px solid #EF4444; border-radius: 6px; margin-bottom: 16px;">
                                        <strong>🚨 Found {len(failure_patterns)} significant failure patterns</strong><br>
                                        <span style="font-size: 13px; color: #991B1B;">
                                            These agent sequences are statistically associated with lower success rates.
                                            Effect shows the decrease in success probability when the pattern is present.
                                        </span>
                                    </div>
                                ''')
                                
                                # Convert to display format with visual enhancements
                                failure_rows = []
                                for idx, row in enumerate(failure_patterns.head(15).iterrows(), 1):
                                    _, row = row
                                    effect_pct = abs(row['effect']*100)
                                    # Visual indicator based on negative effect size
                                    if effect_pct > 30:
                                        indicator = "🔴"
                                    elif effect_pct > 20:
                                        indicator = "⚠️"
                                    else:
                                        indicator = "⚡"
                                    
                                    failure_rows.append({
                                        "Rank": f"#{idx}",
                                        "Risk": indicator,
                                        "Pattern": row['pattern'],
                                        "Length": row['length'],
                                        "Freq": row['occurrences'],
                                        "Success w/": f"{row['success_rate_with']*100:.1f}%",
                                       # "Success w/o": f"{row['success_rate_without']*100:.1f}%",
                                        "Effect": f"{row['effect']*100:.1f}%",
                                       # "Lift": f"{row['lift']:.2f}x",
                                       # "p-value": f"{row['p_value']:.4f}"
                                     #   "Predictive Score": row["predictive_score"],
                                    })
                                print(f"base rate = {np.mean(labels)}")
                                failure_cols = [
                                    {"name": "Rank", "label": "Rank", "field": "Rank", "sortable": False, "align": "center", "style": "width: 60px"},
                                    {"name": "Risk", "label": "⚠️", "field": "Risk", "sortable": False, "align": "center", "style": "width: 50px"},
                                    {"name": "Pattern", "label": "Agent Sequence", "field": "Pattern", "sortable": True, "align": "left", "style": "font-family: monospace; font-size: 13px; max-width: 400px; overflow-x: auto; white-space: nowrap;"},
                                    {"name": "Length", "label": "Len", "field": "Length", "sortable": True, "align": "center", "style": "width: 60px"},
                                    {"name": "Freq", "label": "Freq", "field": "Freq", "sortable": True, "align": "center", "style": "width: 70px"},
                                    {"name": "Success w/", "label": "Success Rate (with)", "field": "Success w/", "sortable": True, "align": "center"},
                                  #  {"name": "Success w/o", "label": "Success Rate (w/o)", "field": "Success w/o", "sortable": True, "align": "center"},
                                    {"name": "Effect", "label": "Effect Size", "field": "Effect", "sortable": True, "align": "center", "style": "width: 100px"},
                                    #{"name": "Predictive Score", "label": "Predictive Score", "field": "Predictive Score", "sortable": True, "align": "center", "style": "width: 130px"},
                                  #  {"name": "Lift", "label": "Lift", "field": "Lift", "sortable": True, "align": "center", "style": "width: 80px"},
                                  #  {"name": "p-value", "label": "p-value", "field": "p-value", "sortable": True, "align": "center", "style": "width: 90px"}
                                ]
                                ui.table(
                                    columns=failure_cols,
                                    rows=failure_rows,
                                    row_key="Pattern",
                                    pagination={"rowsPerPage": 10}
                                ).classes("w-full shadow-sm").props("flat bordered").style("border-radius: 8px; overflow: hidden;")
                            else:
                                ui.html('''
                                    <div style="text-align:center; padding:20px; color:#64748B;">
                                        No statistically significant failure patterns found with current filters.
                                        Try adjusting the min frequency or length range.
                                    </div>
                                ''')
                        else:
                            ui.html('''
                                <div style="text-align:center; padding:40px; color:#64748B;">
                                    <strong>No significant patterns found</strong><br>
                                    Try adjusting the filters (lower min frequency or change length range) to find patterns.
                                </div>
                            ''')
                    except Exception as e:
                        ui.html(f'''
                            <div style="padding: 12px; background: #FEE2E2; border-left: 4px solid #EF4444; border-radius: 6px;">
                                <strong>Error analyzing patterns:</strong> {str(e)}
                            </div>
                        ''')
                else:
                    ui.html('''
                        <div style="text-align:center; padding:40px; color:#64748B;">
                            No trace data available for analysis.
                        </div>
                    ''')

        
        # Initial render
        update_paths()

    # ────────────────────────────────────────────────────────────────
    # TAB 5: Temporal Analysis
    # ────────────────────────────────────────────────────────────────
    def build_temporal_tab(state: DashboardState):
        render_section_header("Temporal Analysis", "Analyze how agent performance and behavior changes through traces")

        # Trace Length Analysis
        render_section_header("Trace Length Analysis", "Distribution and score correlation with trace length")

        traj_lengths = {}
        traj_scores = {}
        for task_id in state.traj_df["task_id"].unique():
            task_data = state.traj_df[state.traj_df["task_id"] == task_id]
            traj_lengths[task_id] = len(task_data)
            for agent_name, agent_data in state.all_agent_scores.items():
                if "id_to_score" in agent_data and task_id in agent_data["id_to_score"]:
                    traj_scores[task_id] = agent_data["id_to_score"][task_id]
                    break

        length_df = pd.DataFrame({"task_id": list(traj_lengths.keys()), "length": list(traj_lengths.values())})
        if traj_scores:
            length_df["score"] = length_df["task_id"].map(traj_scores)
            length_df = length_df.dropna(subset=["score"])

        # Gradient banner for temporal metrics
        avg_length = f"{length_df['length'].mean():.1f}"
        median_length = f"{length_df['length'].median():.0f}"
        avg_score = f"{length_df['score'].mean():.2f}" if traj_scores and "score" in length_df.columns and not length_df.empty else "N/A"
        total_trajs = len(length_df)
        
        ui.html(f'''
            <div style="
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                padding: 24px;
                border-radius: 12px;
                margin-bottom: 24px;
                box-shadow: 0 10px 25px rgba(250, 112, 154, 0.3);
            ">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px;">
                    <div style="text-align: center; color: white;">
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📊 Total Traces</div>
                        <div style="font-size: 32px; font-weight: 700;">{total_trajs}</div>
                    </div>
                    <div style="text-align: center; color: white;">
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📏 Avg Length</div>
                        <div style="font-size: 32px; font-weight: 700;">{avg_length}</div>
                    </div>
                    <div style="text-align: center; color: white;">
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">📍 Median Length</div>
                        <div style="font-size: 32px; font-weight: 700;">{median_length}</div>
                    </div>
                    <div style="text-align: center; color: white;">
                        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 4px;">⭐ Avg Score</div>
                        <div style="font-size: 32px; font-weight: 700;">{avg_score}</div>
                    </div>
                </div>
            </div>
        ''')
        
        with ui.row().classes("w-full gap-4"):
            with ui.column().classes("flex-1"):
                fig_dist = px.histogram(
                    length_df, x="length", nbins=20,
                    labels={"length": "Number of Steps", "count": "Traces"},
                    color_discrete_sequence=[COLORS["primary"]],
                )
                fig_dist.update_layout(showlegend=False, height=350, title=dict(text="Distribution of Trace Lengths"))
                ui.plotly(fig_dist).classes("w-full")

            with ui.column().classes("flex-1"):
                if traj_scores and "score" in length_df.columns and not length_df.empty:
                    score_by_length = length_df.groupby("length")["score"].agg(["mean", "count"]).reset_index().sort_values("length")
                    fig_score = go.Figure()
                    fig_score.add_trace(go.Scatter(
                        x=score_by_length["length"], y=score_by_length["mean"],
                        mode="lines+markers", name="Average Score",
                        line=dict(color=COLORS["primary"], width=2.5),
                        marker=dict(size=8, color=COLORS["primary"]),
                        hovertemplate="<b>Length:</b> %{x} steps<br><b>Avg Score:</b> %{y:.3f}<extra></extra>",
                    ))
                    fig_score.update_layout(
                        title=dict(text="Average Score by Trace Length"),
                        xaxis_title="Number of Steps", yaxis_title="Average Score",
                        showlegend=False, height=350, hovermode="closest",
                    )
                    ui.plotly(fig_score).classes("w-full")
                else:
                    with ui.card().classes("w-full items-center justify-center").style("min-height: 350px;"):
                        ui.icon("info").classes("text-4xl text-blue-300")
                        ui.label("Score data not available for length analysis").classes("text-slate-500")

        render_divider()

        # Agent Position Analysis
        render_section_header("Agent Position in Traces", "Where do agents typically appear in the workflow?")

        position_df = analyze_agent_positions(state.traj_df)

        if not position_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=position_df["agent"], y=position_df["avg_normalized_position"],
                name="Avg Position",
                marker_color=COLORS["primary"],
                marker=dict(opacity=0.85),
                hovertemplate="<b>%{x}</b><br>Avg Position: %{y:.2f}<br>Appearances: %{text}<extra></extra>",
                text=position_df["appearances"],
            ))
            fig.update_layout(
                title=dict(text="Average Normalized Position of Agents (0=Start, 1=End)"),
                xaxis_title="Agent", yaxis_title="Normalized Position",
                height=400, yaxis=dict(range=[0, 1]),
            )
            ui.plotly(fig).classes("w-full")

            # Table
            display_df = position_df[["agent", "avg_position", "avg_normalized_position", "appearances",
                                      "early_appearances", "mid_appearances", "late_appearances"]].copy()
            display_df.columns = ["Agent", "Avg Position", "Normalized Pos", "Total", "Early", "Mid", "Late"]
            display_df["Normalized Pos"] = display_df["Normalized Pos"].apply(lambda x: f"{x:.3f}")
            display_df["Avg Position"] = display_df["Avg Position"].apply(lambda x: f"{x:.1f}")
            rows = display_df.to_dict("records")
            cols = [{"name": c, "label": c, "field": c, "sortable": True, "align": "left"} for c in display_df.columns]
            ui.table(columns=cols, rows=rows, row_key="Agent", pagination={"rowsPerPage": 10}).classes("w-full shadow-md").props("flat bordered").style("border-radius: 8px; overflow: hidden;")

        # Score Progression
        if state.all_agent_scores_df:
            render_divider()
            render_section_header("Score Progression Through Traces", "How do scores change as traces progress?")

            progression_data = analyze_score_progression(state.traj_df, state.all_agent_scores_df)

            if progression_data and len(progression_data["score_change"]) > 0:
                position_changes = defaultdict(list)
                for pos, change in zip(progression_data["position"], progression_data["score_change"]):
                    position_changes[pos].append(change)
                avg_changes = {pos: sum(changes) / len(changes) for pos, changes in position_changes.items()}

                positions = sorted(avg_changes.keys())
                changes = [avg_changes[p] for p in positions]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=positions, y=changes, mode="lines+markers",
                    name="Avg Score Change",
                    line=dict(color=COLORS["success"] if sum(changes) > 0 else COLORS["danger"], width=2.5),
                    marker=dict(size=8),
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="#CBD5E1", annotation_text="No Change")
                fig.update_layout(
                    title=dict(text="Average Score Change by Position in Trace"),
                    xaxis_title="Step Position", yaxis_title="Average Score Change", height=400,
                )
                ui.plotly(fig).classes("w-full")

                improving = sum(1 for x in progression_data["improving"] if x)
                total = len(progression_data["improving"])
                avg_change = sum(progression_data["score_change"]) / len(progression_data["score_change"])

                with ui.row().classes("w-full gap-4 justify-center"):
                    render_metric_card("Improving Steps", f"{improving}/{total} ({improving / total * 100:.1f}%)")
                    render_metric_card("Avg Score Change", f"{avg_change:+.4f}")
                    render_metric_card("Total Transitions", len(progression_data["score_change"]))
            else:
                ui.html('<div style="text-align:center; padding:20px; color:#0EA5E9;">Not enough score data to analyze progression</div>')



# ─── Entry Point ─────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # TAB 6: Score Prediction Analysis
    # ────────────────────────────────────────────────────────────────
    def build_score_prediction_tab(state: DashboardState):
        render_section_header("Trace Score Prediction Analysis", "Predict trace success using step-level scores")
        
        # Check if we have agent results (CLEAR evaluation data)
        if not state.agent_results:
            ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;"><p>No agent evaluation results available.</p><p style="margin-top:10px; color:#94A3B8; font-size:14px;">CLEAR evaluation results are required for this analysis.</p></div>')
            return
        
        # Load and combine all agent result DataFrames, looking for traj_score
        # Use state.traj_df which already has traj_score loaded from clear_data CSVs
        if state.traj_df.empty:
            ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;"><p>No trace data available.</p></div>')
            return
        
        # Check if traj_score column exists
        has_traj_score = "traj_score" in state.traj_df.columns and not state.traj_df["traj_score"].isnull().all()
        available_columns = set(state.traj_df.columns)
        
        # Debug: show available columns
        #ui.html(f'<div style="color:#64748B; font-size:12px; margin-bottom:10px;">Available columns: {", ".join(sorted(available_columns))}</div>')
        
        if not has_traj_score:
            ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;"><p>No trace score data available.</p><p style="margin-top:10px; color:#94A3B8; font-size:14px;">The "traj_score" column is required for this analysis. This column should contain the ground truth success score for each trace.</p></div>')
            return
        
        # Start with trace dataframe which has traj_score
        traj_scores_df = state.traj_df.copy()
        
        print(f"  📊 Starting with traj_df: {traj_scores_df.shape} rows")
        print(f"  📊 Has traj_score: {'traj_score' in traj_scores_df.columns}")
        
        # Merge agent evaluation scores (which have the "score" column)
        trajectory_df = state.metadata.get("trajectory_df")
        all_agent_dfs = []
        
        for agent_name, agent_data in state.agent_results.items():
            try:
                file_bytes = agent_data["zip_bytes"]
                zip_name = agent_data.get("zip_name", f"{agent_name}.zip")
                df, _ = load_clear_data_from_bytes(file_bytes, zip_name, trajectory_df)
                
                if not df.empty and "score" in df.columns:
                    # Keep only necessary columns for merging
                    merge_cols = ["task_id", "step_in_trace_general", "score"]
                    if all(col in df.columns for col in merge_cols):
                        all_agent_dfs.append(df[merge_cols])
            except Exception as e:
                continue
        
        # Merge agent scores with trace data
        if all_agent_dfs:
            agent_scores_df = pd.concat(all_agent_dfs, ignore_index=True)
            # Group by task_id and step to get average score across agents
            agent_scores_df = agent_scores_df.groupby(["task_id", "step_in_trace_general"])["score"].mean().reset_index()
            
            # Merge with trace data
            traj_scores_df = traj_scores_df.merge(
                agent_scores_df,
                on=["task_id", "step_in_trace_general"],
                how="left"
            )
            print(f"  ✓ Merged agent scores: {traj_scores_df.shape} rows, has score: {'score' in traj_scores_df.columns}")
        else:
            print(f"  ⚠️  No agent scores found to merge")
        
        print(f"  📊 Final traj_scores_df: {traj_scores_df.shape} rows")
        print(f"  📊 Columns: {list(traj_scores_df.columns)}")
        print(f"  📊 Unique task_ids: {len(traj_scores_df['task_id'].unique()) if 'task_id' in traj_scores_df.columns else 0}")
        
        # Get trace evaluation results if available
        traj_eval_results = state.metadata.get("traj_eval_results", {})
        
        # Get rubric evaluation results if available
        rubric_eval_results = state.metadata.get("rubric_eval_results", {})
        
        # Group by task_id to get trace-level data
        traj_data = []
        for task_id in traj_scores_df["task_id"].unique():
            task_rows = traj_scores_df[traj_scores_df["task_id"] == task_id]
            if len(task_rows) > 0:
                traj_score = task_rows["traj_score"].iloc[0]  # Ground truth
                
                # Get step scores if available
                step_scores = None
                min_score = None
                avg_score = None
                if "score" in task_rows.columns:
                    step_scores = task_rows["score"].dropna()
                    if len(step_scores) > 0:
                        min_score = step_scores.min()
                        avg_score = step_scores.mean()
                
                # Try to get trace-level prediction from traj_eval_results
                traj_prediction = None
                if task_id in traj_eval_results:
                    traj_prediction = traj_eval_results[task_id].get("overall_score")
                
                # Try to get rubric score from rubric_eval_results
                rubric_score = None
                if task_id in rubric_eval_results:
                    rubric_score = rubric_eval_results[task_id].get("score")
                
                traj_data.append({
                    "task_id": task_id,
                    "traj_score": traj_score,
                    "min_score": min_score,
                    "avg_score": avg_score,
                    "traj_length": len(task_rows),
                    "traj_prediction": traj_prediction,  # Trace-level prediction
                    "rubric_score": rubric_score  # Rubric evaluation score
                })
        
        if not traj_data:
            ui.html(f'<div style="text-align:center; padding:40px; color:#F59E0B;">No valid trace data found for analysis.<br><small style="color:#94A3B8;">Checked {len(traj_scores_df["task_id"].unique())} unique tasks</small></div>')
            return
        
        print(f"  ✓ Created trace data for {len(traj_data)} tasks")
        traj_df = pd.DataFrame(traj_data)
        
        # Determine success threshold (e.g., traj_score > 0.5)
        threshold = 0.5
        traj_df["success"] = (traj_df["traj_score"] > threshold).astype(int)
        
        # Statistics with gradient banner
        total_trajs = len(traj_df)
        success_count = traj_df["success"].sum()
        success_pct = f"{success_count/len(traj_df)*100:.1f}"
        avg_length = f"{traj_df['traj_length'].mean():.1f}"
        
        ui.html(f'''
            <div style="
                background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                padding: 24px;
                border-radius: 12px;
                margin-bottom: 24px;
                box-shadow: 0 10px 25px rgba(168, 237, 234, 0.3);
            ">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px;">
                    <div style="text-align: center; color: #1e293b;">
                        <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">📊 Total Traces</div>
                        <div style="font-size: 32px; font-weight: 700;">{total_trajs}</div>
                    </div>
                    <div style="text-align: center; color: #1e293b;">
                        <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">✅ Successful</div>
                        <div style="font-size: 32px; font-weight: 700;">{success_count} ({success_pct}%)</div>
                    </div>
                    <div style="text-align: center; color: #1e293b;">
                        <div style="font-size: 14px; opacity: 0.8; margin-bottom: 4px;">📏 Avg Length</div>
                        <div style="font-size: 32px; font-weight: 700;">{avg_length}</div>
                    </div>
                </div>
            </div>
        ''')
        
        render_divider()
        
        # ROC Curve Analysis - header will be updated dynamically
        header_container = ui.column().classes("w-full")
        
        try:
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            
            # Get list of unique agents - use "Name" column (which may have been created from "agent_name")
            agent_col = "Name" if "Name" in traj_scores_df.columns else "agent_name"
            agent_names = sorted(traj_scores_df[agent_col].unique())
            agent_options = ["All Agents"] + agent_names
            
            # Create containers for controls
            controls_row = ui.row().classes("w-full gap-4")
            
            with controls_row:
                # Left column: Prediction Method (always visible, fixed width)
                with ui.column().classes("flex-1"):
                    ui.label("Prediction Method").classes("text-sm font-semibold text-slate-600")
                    method_select = ui.select(
                        ["Average Score", "Trace Score", "Rubric Score"],
                        value="Average Score",
                        label="Select prediction method"
                    ).classes("w-full").props("outlined dense")
                
                # Right column: Agent Selection (conditionally visible content, but column always takes space)
                agent_column = ui.column().classes("flex-1")
                with agent_column:
                    agent_content = ui.column().classes("w-full")
                    with agent_content:
                        ui.label("Agent Selection").classes("text-sm font-semibold text-slate-600")
                        agent_select = ui.select(
                            agent_options,
                            value="All Agents",
                            label="Select agent for analysis"
                        ).classes("w-full").props("outlined dense")
            
            roc_container = ui.column().classes("w-full gap-4")
            
            def update_roc():
                # Update header based on method
                header_container.clear()
                with header_container:
                    if method_select.value == "Trace Score":
                        render_section_header("ROC Curve Analysis", "Predicting trace success using trace-level predictions")
                    elif method_select.value == "Rubric Score":
                        render_section_header("ROC Curve Analysis", "Predicting trace success using rubric evaluation scores")
                    else:
                        render_section_header("ROC Curve Analysis", "Predicting trace success using step-level scores")
                
                # Show/hide agent selector content based on method (column stays to preserve layout)
                if method_select.value in ["Trace Score", "Rubric Score"]:
                    agent_content.set_visibility(False)
                else:
                    agent_content.set_visibility(True)
                
                roc_container.clear()
                with roc_container:
                    method = method_select.value
                    
                    # For trace score and rubric score, always use all agents (no filtering)
                    if method in ["Trace Score", "Rubric Score"]:
                        selected_agent = "All Agents"
                    else:
                        selected_agent = agent_select.value
                    
                    # Filter data by selected agent
                    if selected_agent == "All Agents":
                        # Use all data - recalculate trace-level scores from all agents
                        filtered_scores_df = traj_scores_df
                        title_suffix = "All Agents"
                    else:
                        # Filter to specific agent - use the same column we determined earlier
                        filtered_scores_df = traj_scores_df[traj_scores_df[agent_col] == selected_agent]
                        title_suffix = selected_agent
                    
                    # Recalculate trace-level data for filtered agent(s)
                    filtered_traj_data = []
                    for task_id in filtered_scores_df["task_id"].unique():
                        task_rows = filtered_scores_df[filtered_scores_df["task_id"] == task_id]
                        if len(task_rows) > 0 and "score" in task_rows.columns:
                            traj_score = task_rows["traj_score"].iloc[0]
                            step_scores = task_rows["score"].dropna()
                            
                            if len(step_scores) > 0:
                                # Try to get trace-level prediction from traj_eval_results
                                traj_prediction = None
                                if task_id in traj_eval_results:
                                    traj_prediction = traj_eval_results[task_id].get("overall_score")
                                
                                # Try to get rubric score from rubric_eval_results
                                rubric_score = None
                                if task_id in rubric_eval_results:
                                    rubric_score = rubric_eval_results[task_id].get("score")
                                
                                filtered_traj_data.append({
                                    "task_id": task_id,
                                    "traj_score": traj_score,
                                    "min_score": step_scores.min(),
                                    "avg_score": step_scores.mean(),
                                    "traj_length": len(task_rows),
                                    "traj_prediction": traj_prediction,
                                    "rubric_score": rubric_score
                                })
                    
                    if not filtered_traj_data:
                        ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;">No valid trace data found for selected agent.</div>')
                        return
                    
                    filtered_traj_df = pd.DataFrame(filtered_traj_data)
                    filtered_traj_df["success"] = (filtered_traj_df["traj_score"] > threshold).astype(int)
                    
                    # Select prediction scores
                    # Note: y_true is always the binary success indicator (traj_score > threshold)
                    # y_pred is what we're using to predict that success
                    y_true = filtered_traj_df["success"]
                    
                    if method == "Trace Score":
                        # Use trace-level predictions from full_traj_results
                        # Filter out rows without trace predictions
                        valid_rows = filtered_traj_df[filtered_traj_df["traj_prediction"].notna()]
                        if len(valid_rows) == 0:
                            ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;">No trace-level predictions available. Make sure ui_results.zip contains full_traj_results/per_traj_reuslts/ with overall_score field.</div>')
                            return
                        y_pred = valid_rows["traj_prediction"]
                        y_true = valid_rows["success"]
                        filtered_traj_df = valid_rows
                        pred_label = "Trace-Level Prediction"
                    elif method == "Rubric Score":
                        # Use rubric scores from rubric_eval_results
                        # Filter out rows without rubric scores
                        valid_rows = filtered_traj_df[filtered_traj_df["rubric_score"].notna()]
                        if len(valid_rows) == 0:
                            ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;">No rubric evaluation scores available. Make sure ui_results.zip contains full_traj_results/rubric_eval_results/ with score field.</div>')
                            return
                        y_pred = valid_rows["rubric_score"]
                        y_true = valid_rows["success"]
                        filtered_traj_df = valid_rows
                        pred_label = "Rubric Evaluation Score"
                    else:
                        # Use average of step scores as prediction
                        y_pred = filtered_traj_df["avg_score"]
                        pred_label = "Avg Step Score"
                    
                    # Check if we have both classes
                    if len(y_true.unique()) < 2:
                        ui.html('<div style="text-align:center; padding:40px; color:#F59E0B;">ROC curve requires both successful and failed traces. Current selection has only one class.</div>')
                        return
                    
                    # Calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                    roc_auc = auc(fpr, tpr)
                    
                    # Plot ROC curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode="lines",
                        name=f"ROC curve (AUC = {roc_auc:.3f})",
                        line=dict(color=COLORS["primary"], width=2.5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines",
                        name="Random (AUC = 0.5)",
                        line=dict(color="#CBD5E1", width=2, dash="dash")
                    ))
                    fig.update_layout(
                        title=f"ROC Curve - {pred_label} Predicting Trace Success<br><sub>{title_suffix} ({len(filtered_traj_df)} traces)</sub>",
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=500,
                        showlegend=True,
                        legend=dict(x=0.6, y=0.1)
                    )
                    ui.plotly(fig).classes("w-full")
                    
                    # Show statistics
                    success_count = filtered_traj_df["success"].sum()
                    with ui.row().classes("w-full gap-4 justify-center mt-4"):
                        ui.html(f'<div style="text-align:center;"><div style="font-size:14px; color:#64748B;">Traces</div><div style="font-size:20px; font-weight:bold; color:#334155;">{len(filtered_traj_df)}</div></div>')
                        ui.html(f'<div style="text-align:center;"><div style="font-size:14px; color:#64748B;">Successful</div><div style="font-size:20px; font-weight:bold; color:{COLORS["success"]};">{success_count} ({success_count/len(filtered_traj_df)*100:.1f}%)</div></div>')
                        ui.html(f'<div style="text-align:center;"><div style="font-size:14px; color:#64748B;">AUC Score</div><div style="font-size:20px; font-weight:bold; color:{COLORS["primary"]};">{roc_auc:.3f}</div></div>')
                    
                    # Interpretation
                    if roc_auc >= 0.9:
                        interpretation = "Excellent prediction performance"
                        color = COLORS["success"]
                    elif roc_auc >= 0.8:
                        interpretation = "Good prediction performance"
                        color = COLORS["primary"]
                    elif roc_auc >= 0.7:
                        interpretation = "Fair prediction performance"
                        color = COLORS["warning"]
                    else:
                        interpretation = "Poor prediction performance"
                        color = COLORS["danger"]
                    
                    ui.html(f'<div style="text-align:center; margin-top:10px; color:{color}; font-size:16px;">{interpretation}</div>')
            
            agent_select.on_value_change(update_roc)
            method_select.on_value_change(update_roc)
            update_roc()
            
        except ImportError:
            ui.html('<div style="text-align:center; padding:20px; color:#F59E0B;">sklearn is required for ROC curve analysis. Please install it: pip install scikit-learn</div>')
        
        render_divider()
        
        # Success Rate by Trace Length
        render_section_header("Success Analysis by Trace Length", "How does trace length relate to success?")
        
        # Group by length and calculate success rate
        length_analysis = traj_df.groupby("traj_length").agg({
            "success": ["sum", "count", "mean"],
            "task_id": "count"
        }).reset_index()
        length_analysis.columns = ["traj_length", "successes", "total", "success_rate", "count"]
        length_analysis = length_analysis[length_analysis["count"] >= 2]  # Filter out lengths with < 2 samples
        
        if len(length_analysis) > 0:
            # Success rate by length
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=length_analysis["traj_length"],
                y=length_analysis["success_rate"] * 100,
                name="Success Rate",
                marker_color=COLORS["primary"],
                text=length_analysis["success_rate"].apply(lambda x: f"{x*100:.1f}%"),
                textposition="outside"
            ))
            fig.update_layout(
                title="Success Rate by Trace Length",
                xaxis_title="Trace Length (# of steps)",
                yaxis_title="Success Rate (%)",
                height=400,
                yaxis=dict(range=[0, 105])
            )
            ui.plotly(fig).classes("w-full")
        else:
            ui.html('<div style="text-align:center; padding:20px; color:#94A3B8;">Not enough data for length analysis (need at least 2 traces per length)</div>')
        
        render_divider()
        
        # Mean length comparison
        render_section_header("Trace Length Comparison", "Comparing successful vs failed traces")
        
        success_lengths = traj_df[traj_df["success"] == 1]["traj_length"]
        failure_lengths = traj_df[traj_df["success"] == 0]["traj_length"]
        
        with ui.row().classes("w-full gap-4 justify-center"):
            if len(success_lengths) > 0:
                with ui.card().classes("flex-1 custom-card"):
                    ui.html(f'<div class="metric-label">Mean Length (Success)</div>')
                    ui.html(f'<div class="metric-value">{success_lengths.mean():.1f} steps</div>')
                    ui.label(f"Std: {success_lengths.std():.1f}").classes("text-sm text-slate-500 mt-2")
            if len(failure_lengths) > 0:
                with ui.card().classes("flex-1 custom-card"):
                    ui.html(f'<div class="metric-label">Mean Length (Failure)</div>')
                    ui.html(f'<div class="metric-value">{failure_lengths.mean():.1f} steps</div>')
                    ui.label(f"Std: {failure_lengths.std():.1f}").classes("text-sm text-slate-500 mt-2")
            
            if len(success_lengths) > 0 and len(failure_lengths) > 0:
                diff = success_lengths.mean() - failure_lengths.mean()
                diff_pct = (diff / failure_lengths.mean()) * 100 if failure_lengths.mean() > 0 else 0
                with ui.card().classes("flex-1 custom-card"):
                    ui.html(f'<div class="metric-label">Length Difference</div>')
                    ui.html(f'<div class="metric-value">{diff:+.1f} steps</div>')
                    ui.label(f"{diff_pct:+.1f}% {'longer' if diff > 0 else 'shorter'}").classes("text-sm text-slate-500 mt-2")
        
        # Distribution comparison
        if len(success_lengths) > 0 and len(failure_lengths) > 0:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=success_lengths,
                name="Successful",
                opacity=0.7,
                marker_color=COLORS["success"],
                nbinsx=20
            ))
            fig.add_trace(go.Histogram(
                x=failure_lengths,
                name="Failed",
                opacity=0.7,
                marker_color=COLORS["danger"],
                nbinsx=20
            ))
            fig.update_layout(
                barmode='overlay',
                title="Distribution of Trace Lengths",
                xaxis_title="Trace Length (# of steps)",
                yaxis_title="Count",
                height=400,
                showlegend=True
            )
            ui.plotly(fig).classes("w-full")

def main():
    """Run the Agentic Workflow Dashboard."""
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nShutting down dashboard...")
        sys.exit(0)
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        ui.run(
            title="Agentic Workflow Dashboard",
            favicon="\U0001F916",
            dark=False,
            port=8080,
            reload=False,
            show=True,
        )
    except KeyboardInterrupt:
        print("\n\nShutting down dashboard...")
        sys.exit(0)


if __name__ == "__main__":
    main()
