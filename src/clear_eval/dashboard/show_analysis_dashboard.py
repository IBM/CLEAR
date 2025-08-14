import hashlib
import math
import re
import textwrap
import zipfile
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast  # For safely evaluating string representations of lists if needed
import os
from dotenv import load_dotenv
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import streamlit.components.v1 as components

from streamlit.runtime.uploaded_file_manager import UploadedFile
import pyarrow.parquet as pq

load_dotenv()
max_presented_examples = 1000
EXPECTED_COLS =  [
            "question_id", "model_input","response", "score",
             "evaluation_text", "evaluation_summary", "recurring_issues", "recurring_issues_str",
            "ground_truth"
         ]
DISPLAY_COLS = [
            "question_id", "model_input_preview","response", "score",
            "evaluation_summary", "recurring_issues_str", "ground_truth"
         ]
MAX_NUM_ISSUES = 30
OTHER = "Other Issues"
NO_ISSUE = "No Issues"

def get_display_filename(file_ref):
    if isinstance(file_ref, str):
        return os.path.basename(file_ref)
    elif isinstance(file_ref, UploadedFile):
        return file_ref.name
    else:
        return "Unknown source"


@st.cache_data
def load_data(file_bytes, file_name):
    with zipfile.ZipFile(BytesIO(file_bytes)) as zf:
        # Validate
        try:
            names = set(zf.namelist())
            metadata = None
            csv_file_name = None
            parquet_file_name = None
            for name in names:
                if name.endswith(".csv") and name.startswith('results'):
                    csv_file_name = name
                if name.endswith(".parquet") and name.startswith('results'):
                    parquet_file_name = name
                if name.endswith(".json") and name.startswith('metadata'):
                    metadata = json.load(zf.open(name))

            if csv_file_name is None and parquet_file_name is None:
                raise Exception("No valid csv or parquet file found in input")
            if metadata is None:
                raise Exception("No valid metadata json found in input")
            input_columns = get_input_columns(metadata)
            expected_cols = EXPECTED_COLS + input_columns
            if csv_file_name is not None:
                with zf.open(csv_file_name) as csv_file:
                    df = pd.read_csv(csv_file, usecols=lambda c: c in expected_cols)
            elif parquet_file_name is not None:
                with zf.open(parquet_file_name) as parquet_file:
                    actual_cols = pq.ParquetFile(parquet_file).schema.names
                    selected_cols = [c for c in expected_cols if c in actual_cols]
                    df = pd.read_parquet(parquet_file, engine="pyarrow", columns=selected_cols)
            # for col in expected_cols:
            #     if col not in df.columns:
            #         df[col] = np.nan
            if 'score' in df.columns:
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                df.dropna(subset=['score'], inplace=True)
            df.loc[:,"discovered_issues"] = df.apply(lambda r: ",\n".join(extract_issues(r["recurring_issues_str"])), axis=1)
            df["model_input_preview"] = df["model_input"].apply(lambda x: x[:300] if isinstance(x, str) else x)
            if "question_id" in df.columns:
                df.set_index("question_id", inplace=True)
        except FileNotFoundError:
            st.error(f"Error: {file_name} not found. Please ensure the file is in the correct directory.")
            return pd.DataFrame(columns=expected_cols), {}
        except ValueError as e:
            st.error(f"Error loading data from {file_name}. Please check column names and data format. Details: {e}")
            return pd.DataFrame(columns=expected_cols), {}
        return df, metadata

def extract_issues(text, delimiter=';'):
    if isinstance(text, np.ndarray):
        text = text.tolist()
    if isinstance(text, list):
        text = json.dumps(text)
    issues_list = extract_issues_from_str(text, delimiter)
    if len(issues_list) == 0:
        return [NO_ISSUE]
    return issues_list


def extract_issues_from_str(text, delimiter=';'):
    if pd.isna(text) or not text or text == "[]":
        return []
    try:
        evaluated = ast.literal_eval(text)
        if isinstance(evaluated, list):
            return [str(item).strip() for item in evaluated if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return [issue.strip() for issue in str(text).split(delimiter) if issue.strip()]

@st.cache_data
def get_issue_analysis(df, max_num_issues = None):
    if 'recurring_issues_str' not in df.columns or df['recurring_issues_str'].isnull().all():
        return pd.Series(dtype='int'), [], pd.Series(dtype='float')
    issues_per_row = df['recurring_issues_str'].apply(extract_issues)
    issues_score_df = pd.DataFrame({"issue": issues_per_row, "score": df['score']})
    issues_score_df_flat = issues_score_df.explode('issue')
    total_stats = len(df)
    issues_stats = issues_score_df_flat.groupby('issue')['score'].agg(['mean', 'std']).round(2)
    issues_stats.index.name = 'issue'
    issues_stats["issue_count"] = issues_score_df_flat['issue'].value_counts()
    issues_stats.loc[:,"issue_freq"] = issues_stats.apply(lambda r: (100*r["issue_count"]/total_stats).round(1), axis=1)
    return issues_stats

    # if max_num_issues and max_num_issues < len(issue_counts):
    #     # trim list of issues, move the rest to "other"
    #     issue_counts = issue_counts.head(max_num_issues)
    #     n_others = 0
    #
    #     # modify df to replace trimmed issues with "other"
    #     row_to_issues = dict(issues_per_row)
    #     filtered_row_to_top_issues = []
    #     filtered_row_to_other_issues = []
    #     for row, issues in row_to_issues.items():
    #         issues = list(filter(lambda issue: issue != NO_ISSUE, issues))
    #         top_issues = list(filter(lambda issue: issue in  issue_counts, issues))
    #         other_issues = list(filter(lambda issue: issue not in top_issues, issues))
    #         if other_issues:#(issues).difference(set(top_issues))) > 0:
    #             top_issues.append(OTHER)
    #             n_others += 1
    #         filtered_row_to_top_issues.append(str(top_issues))
    #         filtered_row_to_other_issues.append(str(other_issues))
    #     issue_counts[OTHER] = n_others
    #     df["recurring_issues_str"] = filtered_row_to_top_issues
    #     df["recurring_issues_other_str"] = filtered_row_to_other_issues
    #
    # unique_issue_names = sorted(issue_counts.index.tolist())
    # total_evals = len(df)
    # if total_evals > 0:
    #     issue_freq = (issue_counts / total_evals * 100).round(1)
    # else:
    #     issue_freq = pd.Series(dtype='float')
    # return issue_counts, unique_issue_names, issue_freq, df


def plot_distribution_for_full_and_filtered(df_full, full_issue_freq, full_issue_count, issues_filtered_df):
    st.header("Comparison of Issue Frequencies:")
    # _, _, filtered_issue_freq, _ = get_issue_analysis(issues_filtered_df)
    issues_stats = get_issue_analysis(issues_filtered_df)
    filtered_issue_freq = dict(issues_stats["issue_freq"])
    # Convert to Series
    full_freq = pd.Series(full_issue_freq)
    subset_freq = pd.Series(filtered_issue_freq).reindex(full_freq.index, fill_value=0)

    # Combine and exclude 'NO_ISSUE'
    df_comp = pd.DataFrame({
        "Full Dataset": full_freq,
        "Filtered Subset": subset_freq
    }).drop(index=NO_ISSUE, errors="ignore")

    df_sorted = df_comp.sort_values("Full Dataset", ascending=False)

    def wrap_label(label, width=40):
        trimmed_label = ' '.join(label.split(' ')[:14])
        if len(trimmed_label) < len(label):
            trimmed_label += "..."
        return '\n'.join(textwrap.wrap(trimmed_label, width))

    # Wrap long labels
    def wrap_labels(labels, width=40):
        return [wrap_label(label, width) for label in labels]

    wrapped_labels = wrap_labels(df_sorted.index.tolist(), width=40)

    # Plot setup
    fig, ax = plt.subplots(figsize=(6.5, 5))
    bar_width = 0.35
    y_positions = range(len(df_sorted))

    # Get 2 colors from the 'mako' palette
    colors = sns.color_palette("mako", n_colors=2)
    full_color, subset_color = colors

    # Bars
    ax.barh(
        [y - bar_width / 2 for y in y_positions],
        df_sorted["Full Dataset"],
        height=bar_width,
        label="Full Dataset",
        color=full_color
    )
    ax.barh(
        [y + bar_width / 2 for y in y_positions],
        df_sorted["Filtered Subset"],
        height=bar_width,
        label="Filtered Subset",
        color=subset_color
    )

    for y, value in zip(y_positions, df_sorted["Full Dataset"]):
        ax.text(
            value + 0.5,  # just right of the bar
            y - bar_width / 2,  # align with bar center
            f"{value}%",  # formatted label
            va='center',
            ha='left',
            fontsize=5,
            #color=full_color
        )

    for y, value in zip(y_positions, df_sorted["Filtered Subset"]):
        ax.text(
            value + 0.5,
            y + bar_width / 2,
            f"{value}%",
            va='center',
            ha='left',
            fontsize=5,
            #color=subset_color
        )

    # Axis and label formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(wrapped_labels, fontsize=5)
    ax.set_xlim(0, max(df_sorted.max()) * 1.2)
    ax.set_xlabel("Frequency (%)", fontsize=9)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=8)

    # Horizontal gridlines (subtle)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', color='#dddddd', linewidth=0.6)

    # Legend
    ax.legend(
        fontsize=7,
        loc='lower right',
        frameon=True,  # ✅ show legend box
        fancybox=True,  # ✅ rounded box
        framealpha=0.9,  # ✅ slightly transparent
        edgecolor='gray',  # ✅ subtle border color
        borderpad=0.5)

    plt.tight_layout()
    st.pyplot(fig)

def score_to_hex(score):
        # Create a red-to-green colormap
        cmap = cm.get_cmap('RdYlGn')

        # Normalize the score (already between 0 and 1)
        norm = mcolors.Normalize(vmin=0, vmax=1)

        # Get RGBA color
        rgba = cmap(norm(score))

        # Convert RGBA to HEX
        return mcolors.to_hex(rgba)

def get_scaled_fraction(count, sqrt_max):
        return math.sqrt(count) / sqrt_max if sqrt_max > 0 else 0



def get_table_html(issues_stats, sorted_issues_freq, sqrt_max):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        table { 
            width: 100%;
            border-collapse: collapse;
            font-family: sans-serif;
        }
        th, td {
            padding: 8px;
            text-align: left;
            vertical-align: top;
            border-bottom: 1px dashed #ccc;  
        }
        th {
            border-bottom: 2px solid #ccc;
        }
        .bar-container {
            width: 100px;
            height: 12px;
            background-color: #eee;
            border-radius: 6px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-right: 8px;
        }
        .bar-fill {
            height: 100%;
            background-color: #1f77b4;
        }
    </style>
    </head>
    <body>
    <table>
        <thead>
            <tr>
                <th style="text-align:left;">Issue</th>
                <th style="text-align:center;">Count</th>
                <th style="text-align:center;">Frequency (%)</th>
                <th style="text-align:center;">Severity</th>
                <th></th>
            </tr>
        </thead>
        <tbody>
    """

    max_bar_px = 300
    for issue, freq in sorted_issues_freq:
        count = issues_stats.loc[issue].get("issue_count", 0)
        freq = issues_stats.loc[issue].get("issue_freq", 0.0)
        mean_score = issues_stats.loc[issue].get("mean", 0)
        severity = (1 - mean_score)
        scaled_frac = get_scaled_fraction(count, sqrt_max)
        bar_width = int(scaled_frac * max_bar_px)
        hex_color = score_to_hex(mean_score)

        if issue != NO_ISSUE:
            severity_html = f"""<td style="text-align:center;">{severity:.2f}</td>
                            <td>
                               <div style="height: 12px; background: {hex_color};
                                    width: {bar_width}px; border-radius: 5px;
                                    margin-top: 4px;">
                             </td>"""
        else:
            severity_html =          progress_cell = """
            <td style="text-align:center;">-</td>
            <td><div style="display: flex; flex-direction: row; gap: 4px;">
                <span style="font-size: 11px; white-space: nowrap;">Severity</span>
                <div style="position: relative; height: 12px; width: 200px;
                            background: linear-gradient(to right, green, yellow, red);
                            border-radius: 5px;">
                    <div style="position: absolute; top: 14px; left: 0; font-size: 10px;">0</div>
                    <div style="position: absolute; top: 14px; right: 0; font-size: 10px;">1</div>
                </div>
            </div>
            </td>
        """

        html += f"""
                    <tr>
                  <td class="wrap">{issue}</td>
                        <td style="text-align:center;">{count:.0f}</td>
                        <td style="text-align:center;">{freq:.1f}</td>
                        {severity_html}         
                        </tr>
                    """

    html += """
        </tbody>
    </table>
    </body>
    </html>
    """
    row_height = 50
    header_height = 60
    table_height = header_height + len(issues_stats) * row_height

    components.html(html, height=table_height, scrolling=True)


def list_issues_frequency(issues_stats):
    st.write("Frequency of each issue (sorted high to low):")
    issue_counts = dict(issues_stats["issue_count"])
    issue_freq = issues_stats["issue_freq"]

    included_counts = [v for k, v in issue_counts.items() if k != NO_ISSUE]
    sqrt_max = math.sqrt(max(included_counts))

    sorted_issues_freq = sorted(issue_freq.items(), key=lambda x: issue_freq.get(x[0], 0), reverse=True)

    # Split out the target and the rest
    no_issue_item = [item for item in sorted_issues_freq if item[0] == NO_ISSUE]
    other_items = [item for item in sorted_issues_freq if item[0] != NO_ISSUE]

    # Combine with 'no_issue' first
    sorted_issues_freq = no_issue_item + other_items

    get_table_html(issues_stats, sorted_issues_freq, sqrt_max)


def perform_instance_filtering_by_issue_and_score(df_full, issue_counts, issue_freq):
    defaults = {
        "include": [],
        "must_have": [],
        "exclude": [],
        "apply_clicked": False,
        "only": False,
        "score_range":  (0.0, 1.0),
        "clear_trigger": False,
    }
    # --- Session State Defaults ---
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "widget_key_suffix" not in st.session_state:
        st.session_state["widget_key_suffix"] = 0

    if st.session_state.get("clear_trigger", False):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.session_state["widget_key_suffix"] += 1
        st.rerun()

    # --- UI in Main Column ---
    all_issues = list(issue_freq.keys())

    # --- UI Layout ---
    suffix = st.session_state["widget_key_suffix"]
    explanation = """Use the filters below to narrow down instances based on issues and score:"""
    include, must_have, exclude, only = show_issues_selection_buttons(all_issues, suffix, explanation)
    if 'score' in df_full.columns and not df_full['score'].empty:
        score_range = st.slider(
            "Select score range",
            0.0, 1.0,
            value=st.session_state.score_range,
            step=0.01,
            help = "Only include rows within this score range.",
            key=f"score_slider_{st.session_state.widget_key_suffix}",  # ← dynamic key
        )
    else:
        score_range = (0.0, 1.0)

    # --- Buttons ---
    _, button_col1, button_col2, _ = st.columns([3, 1, 1, 3])
    with button_col1:
        apply = st.button("✅ Apply Filter")
    with button_col2:
        clear = st.button("🧹 Clear Filters")

    # --- Handle Apply ---
    if apply:
        st.session_state.include = include.copy()
        st.session_state.must_have = must_have.copy()
        st.session_state.exclude = exclude.copy()
        st.session_state.score_range = tuple(score_range)
        st.session_state.apply_clicked = True
        st.session_state.only = only

    # --- Handle Clear ---
    if clear:
        st.session_state.clear_trigger = True
        st.rerun()

    # --- Filtering Logic ---
    if st.session_state.apply_clicked:
        def issue_filter(text_issues_str):
            issues = extract_issues(text_issues_str)

            # OR logic (any of the included issues)
            if st.session_state.include:
                if not any(i in issues for i in st.session_state.include):
                    return False

            # AND logic (must contain all required)
            if not all(i in issues for i in st.session_state.must_have):
                return False

            # NOT logic (must not contain any)
            if any(i in issues for i in st.session_state.exclude):
                return False

            # "All the rest" logic: no extra issues beyond include + must_have
            if st.session_state.only:
                allowed = set(st.session_state.include + st.session_state.must_have)
                if any(i not in allowed for i in issues):
                    return False

            return True

        issues_filtered_df = df_full[
                df_full.apply(lambda r: issue_filter(r["recurring_issues_str"]), axis=1)
            ]
        if score_range != (0.0, 1.0):
            issues_filtered_df = issues_filtered_df[issues_filtered_df["score"].between(*st.session_state.score_range)]

        # --- Query Summary ---
        st.markdown("Filter Summary")
        summary_parts = []
        if st.session_state.include:
            summary_parts.append(f"ANY of **{st.session_state.include}**")
        if st.session_state.must_have:
            summary_parts.append(f"ALL of **{st.session_state.must_have}**")
        if st.session_state.exclude:
            summary_parts.append(f"NOT any of **{st.session_state.exclude}**")
        if st.session_state.only:
            summary_parts.append("Only selected issues")

        summary_parts.append(
            f"Score between **{st.session_state.score_range[0]} and {st.session_state.score_range[1]}**")
        st.markdown(" - " + "\n - ".join(summary_parts))

        st.info(f"Found {len(issues_filtered_df)} entries matching entries")

    else:
        issues_filtered_df = df_full.copy()
        st.info(f"Showing all **{len(issues_filtered_df)}** entries (no filters applied)")

    return issues_filtered_df

def write_qa_header():
   # if usecase == GENERAL_USECASE:
    st.title("CLEAR: Comprehensive LLM Error Analysis and Reporting")
    # else:
    #     st.title(f"LLM-Critique: Evaluation Results Analyzer {usecase})")

    st.markdown("""
        <div style="
            border: 1px solid #ccc;
            border-radius: 10px;
            padding: 1.2em;
            background-color: #f9f9f9;
            text-align: left;
            font-size: 1.3em;
            line-height: 1.3em;
            margin-bottom: 1.5em;
        ">
            <strong>Easily explore and filter your dataset based on discovered issues and score.</strong><br><br>
            🔍 <strong>What you can do here:</strong><br><br>
            📊 <strong>View recurring problems</strong> discovered in your dataset<br>
            🎯 <strong>Filter</strong> rows by issue type and score range<br>
            📈 <strong>See stats</strong> and score distribution for filtered data<br>
            🧵 <strong>Drill down</strong> into individual examples by clicking on rows
        </div>
        """, unsafe_allow_html=True)

def show_data_explorer_select_index(issues_filtered_df, total_examples, format_row_func,
                                    usecase="GENERAL_USECASE"):
    st.header("📊 Data Explorer")
    default_columns = get_input_columns(st.session_state.metadata) + DISPLAY_COLS

    df_to_present = issues_filtered_df[[c for c in default_columns if c in issues_filtered_df.columns]]

    st.write(f"Found {len(df_to_present)}/{total_examples} matching input examples.")
    if len(df_to_present) > max_presented_examples:
        st.write(f"Displaying first {max_presented_examples} input examples.")
        df_to_present = df_to_present.head(max_presented_examples)
    available_indices_issue_filtered = df_to_present.index.tolist()

    st.markdown("""
      📋 Please select an entry from the table or dropdown:

    - 🧾 **From the table:** Click the **grey strip to the left** of the first column to select a row.
    - 🔽 **From the dropdown:** Select an option or start typing the desired **index or question**.
    """)
    if "selection_source" not in st.session_state:
        st.session_state.selection_source = None
    if "selection_id" not in st.session_state:
        st.session_state.selected_id = None
    if "question_selector" not in st.session_state:
        st.session_state.question_selector = None
    if "df_key" not in st.session_state:
        st.session_state.df_key = f"df_{usecase}_0"
    if "df_counter" not in st.session_state:
        st.session_state.df_counter = 0

    # --- Clear Selection Button ---
    def clear_selection():
        st.session_state.selected_id = None
        st.session_state.selection_source = None
        st.session_state.question_selector = None
        st.session_state.df_counter += 1
        st.session_state.df_key = f"df_{usecase}_{st.session_state.df_counter}"

    if st.button("Clear selection"):
        clear_selection()

    event = st.dataframe(
        df_to_present,
        use_container_width=True,
        key=st.session_state.df_key,
        on_select="rerun",
        selection_mode="single-row"
    )

    # Check if a row is selected
    if event:
        if event.selection.rows:
            idx = event.selection.rows[0]
            selected_index = available_indices_issue_filtered[idx]
            if st.session_state.selected_id != selected_index:
                st.session_state.selected_id = selected_index
                st.session_state.selection_source = "table"
                st.session_state.question_selector = None  # clear dropdown
        elif st.session_state.selection_source == "table":
            # Manual deselection of table row
            clear_selection()

    disable_dropdown = st.session_state.selection_source == "table"
    # Determine the index to show in dropdown
    dropdown_index = (
        available_indices_issue_filtered.index(st.session_state.question_selector)
        if st.session_state.question_selector in available_indices_issue_filtered else 0
    )
    dropdown_disabled = st.session_state.selection_source == "table"

    if st.session_state.question_selector not in df_to_present.index:
        st.session_state.question_selector = None
        # Force dropdown re-render
        if "dropdown_key_counter" not in st.session_state:
            st.session_state.dropdown_key_counter = 0
        st.session_state.dropdown_key_counter += 1

    st.selectbox(
        "Or select an entry to view details:",
        options=available_indices_issue_filtered,
        format_func=format_row_func,
        index=dropdown_index,
        placeholder="Choose an entry",
        key="question_selector",
        disabled=disable_dropdown,
    )

    if st.session_state.question_selector is not None and not disable_dropdown:
        st.session_state.selection_source = "dropdown"
        st.session_state.selected_id = st.session_state.question_selector

    if st.session_state.selected_id is not None:
        selected_row = issues_filtered_df.loc[st.session_state.selected_id]
    else:
        selected_row = None

    if dropdown_disabled:
        st.warning(
            f"You've selected row {st.session_state.selected_id} from the table. Clear the table selection to use the dropdown.")
    return selected_row, st.session_state.selected_id

def qa_instance_row_format(x):
    issues_filtered_df = st.session_state.get("issues_filtered_df", pd.DataFrame())
    question_col = st.session_state.metadata.get("question_column", "question")
    if not question_col in issues_filtered_df.columns:
        question_col = "model_input"
    return f"Entry Index: {x} - Q: " \
           f"{str(issues_filtered_df.loc[x, question_col])[:50]}..."


def print_experiment_metadata():
    metadata = st.session_state.get("metadata", {})
    st.sidebar.title(f"Metadata:")
    st.sidebar.markdown(f"Run Name: `{metadata.get('run_name', 'N/A')}`")
    st.sidebar.markdown(f"Model: `{metadata.get('gen_model_name', 'N/A')}`")
    st.sidebar.markdown(f"Judge: `{metadata.get('eval_model_name', 'N/A')}`")


def display_qa_style_analysis(instance_format_func=qa_instance_row_format):
    write_qa_header()
    file_to_load, file_id = get_uploaded_file()

        # test if first run or data selection changed : reload data
    if 'current_file_id' not in st.session_state or file_id != st.session_state.current_file_id:
        file_bytes = file_to_load.read()
        file_name = file_to_load.name
        df_full, metadata = load_data(file_bytes, file_name)
        st.session_state.current_file_id = file_id
        st.session_state.issues_filtered_df = df_full.copy()
        st.session_state.full_df = df_full.copy()
        st.session_state.metadata = metadata.copy()
        st.session_state.include = []
        st.session_state.must_have = []
        st.session_state.exclude = []
        st.session_state.score_range = tuple((0.0, 1.0))
        st.session_state.apply_clicked = False
        st.session_state.only = False
        st.session_state.question_selector = None
        st.session_state["widget_key_suffix"] = st.session_state.get("widget_key_suffix", -1) + 1
    else:
        df_full = st.session_state.full_df
    
    # print metadata
    print_experiment_metadata()
    if df_full.empty:
        st.warning("No data loaded for analysis. Please check the CSV file and its path.")
        return

    # issue_counts, unique_issue_names, issue_freq, _ = get_issue_analysis(df_full, MAX_NUM_ISSUES)
    issues_stats = get_issue_analysis(df_full, MAX_NUM_ISSUES)

    st.success(f"Successfully loaded {len(df_full)} records")
    # if not issue_freq.empty:
    #     plot_issue_freq(dict(issue_freq))

    st.header("Scores Distribution")
    if 'score' in df_full.columns and not df_full['score'].empty:
        with st.expander("View Distribution of Scores"):
            st.bar_chart(df_full["score"].value_counts().sort_index())
    else:
        st.write("Score data is not available or empty.")

    st.header("Issues Distribution")
    total_evals = len(df_full)
    st.write(f"Total evaluations processed: {total_evals}")

    if not issues_stats.empty:
        issue_freq = dict(issues_stats['issue_freq'])
        issue_counts =  dict(issues_stats['issue_count'])
        list_issues_frequency(issues_stats)
        st.session_state.issues_filtered_df = perform_instance_filtering_by_issue_and_score(df_full, issue_counts, issue_freq)
        if len(st.session_state.issues_filtered_df) > 0 and not st.session_state.issues_filtered_df.empty:
            plot_distribution_for_full_and_filtered(df_full, issue_freq, issue_counts, st.session_state.issues_filtered_df)

            selected_row, selected_id = show_data_explorer_select_index(st.session_state.issues_filtered_df,
                                                                        len(df_full), format_row_func=instance_format_func)
            # ---- Show Selection ----
            if selected_row is not None:
                show_instance_results(selected_row, selected_id)
            else:
                st.info("Please select a question from the table or dropdown.")

        else:  # This 'else' is for 'if not issues_filtered_df.empty:'
            st.write("No entries found for the selected issue(s).")

    else:  # This 'else' is for 'if not issue_counts.empty:'
        st.write("No recurring issues found or 'recurring_issues_str' column is missing/empty.")

    if df_full.empty and 'file_to_load' in locals():
        st.info(f"Attempted to load file`. Please ensure it exists and is correctly formatted.")


def file_from_path(path):
    with open(path, "rb") as f:
        content = f.read()
    bio = BytesIO(content)
    bio.name = os.path.basename(path)  # Optional: match file_uploader
    return bio

def file_hash(file_obj):
    return hashlib.md5(file_obj.getbuffer()).hexdigest()

def get_uploaded_file(uploader_key="zip_uploader"):
    # File upload widget in sidebar
    msg = "📁 Upload the zip file generated by running the pipeline to visualize results"
    uploaded_file = st.sidebar.file_uploader(msg, type="zip", key=uploader_key)

    if uploaded_file is not None:
        st.sidebar.info("❌ Press the X next to the uploaded file to go back to built-in configuration selection")

    results_dir = os.getenv("CLEAR_EVAL_RESULTS_DIR")
    if results_dir and not uploaded_file:
        # If no file uploaded, show select boxes (internal file options)
        st.sidebar.markdown("Or select a built-in configuration:")
        options = [d for d in os.listdir(results_dir) if \
                   os.path.isdir(os.path.join(results_dir, d))]
        selected_dataset_name = st.sidebar.radio(
            "Choose a usecase to Analyze:",
            options=options,
            key=f"use_case_selection"
        )
        if selected_dataset_name:
            use_case_results_dir = os.path.join(results_dir, selected_dataset_name)
            data_options = os.listdir(use_case_results_dir)
            data_options = ["None"] + [d for d in data_options if
                            os.path.isdir(os.path.join(use_case_results_dir, d))
                                       and d !="final_results"]
            selected_dataset = st.sidebar.selectbox(
                "Choose dataset:",
                options=data_options,
                key=f"dataset_selection_{use_case_results_dir}"
            )
            if selected_dataset != "None":
                data_results_dir = os.path.join(use_case_results_dir, selected_dataset)
                for f in os.listdir(data_results_dir):
                    if f.endswith(".zip") or f.endswith(".parquet"):
                        file_path = os.path.join(data_results_dir, f)
                        uploaded_file = file_from_path(file_path)
    if uploaded_file is not None:
        # Show file info
        file_id = file_hash(uploaded_file)
        st.sidebar.success(f"Loaded file: {uploaded_file.name}")
        return uploaded_file, file_id

    for key in ("current_file_id", "full_df", "metadata"):
        st.session_state.pop(key, None)
    st.warning("⚠️ Please upload a file to proceed.")
    st.stop()

def fix_illegal_json(data):
    """
    Fixes messy pseudo-JSON:
    - Handles single quotes, Python-style values
    - Escapes inner double quotes in string values
    - Returns a valid JSON-compatible dict
    """
    # Already a dict/list? Return as-is
    if isinstance(data, (dict, list)):
        return data

    # Step 1: Try loading it directly as JSON
    try:
        return json.loads(data)
    except:
        pass

    # Step 2: Try to parse Python-like dict using ast.literal_eval
    try:
        py_obj = ast.literal_eval(data)
        return json.loads(json.dumps(py_obj))  # Ensure JSON-safe
    except:
        pass

    # Step 3: Replace True/False/None with JSON equivalents
    fixed = (
        data.replace("None", "null")
            .replace("True", "true")
            .replace("False", "false")
    )

    # Step 4: Last-ditch fix — use regex to fix inner quotes in values
    def escape_quotes_in_values(match):
        key = match.group(1)
        val = match.group(2)
        val = val.replace('"', r'\"').replace("'", r"\'")
        return f'"{key}": "{val}"'

    # Replace single-quoted keys/values
    fixed = re.sub(r"'([^']+)'\s*:\s*'([^']*)'", escape_quotes_in_values, fixed)
    fixed = re.sub(r"'([^']+)'\s*:\s*\"([^\"]*)\"", escape_quotes_in_values, fixed)

    try:
        return json.loads(fixed)
    except Exception as e:
        return None


def print_json_fallback_string(val):
    try:
        try:
            parsed = json.loads(val)
        except json.JSONDecodeError:
            parsed = fix_illegal_json(val)
        if parsed:
            st.json(parsed)
        else:
            st.markdown(val)

    except Exception:
        st.markdown(val)

def show_issues_selection_buttons(all_issues, suffix, explanation):
    st.header("🔍 Select Filters")
    st.markdown(explanation)
    st.markdown("Click **Apply Filter** to update the results or **Clear Filters** to reset everything.")

    col1, col2, col3 = st.columns(3)
    with col1:
        include = st.multiselect("Include ANY of (OR)", all_issues, key=f"include_temp_bfcl_{suffix}", help="Show rows with at least one of these issues")
    with col2:
        must_have = st.multiselect("Must ALSO have (AND)", all_issues, key=f"must_temp_bfcl_{suffix}", help="Rows must contain all of these issues")
    with col3:
        exclude = st.multiselect("Exclude ANY of (NOT)", all_issues, key=f"exclude_temp_bfcl_{suffix}", help = "Remove rows with any of these issues")

    only = st.checkbox("Select instances with failures in ONLY the above issues", key=f"only_temp__bfcl_{suffix}",  help="Exclude rows that have other issues beyond what you selected above")
    return include, must_have, exclude, only


def get_as_list(issues):
    if issues is None:
        return []
    if isinstance(issues, list):
        return issues
    if isinstance(issues, str):
        try:
            return ast.literal_eval(issues)
        except Exception as e:
            return []
    return []

def write_recurring_issues(selected_row):
    st.markdown("**Recurring Issues:**")
    recurring_issues =selected_row.get('recurring_issues_str')
    recurring_issues = get_as_list(recurring_issues)
    if recurring_issues:
        for r in recurring_issues:
            st.write(f"- {r}")
    else:
        st.write("N/A")

    try:
        other_recurring_issues = selected_row.get('recurring_issues_other_str')
        other_recurring_issues = get_as_list(other_recurring_issues)
        if other_recurring_issues:
            st.markdown("**Other Recurring Issues:**")
            for r in other_recurring_issues:
                st.write(f"- {r}")
    except Exception as e:
        pass

def plot_issue_freq(issue_to_freq, x_col='Issue', y_col='Frequency (%)'):
    st.header('Recurring issues discovered in the data and their frequencies')
    issue_freq_df =  pd.DataFrame(list(issue_to_freq.items()), columns=[x_col, y_col])
    issue_freq_df = issue_freq_df[issue_freq_df[x_col] != NO_ISSUE]
    fig_math, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=issue_freq_df, x=x_col, y=y_col)
    ax.set_ylabel(y_col)
    ax.set_xlabel(x_col)
    plt.xticks(rotation=45, ha='right')
    fig_math.tight_layout()
    st.pyplot(fig_math)


def get_input_columns(metadata):
    return metadata.get('input_columns', ["question","documents"])


def show_instance_results(elected_row, selected_index):
        st.write(f"Details for Math Entry Index: {selected_index}")

        input_columns = get_input_columns(st.session_state.metadata)
        for column in input_columns:
            if column in elected_row:
                with st.expander(f"**{column}:**"):
                    st.markdown(elected_row[column])

        with st.expander("**Response:**"):
            print_json_fallback_string(elected_row.get('response', 'N/A'))

        with st.expander("**Model Input (Prompt):**"):
            st.text(elected_row.get('model_input', 'N/A'))

        with st.expander("**Full Evaluation Text:**"):
            st.text_area("Eval Text", elected_row.get('evaluation_text', 'N/A'), height=150,
                     key=f"math_evaltext_explore_{selected_index}")  # Unique key

        evaluation_summary = elected_row.get('evaluation_summary')
        if evaluation_summary and not pd.isna(evaluation_summary):
            st.markdown("**Evaluation Summary:**")
            st.text(evaluation_summary)

        ground_truth = elected_row.get('ground_truth')
        if ground_truth and not pd.isnull(ground_truth):
            st.markdown(f"**Ground Truth:**")
            st.text(ground_truth)

        st.markdown(f"**Score:** {elected_row.get('score', 'N/A')}")

        write_recurring_issues(elected_row)


def show_dashboard():
    st.set_page_config(layout="wide")
    display_qa_style_analysis()