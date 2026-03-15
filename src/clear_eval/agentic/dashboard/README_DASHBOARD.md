# Agentic Workflow Dashboard

A comprehensive NiceGUI dashboard for visualizing and analyzing multi-agent system trajectories and CLEAR evaluation results.

## Features

### 1. Workflow View
- Interactive graph visualization showing all agents and their relationships
- Node statistics (call counts, unique tasks, tool vs agent calls)
- Edge weights showing transition frequencies
- Select agents from dropdown to highlight and see detailed statistics
- Trajectory length distribution and score correlation

### 2. Node-Specific CLEAR Analysis
- Full CLEAR analysis results for each agent
- Score distribution visualization
- Issue frequency table with severity indicators
- Interactive data explorer with drill-down into individual records
- Full evaluation details including input, response, evaluation text, and recurring issues

### 3. Trajectory Explorer
- Browse individual trajectories by task ID
- Advanced filtering by trajectory length, agents, and score range
- View complete agent flow for each trajectory
- Step-by-step execution details with inputs/outputs
- Score progression visualization across trajectory steps
- Metadata inspection

### 4. Path Analysis
- Identify most common paths through the workflow
- Path frequency distribution charts
- Success vs failure pattern analysis
- Dead-end node detection

### 5. Temporal Analysis
- Agent position analysis (where do agents appear in trajectories)
- Score progression analysis through trajectory steps
- Agent retry pattern detection and visualization

## Installation

### Prerequisites

```bash
# Install required packages
pip install nicegui plotly networkx pandas numpy
```

The dashboard also requires the CLEAR package to be installed.

## Usage

### Quick Start

```bash
# From the project root
python -m clear_eval.agentic.dashboard.agentic_workflow_dashboard
```

Or use the launcher:

```bash
python -m clear_eval.agentic.dashboard.launch_dashboard
```

### Options

```bash
python -m clear_eval.agentic.dashboard.launch_dashboard --port 8080 --host 0.0.0.0 --no-open
```

| Flag | Description | Default |
|------|-------------|---------|
| `--port` | Port to run the dashboard on | 8080 |
| `--host` | Host to bind to | 0.0.0.0 |
| `--no-open` | Don't auto-open browser | False |

### How to Use

1. **Launch the dashboard** using one of the methods above
2. **Upload** a `ui_results.zip` file using the sidebar upload widget
3. **Explore** the five analysis tabs:
   - **Workflow View**: See the overall agent graph and trajectory statistics
   - **Node Analysis**: Dive into CLEAR evaluation results for each agent
   - **Trajectory Explorer**: Browse individual agent trajectories step-by-step
   - **Path Analysis**: Discover common paths and success/failure patterns
   - **Temporal Analysis**: Analyze agent position patterns and retry behavior

## Data Format

### Expected ZIP Structure

The dashboard expects a `ui_results.zip` file with the following structure:

```
ui_results.zip
├── metadata.json           # Overall metadata
├── trajectory_data.zip     # Trajectory CSV files (or trajectory_data/ folder)
│   ├── trace_1.csv
│   ├── trace_2.csv
│   └── ...
└── agent_results/          # CLEAR analysis results per agent
    ├── agent1.zip
    ├── agent2.zip
    └── ...
```

### Trajectory Data Columns
- `id`: Unique identifier for each call
- `Name`: Agent name
- `intent`: User's original query/intent
- `task_id`: Trajectory identifier
- `step_in_trace_general`: Step number in trajectory
- `step_in_trace_node`: Step number within node
- `model_input`: Input to the agent/tool
- `response`: Output from the agent/tool
- `tool_or_agent`: Type of call ('tool' or 'agent')
- `meta_data`: JSON metadata (tokens, latency, cost, etc.)

## Technology Stack

- **NiceGUI**: Modern Python web UI framework (Quasar/Vue.js under the hood)
- **Plotly**: Interactive chart visualizations
- **NetworkX**: Graph analysis and layout
- **Pandas**: Data processing and manipulation

## Tips

1. **Performance**: For large datasets, the initial load may take a moment. Data is processed once on upload.

2. **Graph Visualization**:
   - Larger nodes = more calls
   - Thicker edges = more transitions
   - Orange nodes = selected
   - Blue nodes = unselected

3. **Filtering**: In Trajectory Explorer, use the Advanced Filters panel to narrow down by length, agent, or score range.

4. **Data Explorer**: Click on rows in the Node Analysis data table to see full record details.

## Troubleshooting

### "No data loaded"
- Check that the uploaded ZIP file follows the expected structure
- Ensure trajectory CSV files exist in the ZIP

### "No CLEAR analysis results found"
- Make sure the CLEAR analysis has been run before creating the ZIP
- Check that agent_results/ contains valid ZIP files

### Import errors
- Ensure all dependencies are installed: `pip install nicegui plotly networkx pandas numpy`
- Make sure CLEAR is installed: `pip install -e .` from the CLEAR root directory
