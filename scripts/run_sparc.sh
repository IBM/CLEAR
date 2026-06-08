#!/usr/bin/env bash
# Run CLEAR + SPARC tool-calling evaluation on the three converted CSVs
# (tau2_retail, tau2_airline, appworld) using watsonx/rits + openai/gpt-oss-120b.
#
# Per-benchmark input CSVs live under scripts/runs/input/<bench>/<agent_type>
# Results go to
# "scripts/runs/output/$bench/$agent_type/${track}_${mode}"
# don't clobber each other.
#
# Usage:
#   ./run_sparc.sh                                           # slow_track, all benches & agent types
#   ./run_sparc.sh tau2_retail                               # slow_track, specific bench, all agent types
#   ./run_sparc.sh appworld --agent-types agent1 agent2      # specific bench & agents
#   ./run_sparc.sh appworld --agent-types agent1 agent2 --track fast_track
set -euo pipefail

here="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root="$(dirname "$here")"
cd "$root"

track="slow_track"
mode="runtime"   # runtime = fast prompts / no recommendations; eval = recs on
agent_types=()   # list of agent types to process
benches=()       # list of benchmarks to process

# Parse all arguments (options and positional)
while [ $# -gt 0 ]; do
  case "$1" in
    --track)     track="$2";  shift 2 ;;
    --track=*)   track="${1#--track=}"; shift ;;
    --mode)      mode="$2";   shift 2 ;;
    --mode=*)    mode="${1#--mode=}";   shift ;;
    --agent-types)
      shift
      # Collect all following non-option arguments as agent types
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
        agent_types+=("$1")
        shift
      done
      ;;
    --*)
      echo "error: unknown option '$1'" >&2
      exit 2
      ;;
    *)
      # Positional argument (benchmark name)
      benches+=("$1")
      shift
      ;;
  esac
done

case "$track" in
  slow_track|fast_track|spec_free|syntax|transformations_only) ;;
  *) echo "error: invalid --track '$track'" >&2; exit 2 ;;
esac
case "$mode" in
  runtime|eval) ;;
  *) echo "error: invalid --mode '$mode' (use runtime|eval)" >&2; exit 2 ;;
esac
# `runtime_pipeline: true` == runtime mode; `false` == eval mode.
if [ "$mode" = "runtime" ]; then
  rp=true
  issues_format="shortcomings"
else
  rp=false
  issues_format="recommendations"
fi

if [ ! -f .env ]; then
  echo "error: $root/.env not found" >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; source .env; set +a

if [ ! -d .venv ]; then
  echo "error: $root/.venv not found (run: python3.12 -m venv .venv && pip install -e .[tool-calls])" >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# Detect provider from sparc_config.yaml and validate required credentials
provider=$(grep "^provider:" scripts/sparc_config.yaml | awk '{print $2}' | tr -d '"' | tr -d "'")
if [ -z "$provider" ]; then
  echo "error: provider not found in scripts/sparc_config.yaml" >&2
  exit 1
fi

case "$provider" in
  watsonx)
    : "${WATSONX_APIKEY?missing WATSONX_APIKEY}"
    : "${WATSONX_URL?missing WATSONX_URL}"
    : "${WATSONX_PROJECT_ID?missing WATSONX_PROJECT_ID}"
    ;;
  rits)
    : "${RITS_API_KEY?missing RITS_API_KEY}"
    ;;
  *)
    echo "error: unsupported provider '$provider' in sparc_config.yaml" >&2
    echo "       supported providers: watsonx, rits" >&2
    exit 2
    ;;
esac

# Build a track-specific config file by overriding the `track:` line in
# sparc_config.yaml. Keeps the base YAML as the single source of truth.
# (macOS mktemp -t does not respect a trailing extension in the template,
# so construct the path explicitly and clean it up on exit.)
tmp_cfg="${TMPDIR:-/tmp}/sparc_config.$$.$RANDOM.yaml"
trap 'rm -f "$tmp_cfg"' EXIT
python - "$track" "$rp" "$tmp_cfg" <<'PY'
import sys, re, pathlib
track, rp, out = sys.argv[1], sys.argv[2], sys.argv[3]
src = pathlib.Path("scripts/sparc_config.yaml").read_text()
if re.search(r"^track\s*:", src, re.MULTILINE):
    src = re.sub(r"^track\s*:.*$", f"track: {track}", src, count=1, flags=re.MULTILINE)
else:
    src = src.rstrip() + f"\ntrack: {track}\n"
if re.search(r"^runtime_pipeline\s*:", src, re.MULTILINE):
    src = re.sub(r"^runtime_pipeline\s*:.*$", f"runtime_pipeline: {rp}", src, count=1, flags=re.MULTILINE)
else:
    src = src.rstrip() + f"\nruntime_pipeline: {rp}\n"
pathlib.Path(out).write_text(src)
PY

# If no benchmarks specified, discover all from input directory
if [ ${#benches[@]} -eq 0 ]; then
  for bench_dir in scripts/runs/input/*/; do
    [ -d "$bench_dir" ] || continue
    benches+=("$(basename "$bench_dir")")
  done
  if [ ${#benches[@]} -eq 0 ]; then
    echo "error: no benchmarks found in scripts/runs/input/" >&2
    exit 1
  fi
fi

for bench in "${benches[@]}"; do
  # If agent_types not specified via --agent-type, discover from input directory
  if [ ${#agent_types[@]} -eq 0 ]; then
    discovered_agent_types=()
    for agent_dir in scripts/runs/input/"$bench"/*/; do
      [ -d "$agent_dir" ] || continue
      discovered_agent_types+=("$(basename "$agent_dir")")
    done
    if [ ${#discovered_agent_types[@]} -eq 0 ]; then
      echo "warning: no agent types found for benchmark '$bench', skipping" >&2
      continue
    fi
    current_agent_types=("${discovered_agent_types[@]}")
  else
    current_agent_types=("${agent_types[@]}")
  fi

  for agent_type in "${current_agent_types[@]}"; do
    data_dir="scripts/runs/input/$bench/$agent_type/"
    results_dir="scripts/runs/output/$bench/$agent_type/${track}_${mode}"

    if [ ! -d "$data_dir" ]; then
      echo "warning: input directory not found: $data_dir, skipping" >&2
      continue
    fi

    echo "=== running SPARC ($track, $mode) on $bench / $agent_type ==="
    python -m clear_eval.agentic.pipeline.run_clear_step_analysis \
      --agentic-config-path "$tmp_cfg" \
      --data-dir "$data_dir" \
      --results-dir "$results_dir" \
      --issues-format="$issues_format"
  done
done
