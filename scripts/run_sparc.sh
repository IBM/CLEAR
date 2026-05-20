#!/usr/bin/env bash
# Run CLEAR + SPARC tool-calling evaluation on the three converted CSVs
# (tau2_retail, tau2_airline, appworld) using watsonx + openai/gpt-oss-120b.
#
# Per-benchmark input CSVs live under scripts/runs/<bench>/input/. Results go to
# scripts/runs/<bench>/output_<track>/<bench>/ so slow- and fast-track passes
# don't clobber each other.
#
# Usage:
#   ./run_sparc.sh                       # slow_track, all 3 benches
#   ./run_sparc.sh --track fast_track    # fast_track, all 3 benches
#   ./run_sparc.sh tau2_airline          # slow_track, single bench
#   ./run_sparc.sh --track fast_track tau2_retail appworld
set -euo pipefail

here="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
root="$(dirname "$here")"
cd "$root"

track="slow_track"
mode="runtime"   # runtime = fast prompts / no recommendations; eval = recs on
while [ $# -gt 0 ]; do
  case "$1" in
    --track)     track="$2";  shift 2 ;;
    --track=*)   track="${1#--track=}"; shift ;;
    --mode)      mode="$2";   shift 2 ;;
    --mode=*)    mode="${1#--mode=}";   shift ;;
    *) break ;;
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
if [ "$mode" = "runtime" ]; then rp=true; else rp=false; fi

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

: "${WATSONX_APIKEY?missing WATSONX_APIKEY}"
: "${WATSONX_URL?missing WATSONX_URL}"
: "${WATSONX_PROJECT_ID?missing WATSONX_PROJECT_ID}"

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

if [ $# -eq 0 ]; then
  benches=(tau2_retail tau2_airline appworld)
else
  benches=("$@")
fi

for bench in "${benches[@]}"; do
  data_dir="scripts/runs/$bench/input"
  results_dir="scripts/runs/$bench/output_${track}_${mode}"
  echo "=== running SPARC ($track, $mode) on $bench ==="
  python -m clear_eval.agentic.pipeline.run_clear_step_analysis \
    --agentic-config-path "$tmp_cfg" \
    --data-dir "$data_dir" \
    --results-dir "$results_dir" \
    --run-name "$bench"
done
