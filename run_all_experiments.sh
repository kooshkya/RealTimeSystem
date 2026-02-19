#!/usr/bin/env bash
# run_all_experiments.sh
# Runs phase2_simulator.py over every leaf taskset directory in parallel,
# then aggregates all results.json files into a single summary.csv.
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh [tasksets_root] [output_dir] [jobs]
#
# Defaults:
#   tasksets_root = ./tasksets
#   output_dir    = ./results
#   jobs          = number of CPU cores

set -euo pipefail

TASKSETS_ROOT="${1:-./tasksets}"
OUTPUT_DIR="${2:-./results}"
JOBS="${3:-$(nproc)}"
SIMULATOR="$(pwd)/phase2_simulator.py"
UPDATE_INTERVAL=50.0

mkdir -p "$OUTPUT_DIR"

# Find all leaf directories containing run_*.json files
mapfile -t DIRS < <(
    find "$TASKSETS_ROOT" -type f -name 'run_*.json' \
        -exec dirname {} \; | sort -u
)

TOTAL=${#DIRS[@]}
if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No run_*.json files found under $TASKSETS_ROOT"
    exit 1
fi

echo "========================================"
echo "  Phase 2 Experiment Runner"
echo "========================================"
echo "  Taskset root : $TASKSETS_ROOT"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Experiments  : $TOTAL"
echo "  Parallel jobs: $JOBS"
echo "========================================"

# ---------------------------------------------------------------------------
# Worker function: run all 3 algorithms on one directory
# ---------------------------------------------------------------------------
run_one() {
    local dir="$1"
    local output_dir="$2"
    local tasksets_root="$3"
    local simulator="$4"
    local update_interval="$5"

    # Build output path mirroring the taskset structure
    local rel_path
    rel_path=$(echo "$dir" | sed "s|${tasksets_root}/||")
    local outdir="${output_dir}/${rel_path}"
    mkdir -p "$outdir"

    local outfile="${outdir}/results.json"

    # Skip if already done (comment out to always re-run)
    # [[ -f "$outfile" ]] && { echo "SKIP: $dir"; return 0; }

    python3 "$simulator" \
        --dir "$dir" \
        --algo all \
        --update_interval "$update_interval" \
        --out "$outfile" \
        2>"${outdir}/stderr.log"

    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo "FAIL [$exit_code]: $dir  (see ${outdir}/stderr.log)"
    else
        # Clean up empty stderr logs
        [[ ! -s "${outdir}/stderr.log" ]] && rm -f "${outdir}/stderr.log"
        echo "OK: $rel_path"
    fi
}
export -f run_one

# ---------------------------------------------------------------------------
# Run in parallel
# ---------------------------------------------------------------------------
echo ""
echo "Starting simulations..."
echo ""

if command -v parallel &>/dev/null; then
    printf '%s\n' "${DIRS[@]}" | \
        parallel --bar -j "$JOBS" \
            run_one {} "$OUTPUT_DIR" "$TASKSETS_ROOT" "$SIMULATOR" "$UPDATE_INTERVAL"
else
    echo "GNU parallel not found — using xargs -P $JOBS"
    printf '%s\n' "${DIRS[@]}" | \
        xargs -P "$JOBS" -I{} bash -c \
            'run_one "$@"' _ {} "$OUTPUT_DIR" "$TASKSETS_ROOT" "$SIMULATOR" "$UPDATE_INTERVAL"
fi

echo ""
echo "All simulations complete."

# ---------------------------------------------------------------------------
# Aggregate into summary.csv
# ---------------------------------------------------------------------------
echo "Aggregating results into ${OUTPUT_DIR}/summary.csv ..."

AGGR_SCRIPT=$(mktemp /tmp/aggregate_XXXXXX.py)
cat > "$AGGR_SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
import json, os, csv, sys, re

results_root = sys.argv[1]
out_csv = os.path.join(results_root, "summary.csv")

FIELDNAMES = [
    'M', 'n_periodic', 'n_aperiodic', 'U_total', 'lambda_true',
    'periodic_mult', 'aperiodic_mult', 'config_dir',
    'algorithm', 'num_runs',
    # Energy
    'energy_total_mean',
    # Aperiodic
    'ap_total_mean', 'ap_arrived_mean', 'ap_completed_mean',
    'ap_missed_mean', 'ap_miss_rate_mean',
    # Periodic
    'periodic_deadlines_total_mean', 'periodic_deadlines_missed_mean',
    'periodic_miss_rate_mean',
    # System
    'alpha_mean', 'util_mean',
    # Bayesian
    'lambda_est_final_mean', 'lambda_est_error_mean',
]

rows = []
failed = []

for root, dirs, files in os.walk(results_root):
    if 'results.json' not in files:
        continue
    fpath = os.path.join(root, 'results.json')
    try:
        with open(fpath) as f:
            data = json.load(f)
    except Exception as e:
        failed.append((fpath, str(e)))
        continue

    # Parse config from directory name
    name = os.path.basename(root)
    parent = os.path.basename(os.path.dirname(root))

    # Extract M from parent dirs (e.g. M2/pm2_am1_u0_25)
    m_match  = re.search(r'M(\d+)', root)
    pm_match = re.search(r'pm(\d+)', name)
    am_match = re.search(r'am(\d+)', name)
    u_match  = re.search(r'u(\d+)_(\d+)', name) or re.search(r'u(\d+)', name)

    M  = m_match.group(1)  if m_match  else '?'
    pm = pm_match.group(1) if pm_match else '?'
    am = am_match.group(1) if am_match else '?'
    if u_match:
        groups = u_match.groups()
        u_str = f"{groups[0]}.{groups[1]}" if len(groups) == 2 else groups[0]
    else:
        u_str = '?'

    for algo, res in data.items():
        row = {k: '' for k in FIELDNAMES}
        row['config_dir']     = os.path.relpath(root, results_root)
        row['M']              = res.get('M', M)
        row['n_periodic']     = res.get('n_periodic', '?')
        row['n_aperiodic']    = res.get('n_aperiodic', '?')
        row['U_total']        = res.get('U_total', u_str)
        row['lambda_true']    = res.get('lambda_true', '?')
        row['periodic_mult']  = pm
        row['aperiodic_mult'] = am
        row['algorithm']      = algo
        row['num_runs']       = res.get('num_runs', '')

        for key in FIELDNAMES:
            if key in res and row[key] == '':
                row[key] = res[key]

        rows.append(row)

if not rows:
    print("ERROR: No results.json files found or all failed to parse.")
    if failed:
        print("Failed files:")
        for fp, err in failed:
            print(f"  {fp}: {err}")
    sys.exit(1)

with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to {out_csv}")
if failed:
    print(f"WARNING: {len(failed)} files failed to parse:")
    for fp, err in failed:
        print(f"  {fp}: {err}")
PYEOF

python3 "$AGGR_SCRIPT" "$OUTPUT_DIR"
rm -f "$AGGR_SCRIPT"

echo ""
echo "========================================"
echo "  DONE"
echo "  Summary CSV : ${OUTPUT_DIR}/summary.csv"
echo "  Run plots   : python3 analyze_results.py --csv ${OUTPUT_DIR}/summary.csv"
echo "========================================"
