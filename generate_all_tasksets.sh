#!/usr/bin/env bash
# generate_all_tasksets.sh  (parallel version)
set -euo pipefail

SCRIPT="phase1_generator.py"
BASE_DIR="tasksets"
LAMBDA=0.05
NUM_RUNS=100
MAPPING="wfd"
JOBS="${1:-$(nproc)}"   # pass num jobs as arg, default = all CPU cores

M_LEVELS=(2 4 8)
PM_LEVELS=(2 3 4)
AM_LEVELS=(1 2)
UC_LEVELS=("0.25" "0.5" "0.75")

TOTAL=$(( ${#M_LEVELS[@]} * ${#PM_LEVELS[@]} * ${#AM_LEVELS[@]} * ${#UC_LEVELS[@]} * NUM_RUNS ))
echo "Generating $TOTAL task sets using $JOBS parallel jobs..."

# Pre-create all directories (avoid race conditions)
for m in "${M_LEVELS[@]}"; do
  for pm in "${PM_LEVELS[@]}"; do
    for am in "${AM_LEVELS[@]}"; do
      for uc in "${UC_LEVELS[@]}"; do
        UC_LABEL=$(echo "$uc" | sed 's/\./_/')
        mkdir -p "${BASE_DIR}/M${m}/pm${pm}_am${am}_u${UC_LABEL}"
      done
    done
  done
done

# Emit one command per run into a job list, pipe to parallel/xargs
generate_commands() {
  for m in "${M_LEVELS[@]}"; do
    for pm in "${PM_LEVELS[@]}"; do
      for am in "${AM_LEVELS[@]}"; do
        for uc in "${UC_LEVELS[@]}"; do
          UC_LABEL=$(echo "$uc" | sed 's/\./_/')
          DIR="${BASE_DIR}/M${m}/pm${pm}_am${am}_u${UC_LABEL}"
          for ((r=1; r<=NUM_RUNS; r++)); do
            OUTFILE="${DIR}/run_$(printf '%03d' $r).json"
            echo "python3 $SCRIPT --M $m --periodic_mult $pm --aperiodic_mult $am \
--util_coeff $uc --mapping $MAPPING --lam $LAMBDA --out $OUTFILE"
          done
        done
      done
    done
  done
}

# Use GNU parallel if available, otherwise fall back to xargs
if command -v parallel &>/dev/null; then
  generate_commands | parallel --bar -j "$JOBS"
else
  echo "GNU parallel not found, using xargs -P $JOBS"
  generate_commands | xargs -P "$JOBS" -I{} bash -c '{} > /dev/null 2>&1'
fi

echo "Done. $TOTAL task sets written to $BASE_DIR/"
