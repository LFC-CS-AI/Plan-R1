#!/usr/bin/env bash
set -euo pipefail
# Use an already installed Python 3.9+ environment. No downloads or GPU use.
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
target_dir="${1:?Usage: bash reproduce.sh NEW_EMPTY_OUTPUT_DIRECTORY}"
if [[ -e "$target_dir" ]]; then
  echo 'Refusing to overwrite an existing directory.' >&2
  exit 1
fi
mkdir -p -- "$target_dir"
cp -- "$source_dir"/*.py "$source_dir/protocol.json" "$source_dir/baseline_extension.json" "$target_dir/"
cd -- "$target_dir"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
mkdir results
python test_mechanisms.py > results/tests.log 2>&1
python run_experiment.py --out results > results/run.log 2>&1
python extend_baselines.py > results/extension.log 2>&1
python validate_numerics.py > results/numerical_validation.log 2>&1
python verify_results.py > results/verification.log 2>&1
python analyze_results.py > results/analysis.log 2>&1
python build_report.py
echo "Finished; inspect results/verification.json and experiment_results.md in $PWD"
