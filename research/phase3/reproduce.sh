#!/usr/bin/env bash
set -euo pipefail
# Requires the CPU packages listed in results/environment.json already installed.
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "$source_dir" rev-parse --show-toplevel)"
target_dir="${1:?Usage: bash reproduce.sh NEW_WORKTREE_PATH}"
if [[ -e "$target_dir" ]]; then echo 'Target must not exist.' >&2; exit 1; fi
git -C "$repo_dir" worktree add --detach "$target_dir" HEAD
target_root="$(cd -- "$target_dir" && pwd)"
cd -- "$target_root/research/phase3"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python source_audit.py > results/source_audit.log 2>&1
python test_geometry.py > results/tests.log 2>&1
python run_diagnostics.py > results/run.log 2>&1
python validate_results.py > results/validation.log 2>&1
python interval_learning_prototype.py > results/learning_probe.log 2>&1
python analyze_results.py > results/analysis.log 2>&1
python build_report.py > results/report.log 2>&1
echo "Completed. Inspect $PWD/results and experiment_results.md. No originals overwritten."
