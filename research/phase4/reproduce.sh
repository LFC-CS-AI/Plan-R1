#!/usr/bin/env bash
set -euo pipefail
# Run in a NEW checkout to keep these archived outputs intact.
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(git -C "$source_dir" rev-parse --show-toplevel)"
target_dir="${1:?Usage: bash research/phase4/reproduce.sh NEW_WORKTREE_PATH}"
if [[ -e "$target_dir" ]]; then echo 'Target must not exist.' >&2; exit 1; fi
git -C "$repo_dir" worktree add --detach "$target_dir" HEAD
target_root="$(cd -- "$target_dir" && pwd)"
cd -- "$target_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
python research/phase4/source_audit.py > research/phase4/results/source_audit.log 2>&1
python -u research/phase4/build_data.py > research/phase4/results/build_data.log 2>&1
python -u research/phase4/train_heads.py > research/phase4/results/train_heads.log 2>&1
python research/phase4/test_constraints.py > research/phase4/results/constraint_tests.log 2>&1
python research/phase4/audit_labels.py > research/phase4/results/label_audit.log 2>&1
python -u research/phase4/evaluate.py > research/phase4/results/evaluate.log 2>&1
python research/phase4/analyze_results.py > research/phase4/results/analysis.log 2>&1
python research/phase4/verify_saved.py > research/phase4/results/verification.log 2>&1
echo "Completed CPU component reproduction in $target_root/research/phase4; no pretrained/full RL training."
