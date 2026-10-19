# Phase 1 research delivery

This handoff preserves the 24 existing research files byte-for-byte. No research experiments were rerun, and no original paper implementation files were modified.

The original `execution_report.md` is a historical report from before GitHub delivery; its statement that no PR was created describes that earlier phase. This file records the subsequent packaging and delivery.

## Download the existing ZIP

[Download the original phase-1 ZIP](https://github.com/LFC-CS-AI/Plan-R1/raw/refs/heads/research/phase1-audit/research/archives/plan_r1_phase1_research.zip)

ZIP size: 440,846 bytes. CRC integrity: passed. All 24 archived files match the original directory by SHA-256.

`SHA256 fd7a2d7e197729dc1b24940aea924cb2339ace8080acaf6b6ca56d2bc205f269`

## Original files copied to this directory

| Relative path | Bytes |
|---|---:|
| [README.md](README.md) | 1,424 |
| [execution_report.md](execution_report.md) | 5,204 |
| [experiment_results.md](experiment_results.md) | 10,726 |
| [literature_review.md](literature_review.md) | 14,263 |
| [plan_r1_analysis.md](plan_r1_analysis.md) | 15,359 |
| [requirements-toy.txt](requirements-toy.txt) | 137 |
| [research_proposal.md](research_proposal.md) | 11,031 |
| [results/broadcast_weights.csv](results/broadcast_weights.csv) | 166 |
| [results/credit_mechanisms.png](results/credit_mechanisms.png) | 159,867 |
| [results/credit_mechanisms.svg](results/credit_mechanisms.svg) | 67,069 |
| [results/gradient_metrics.csv](results/gradient_metrics.csv) | 5,023 |
| [results/gradient_profiles.csv](results/gradient_profiles.csv) | 45,055 |
| [results/learning.csv](results/learning.csv) | 39,254 |
| [results/learning.png](results/learning.png) | 111,050 |
| [results/learning.svg](results/learning.svg) | 57,796 |
| [results/model_error.csv](results/model_error.csv) | 2,142 |
| [results/reproducibility.json](results/reproducibility.json) | 412 |
| [results/run.log](results/run.log) | 3,217 |
| [results/source_audit.json](results/source_audit.json) | 2,283 |
| [results/source_mechanisms.png](results/source_mechanisms.png) | 98,753 |
| [results/summary.json](results/summary.json) | 3,216 |
| [results/termination.csv](results/termination.csv) | 355 |
| [source_manifest.json](source_manifest.json) | 1,602 |
| [toy_experiment.py](toy_experiment.py) | 21,054 |

Original files: 24; total uncompressed bytes: 676,458. The delivery adds this manifest and the unchanged original ZIP under `archives/`.

Included: reports, experiment source, six CSV tables, four JSON evidence/metadata files, three PNG figures, two SVG figures, one execution log and dependency pins. No environments, caches, checkpoints or datasets are included.

The original files remain at `/workspace/research/` (backed by `/workspace/output/research/`), and the original ZIP remains at `/workspace/output/plan_r1_phase1_research.zip`.

Target repository: `LFC-CS-AI/Plan-R1`; delivery branch: `research/phase1-audit`; PR base: `main`. The main branch is not modified or merged by this delivery.
