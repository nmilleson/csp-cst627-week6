#!/usr/bin/env bash
# Reproduce the full Week 6 audit end to end (Git Bash on Windows, or Linux/macOS).
# Expects the virtualenv at .venv with requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/Scripts/python"; [ -x "$PY" ] || PY=".venv/bin/python"
export PYTHONIOENCODING=utf-8

# 1. Synthetic member population + appeal datasets (with planted canary)
$PY data/generate_population.py --seed 627 --output-dir data

# 2. Fine-tune the LoRA adapter (micro-batch 2 x grad-accum 8 fits an 8 GB GPU)
$PY training/sft_train.py --output-dir checkpoints/sft-adapter --batch-size 2 --grad-accum 8

# 3. Task evaluation (in-distribution + out-of-template), base vs adapted
$PY audit/evaluate.py --label base
$PY audit/evaluate.py --label sft --adapter checkpoints/sft-adapter

# 4. Safety probe suite
$PY audit/safety_probes.py --label base
$PY audit/safety_probes.py --label sft --adapter checkpoints/sft-adapter

# 5. Fairness across member slices
$PY audit/fairness.py --labels base sft

# 6. Contamination: n-gram overlap + canary exposure
$PY audit/contamination.py

# 7. Risk register workbook
$PY governance/build_risk_register.py
