#!/usr/bin/env bash
# Small, deterministic end-to-end pipeline. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/rwm probe
.venv/bin/rwm collect --out data/smoke.npz --episodes 12 --seed 7 --steps 40
.venv/bin/rwm train --data data/smoke.npz --out results/smoke.pt --epochs 18 --seed 7 --metrics results/smoke_train.json
.venv/bin/rwm predict --data data/smoke.npz --model results/smoke.pt --plot results/smoke_prediction.png --metrics results/smoke_prediction.json
.venv/bin/rwm evaluate --model results/smoke.pt --seeds 100 101 --steps 30 \
  --candidates 32 --iterations 2 --horizon 6 --metrics results/smoke_evaluation.json
.venv/bin/pytest -q
