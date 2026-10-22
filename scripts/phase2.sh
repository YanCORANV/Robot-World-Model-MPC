#!/usr/bin/env bash
# Reproduce the phase-2 dataset, model, diagnostics, and 30-seed comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m robot_world_model.phase2 collect
.venv/bin/python -m robot_world_model.phase2 train
.venv/bin/python -m robot_world_model.phase2 predict
.venv/bin/python -m robot_world_model.phase2 ablate
.venv/bin/python -m robot_world_model.phase2 evaluate
.venv/bin/pytest -q
