#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python
# Frozen before the 100-seed evaluation: no tuning on seeds 20000–20099.
"$PYTHON" -m robot_world_model.phase3 predict
"$PYTHON" -m robot_world_model.phase3 dev
"$PYTHON" -m robot_world_model.phase3 final
"$PYTHON" -m pytest -q -p no:cacheprovider
