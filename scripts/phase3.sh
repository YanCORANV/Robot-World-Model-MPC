#!/usr/bin/env bash
set -euo pipefail
# Frozen before the 100-seed evaluation: no tuning on seeds 20000–20099.
python -m robot_world_model.phase3 predict
python -m robot_world_model.phase3 dev
python -m robot_world_model.phase3 final
pytest -q
