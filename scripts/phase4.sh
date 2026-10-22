#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=.venv/bin/python
# The final 100 seeds (30000–30099) were fixed before running this script.
"$PYTHON" -m robot_world_model.slide_experiment probe
"$PYTHON" -m robot_world_model.slide_experiment collect
"$PYTHON" -m robot_world_model.slide_experiment train
"$PYTHON" -m robot_world_model.slide_experiment predict
"$PYTHON" -m robot_world_model.slide_experiment dev
"$PYTHON" -m robot_world_model.slide_experiment final
"$PYTHON" -m robot_world_model.slide_experiment demo
"$PYTHON" -m pytest -q -p no:cacheprovider
