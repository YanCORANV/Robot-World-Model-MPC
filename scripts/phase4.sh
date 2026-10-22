#!/usr/bin/env bash
set -euo pipefail
# The final 100 seeds (30000–30099) were fixed before running this script.
python -m robot_world_model.slide_experiment probe
python -m robot_world_model.slide_experiment collect
python -m robot_world_model.slide_experiment train
python -m robot_world_model.slide_experiment predict
python -m robot_world_model.slide_experiment dev
python -m robot_world_model.slide_experiment final
python -m robot_world_model.slide_experiment demo
python -m pytest -q -p no:cacheprovider
