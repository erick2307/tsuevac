#!/bin/sh
# 4b. The same stored policies evaluated frozen (nothing learned during the run, the default of `evaluate`) and with the agents learning on a copy
# (`--keep-learning`, what calibration.py of the study and the audits gamma_learning.py / demo_cli.sh did): 50 runs each, the same seeds.
#
#   URUSHIBARA_DIR=~/2024_Urushibara/EVACMODEL3_FocalPoints PYTHON=python3 sh frozen_evaluation.sh        (after gamma_learning.py; about 15 min on 4 cores)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CASE="${URUSHIBARA_DIR:-$HOME/2024_Urushibara/EVACMODEL3_FocalPoints}/results_for Usama/kochi2"
export PYTHONPATH="$HERE/../../../src"
exp="${PYTHON:-python} -m evacrl.experiment"
mkdir -p "$HERE/runs"
for p in decision_0.9_100_0 decision_0.9_300_0 second_0.999_100_0 second_0.999_300_0; do
  case $p in
    decision*) set_opts="--discount 0.9" ;;
    *) set_opts="--discount 0.999 --set discounting=second" ;;
  esac
  $exp evaluate "$CASE" --state "$HERE/state_kochi2_$p.csv" --runs 50 --workers 4 --seed 1 $set_opts --out "$HERE/runs/frozen_$p"
  $exp evaluate "$CASE" --state "$HERE/state_kochi2_$p.csv" --runs 50 --workers 4 --seed 1 $set_opts --keep-learning --out "$HERE/runs/adapting_$p"
done
