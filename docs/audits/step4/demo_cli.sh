#!/bin/sh
# 4b. The command line on the real case: the shortest-path baseline, the two learned policies of gamma_learning.py (discount 0.9 per
# decision, which is what the code always did, and 0.999 per second), their evaluation and the figures of this audit.
#
#   URUSHIBARA_DIR=~/2024_Urushibara/EVACMODEL3_FocalPoints PYTHON=python3 sh demo_cli.sh        (run gamma_learning.py first; about 8 min on 4 cores)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CASE="${URUSHIBARA_DIR:-$HOME/2024_Urushibara/EVACMODEL3_FocalPoints}/results_for Usama/kochi2"
OUT="$HERE/runs"
export PYTHONPATH="$HERE/../../../src"
exp="${PYTHON:-python} -m evacrl.experiment"
mkdir -p "$OUT" "$HERE/figures"

$exp sp "$CASE" --runs 50 --time 120 --workers 4 --seed 1 --out "$OUT/sp"
for p in decision_0.9_100_0 second_0.999_100_0; do
  case $p in
    decision*) set_opts="--discount 0.9"; label="discount 0.9 per decision" ;;
    *) set_opts="--discount 0.999 --set discounting=second"; label="discount 0.999 per second" ;;
  esac
  $exp evaluate "$CASE" --state "$HERE/state_kochi2_$p.csv" --runs 50 --workers 4 --seed 1 $set_opts --out "$OUT/eval_$p"
  $exp policy "$CASE" --state "$HERE/state_kochi2_$p.csv" --out "$HERE/figures/policy_$p.png"
  $exp compare --sp "$OUT/sp" --rl "$OUT/eval_$p" --out "$HERE/figures/compare_$p.png" --horizon 30 \
       --title "kochi2: shortest path and Q-learning, $label"
done
