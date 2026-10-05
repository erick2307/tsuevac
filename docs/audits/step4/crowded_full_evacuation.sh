#!/bin/sh
# D9 (continued). The same policies over the whole evacuation of kochi_area4: 120 min simulated, 5 runs each (the policies were trained on 30 min).
#
#   PYTHON=python3 sh crowded_full_evacuation.sh        (after crowded_area.py; about 15 min on 4 cores)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$HERE/../../../src"
exp="${PYTHON:-python} -m evacrl.experiment"
OUT="$HERE/runs/area4_120min"
$exp sp kochi_area4 --runs 5 --time 120 --horizon 60 --workers 4 --seed 5 --out "$OUT/sp"
for level in link segment; do
  $exp evaluate kochi_area4 --state "$HERE/state_kochi_area4_$level.csv" --runs 5 --time 120 --workers 4 --seed 5 --set densityLevel=$level --out "$OUT/$level"
done
