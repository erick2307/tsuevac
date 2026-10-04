#!/bin/sh
# D8. The shortest-path results of the 2024 Kochi study (1,000 runs per area in `results_for Usama/kochi*/results_time.csv` and `results_df.csv`),
# regenerated with the default `ModelOptions` (the far-end freeze fixed), 120 min simulated, mean departure 5 min, on the study's own tables
# (population_1.csv). Runs are added in batches of 10 until the evacuation time and the number safe at 30 min have a standard error of the
# mean under 1 % (`--until-converged`), at most MAX runs: 300 for the three small areas, 120 for the two large ones (about 90 s per run).
#
#   URUSHIBARA_DIR=~/2024_Urushibara/EVACMODEL3_FocalPoints PYTHON=python3 WORKERS=2 sh regenerate_2024.sh       (about 3 h on 2 cores)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
STUDY="${URUSHIBARA_DIR:-$HOME/2024_Urushibara/EVACMODEL3_FocalPoints}/results_for Usama"
OUT="$HERE/../../../results/kochi2024_regenerated"
export PYTHONPATH="$HERE/../../../src"
exp="${PYTHON:-python} -m evacrl.experiment"
for entry in kochi2:300 kochi1:300 kochi0:300 kochi4:120 kochi42:120; do
  area=${entry%%:*}; max=${entry##*:}
  $exp sp "$STUDY/$area" --until-converged --max-runs "$max" --time 120 --workers "${WORKERS:-4}" --seed 1 --out "$OUT/$area"
done
