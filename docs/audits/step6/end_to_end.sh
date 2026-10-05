#!/bin/sh
# Step 6. End to end on every shipped area (Kochi 0, 1, 2, 4) and the two 2021 cases, as a user would run it:
#   rebuild the case from its raw/ folder and compare with the shipped tables -> validate -> shortest-path runs -> training ->
#   frozen greedy evaluation -> comparison plot -> policy map, then the manifests; and the 2021 entry point on `kochi` and `new_kochi`.
# Small settings: it checks that the whole chain works and agrees with itself, not how good a policy is.
#
#   sh end_to_end.sh [WORK]          about 25 min on 4 cores        (area 3 of the study has no shelter in its box: not a case)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
WORK="${1:-$(mktemp -d)}"
PY="${PYTHON:-python}"
export PYTHONPATH="$REPO/src" MPLBACKEND=Agg
status=0
say() { printf '%s\n' "$*"; }
step() {  # step LABEL COMMAND...   runs the command, records its time and exit code
  label=$1; shift; s=$(date +%s)
  if "$@" > "$WORK/last.log" 2>&1; then say "ok   $(( $(date +%s) - s )) s  $label"; else say "FAIL $label"; tail -5 "$WORK/last.log"; status=1; fi
}
one() {   # one AREA
  a=$1; o="$WORK/$a"; mkdir -p "$o"
  step "$a: rebuild from raw/" $PY -m evacrl.casebuild from-raw "$REPO/cases/$a/raw" "$o/case" --strategy proportional --weights "$REPO/cases/$a/node_population.csv" --merge clusters --threshold 5
  same=yes
  for f in linksdb actionsdb transitionsdb nextnode agentsdb; do cmp -s "$o/case/data/$f.csv" "$REPO/cases/$a/data/$f.csv" || { same=no; say "     differs: $f.csv"; }; done
  # coordinates are written with 6 decimals: a cluster centre that falls on a rounding boundary can differ in the last digit between NumPy versions
  $PY - "$o/case/data/nodesdb.csv" "$REPO/cases/$a/data/nodesdb.csv" <<'PYEOF' || { same=no; say "     differs: nodesdb.csv"; }
import sys
import numpy as np
a, b = (np.loadtxt(f, delimiter=",", comments="#") for f in sys.argv[1:3])
assert a.shape == b.shape and np.array_equal(a[:, [0, 3, 4]], b[:, [0, 3, 4]]) and np.abs(a[:, 1:3] - b[:, 1:3]).max() <= 1.5e-6
PYEOF
  say "     rebuilt tables identical to the shipped ones (coordinates to 1.5e-6 m): $same"; [ $same = yes ] || status=1
  step "$a: validate" $PY -m evacrl.casebuild validate "$REPO/cases/$a"
  step "$a: sp (10 runs, 30 min)" $PY -m evacrl.experiment sp "$a" --runs 10 --time 30 --workers 2 --out "$o/sp"
  step "$a: calibrate (10 simulations)" $PY -m evacrl.experiment calibrate "$a" --method qlearning --sims 10 --eval-every 5 --eval-runs 2 --out "$o/ql" --sp "$o/sp"
  step "$a: evaluate (3 frozen runs)" $PY -m evacrl.experiment evaluate "$a" --state "$o/ql/best_state.csv" --runs 3 --workers 2 --out "$o/ev"
  step "$a: compare" $PY -m evacrl.experiment compare --sp "$o/sp" --rl "$o/ev" --out "$o/compare.png"
  step "$a: policy map" $PY -m evacrl.experiment policy "$a" --state "$o/ql/best_state.csv" --out "$o/policy.png"
  $PY - "$o" "$a" <<'PYEOF' || status=1
import json, os, sys
import numpy as np
o, a = sys.argv[1:3]
for d in ("sp", "ql", "ev"):
    m = json.load(open(os.path.join(o, d, "manifest.json")))          # strict JSON, written by the command
    assert m["case"]["name"], d
for f in ("compare.png", "policy.png", os.path.join("ql", "best_state.csv")):
    assert os.path.getsize(os.path.join(o, f)) > 0, f
sp = np.loadtxt(os.path.join(o, "sp", "runs.csv"), delimiter=",", skiprows=1, ndmin=2)
print(f"     manifests and outputs present; sp runs.csv: {len(sp)} rows")
PYEOF
}
for a in kochi_area0 kochi_area1 kochi_area2 kochi_area4; do one $a & done
wait
say "--- the 2021 entry point on the two older cases (2 simulations of 1 min each)"
cd "$REPO"
for a in kochi new_kochi; do
  step "$a: run_ql_mod" $PY -c "
import sys; sys.path.insert(0, 'scripts')
from main_ql_mod import run_ql_mod
run_ql_mod(area='$a', simtime=1, meandeparture=0.5, numBlocks=1, simPerBlock=1, name='e2e')"
  rm -rf "cases/$a/state_e2e"
done
say "work folder: $WORK"
exit $status
