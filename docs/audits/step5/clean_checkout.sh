#!/bin/sh
# Step 5: the repository as a stranger gets it. Clones REPO at REF into a scratch folder, and for each Python version makes a new
# virtual environment, installs the package, runs lint and the whole test suite (once with the core install, once with the casebuild
# extra, where the geospatial tests run too); with the first version it also builds the wheel and the sdist, installs the wheel
# somewhere else and runs the commands of the README quick start.
#
#   sh clean_checkout.sh [REPO] [REF]       REPO defaults to the origin of this checkout, REF to the current branch
#   PYVERS="3.10 3.11 3.12 3.13" (default), WORK=/some/folder (default: a new temporary folder), JOBS=4
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TOP=$(cd "$HERE/../../.." && pwd)
REPO="${1:-$(git -C "$TOP" remote get-url origin)}"
REF="${2:-$(git -C "$TOP" rev-parse --abbrev-ref HEAD)}"
PYVERS="${PYVERS:-3.10 3.11 3.12 3.13}"
WORK="${WORK:-$(mktemp -d)}"
echo "clean checkout of $REPO ($REF) in $WORK"
git clone -q --branch "$REF" "$REPO" "$WORK/src"
git -C "$WORK/src" log --oneline -1
if [ "$REPO" = "$(git -C "$TOP" remote get-url origin)" ] && [ "$(git -C "$WORK/src" rev-parse HEAD)" != "$(git -C "$TOP" rev-parse HEAD)" ]; then
  echo "the clone ($(git -C "$WORK/src" rev-parse --short HEAD)) is not the HEAD of $TOP ($(git -C "$TOP" rev-parse --short HEAD)): push first" >&2
  exit 2
fi

one() {   # one VERSION KIND
  v=$1; kind=$2; env="$WORK/venv-$v-$kind"
  uv venv -q --python "$v" "$env"
  case $kind in core) extra="." ;; geo) extra=".[casebuild]" ;; esac   # core: the install of the README, nothing else
  (cd "$WORK/src" && uv pip install -q --python "$env/bin/python" -e "$extra") > "$WORK/install-$v-$kind.log" 2>&1 || { echo "FAIL install $v $kind"; return 1; }
  (cd "$WORK/src" && MPLBACKEND=Agg "$env/bin/python" -W error -m unittest discover tests) > "$WORK/tests-$v-$kind.log" 2>&1 \
     && echo "ok   tests $v $kind: $(tail -3 "$WORK/tests-$v-$kind.log" | tr '\n' ' ')" \
     || { echo "FAIL tests $v $kind (see $WORK/tests-$v-$kind.log)"; tail -15 "$WORK/tests-$v-$kind.log"; return 1; }
}

status=0
pids=""
for v in $PYVERS; do for kind in core geo; do one "$v" "$kind" & pids="$pids $!"; done; done
for p in $pids; do wait "$p" || status=1; done

first=${PYVERS%% *}
echo "--- lint and build ($first)"
uv venv -q --python "$first" "$WORK/venv-tools"
uv pip install -q --python "$WORK/venv-tools/bin/python" ruff build twine cffconvert
(cd "$WORK/src" && "$WORK/venv-tools/bin/ruff" check src tests scripts && echo "ok   ruff") || status=1
(cd "$WORK/src" && "$WORK/venv-tools/bin/cffconvert" --validate -i CITATION.cff) || status=1
(cd "$WORK/src" && "$WORK/venv-tools/bin/python" -m build --outdir "$WORK/dist" > "$WORK/build.log" 2>&1 && "$WORK/venv-tools/bin/twine" check "$WORK"/dist/*) || { echo "FAIL build"; status=1; }
ls -l "$WORK"/dist

echo "--- wheel installed away from the checkout, quick start run from the checkout"
uv venv -q --python "$first" "$WORK/venv-wheel"
uv pip install -q --python "$WORK/venv-wheel/bin/python" "$WORK"/dist/*.whl
(cd /tmp && "$WORK/venv-wheel/bin/python" -c "import evacrl; print('wheel imports evacrl', evacrl.__version__, evacrl.__file__)")
(cd "$WORK/src" && "$WORK/venv-wheel/bin/python" -m evacrl.casebuild validate cases/kochi_area2 > "$WORK/validate.log" 2>&1 && tail -2 "$WORK/validate.log") || { echo "FAIL casebuild validate"; cat "$WORK/validate.log"; status=1; }
(cd /tmp && "$WORK/venv-wheel/bin/evacrl-casebuild" --help > /dev/null && "$WORK/venv-wheel/bin/evacrl-experiment" --help > /dev/null && echo "ok   both console scripts answer --help") || { echo "FAIL console scripts"; status=1; }
mkdir -p "$WORK/qs" && cd "$WORK/src"
export MPLBACKEND=Agg
for step in \
  "sp kochi_area2 --runs 10 --time 30 --workers 2 --out $WORK/qs/sp" \
  "calibrate kochi_area2 --method qlearning --sims 30 --eval-every 10 --eval-runs 3 --out $WORK/qs/ql --sp $WORK/qs/sp" \
  "evaluate kochi_area2 --state $WORK/qs/ql/best_state.csv --runs 5 --workers 2 --out $WORK/qs/ql_eval" \
  "compare --sp $WORK/qs/sp --rl $WORK/qs/ql_eval --out $WORK/qs/compare.png" \
  "policy kochi_area2 --state $WORK/qs/ql/best_state.csv --out $WORK/qs/policy.png"; do
  start=$(date +%s)
  # shellcheck disable=SC2086
  "$WORK/venv-wheel/bin/python" -m evacrl.experiment $step > "$WORK/qs/last.log" 2>&1 && echo "ok   $(( $(date +%s) - start )) s  evacrl.experiment ${step%% *}" || { echo "FAIL evacrl.experiment $step"; tail -8 "$WORK/qs/last.log"; status=1; }
done
ls "$WORK/qs"
echo "work folder: $WORK"
exit $status
