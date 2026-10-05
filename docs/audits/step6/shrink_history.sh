#!/bin/sh
# D15. Shrinks the history of the repository: removes from every branch the files that are no longer in the tip but weigh most in the
# history (the Arahama and Kochi state dumps of 2021, two AVI videos, the GIS layers of datasets/gis/data), and checks the result.
# Nothing is pushed unless you run it with --push. Read "What a push does" in docs/audits/step6/README.md first.
#
#   sh shrink_history.sh [--push]        (needs: pip install git-filter-repo)
set -eu
REPO="${REPO:-https://github.com/erick2307/tsuevac}"
WORK="${WORK:-$(mktemp -d)}"
mkdir -p "$WORK"
cd "$WORK"
cat > drop.txt <<'LIST'
arahama/
kochi/state_eraseme_mc/
kochi/state_eraseme/
new_kochi/state_sarsa_30_15/
app/ql_arahama_sim_000000010.avi
app/ql_arahama_sim_000000010_wring.avi
datasets/gis/data/qgis/
datasets/gis/data/qgis_1/
system/data/qgis/
system/data/qgis_1/
LIST
git clone -q --mirror "$REPO" before.git
cp -r before.git after.git
(cd after.git && git filter-repo --force --invert-paths --paths-from-file ../drop.txt > ../filter.log 2>&1 && git reflog expire --expire=now --all && git gc -q --prune=now --aggressive)
echo "pack before: $(git -C before.git count-objects -vH | grep size-pack)"
echo "pack after:  $(git -C after.git count-objects -vH | grep size-pack)"
# Check: at the tip of every branch the files are the same as before, except the removed paths.
status=0
for ref in $(git -C before.git for-each-ref --format='%(refname)' refs/heads); do
  git -C before.git ls-tree -r "$ref" | awk '{print $3, $4}' | grep -v -F -f drop.txt | sort > before.tree
  git -C after.git ls-tree -r "$ref" | awk '{print $3, $4}' | sort > after.tree
  if cmp -s before.tree after.tree; then echo "ok   $ref: the tip is the same without the removed paths ($(wc -l < after.tree) files)"
  else echo "FAIL $ref: the tips differ"; diff before.tree after.tree | head; status=1; fi
  echo "     commits: $(git -C before.git rev-list --count "$ref") before, $(git -C after.git rev-list --count "$ref") after"
done
[ "$status" = 0 ] || exit 1
if [ "${1:-}" = "--push" ]; then
  git -C after.git push --force "$REPO" 'refs/heads/*:refs/heads/*'
else
  echo "not pushed. To push: sh shrink_history.sh --push   (or: cd $WORK/after.git && git push --force $REPO 'refs/heads/*:refs/heads/*')"
fi
