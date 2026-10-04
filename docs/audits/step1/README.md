# Step 1 audit: reproducing the engine comparison

These scripts produced the numbers in [docs/engine-reconciliation.md](../../engine-reconciliation.md). They compare
`evacrl` with `EVACMODEL3_FocalPoints/SARSA2024.py` of [erick2307/2024_urushibara](https://github.com/erick2307/2024_urushibara)
on its committed case `kochi2` (309 nodes, 622 agents, 4 shelters). Nothing is written into that repository.

```
git clone https://github.com/erick2307/2024_urushibara ~/2024_Urushibara     # or set URUSHIBARA_DIR
pip install -e ".[plots]" scipy
cd docs/audits/step1
python prepare.py                                  # the engine before Step 1 + integer-formatted kochi2 (for head_sp.py)

python battery.py exact 8                          # Urushibara engine == evacrl(2024 options), shortest path
python battery.py learn_exact 4 2                  # ... and for SARSA training (state matrices equal)
python battery.py sp 30 sp.json  uru k24 k24_ceil legacy legacy_round legacy_clamp k24_clamp k24_ceil_clamp
python head_sp.py 30                               # the engine before Step 1: what main_ShortPath.py reports
python analyze_sp.py sp.json                           # the table of the document
python battery.py learn 50 3                       # SARSA training ablation (about 25 min on 4 cores)
python analyze_learn.py learn_50.json              # its table (tests on per-seed means)
```

The variant names: `uru` is the Urushibara engine; `k24` is `ModelOptions.kochi2024()` (it sizes the link segments with
`round`, as `SARSA2024.py` does; `k24_ceil` uses `ceil` instead); `clamp` adds `segmentIndex="clamped"`; `legacy` is
`ModelOptions.legacy()` (the defaults of `ModelOptions()` changed after Step 1).
A seed fixes the departure times, so the runs of different variants with the same seed are paired.
