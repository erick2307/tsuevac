# The 2024 study's shortest-path results, regenerated

The Kochi study (`erick2307/2024_urushibara`, `EVACMODEL3_FocalPoints/results_for Usama/kochi*/results_time.csv` and `results_df.csv`) ran 1,000
shortest-path simulations per area with an engine in which agents freeze at the far end of a link (Defect A of
[docs/engine-reconciliation.md](../../docs/engine-reconciliation.md)): in 43-63 % of the runs agents were still on their way after 2 h, and the
evacuation time has a tail up to the end of the simulation. These are the same simulations on the same tables (the study's `population_1.csv`,
`nodes.csv`, `edges.csv`, `actionsdb.csv`, `transitionsdb.csv`, `nextnode.csv`), 120 min, mean departure 5 min, with the default `ModelOptions` (that
defect fixed), made with `python -m evacrl.experiment sp AREA --until-converged` (`docs/audits/step4/regenerate_2024.sh`).

One folder per area: `runs.csv` (a row per run: seed, agents, safe at 30 min, safe at the end, evacuation time), `convergence.csv` (the running
mean, sd, CV and standard error after every batch of 10 runs), `curves_summary.csv` (mean and 5th / 95th percentile of the safe agents every 10 s)
and `manifest.json` (the checksums of the input files, the options, every seed, the code version). The curves of every run (`curves.npz`) are not
kept in the repository. The comparison with the committed results is in [docs/audits/step4](../../docs/audits/step4/README.md).

`kochi4_180min` is `kochi4` simulated for 180 min (20 runs, no convergence rule): in the 120-min runs 5 of 30 still had agents walking at the
end, so the evacuation time of that area is censored there; in 180 min every run ends (6,966 ± 234 s).
