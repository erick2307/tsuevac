import json, sys, numpy as np
from scipy import stats
rows=[]
for f in sys.argv[1:]: rows+=json.load(open(f))
V=sorted({r["variant"] for r in rows}); seeds=sorted({r["seed"] for r in rows})
get=lambda v,k: np.array([next(r[k] for r in rows if r["variant"]==v and r["seed"]==s) for s in seeds])
N=622
print(f"{len(seeds)} paired seeds (same seed = same departure times); kochi2, 622 agents, mean departure 5 min, 120 min simulated\n")
print(f"{'variant':16s} {'runs w/ unevacuated':>20s} {'evacuated (mean)':>17s} {'evac time mean±sd (s)':>22s} {'median':>7s} {'p95':>6s} {'at 30 min':>10s}")
for v in V:
    fin=get(v,"flag_final"); et=get(v,"evac_time"); a30=get(v,"at30")
    print(f"{v:16s} {int((fin<N).sum()):>14d}/{len(seeds):<5d} {fin.mean():17.1f} {et.mean():12.0f} ± {et.std(ddof=1):5.0f} {np.median(et):7.0f} {np.percentile(et,95):6.0f} {a30.mean():10.1f}")
print()
def paired(a,b,k,label):
    x=get(a,k)-get(b,k)
    t=stats.ttest_rel(get(a,k),get(b,k)); w=stats.wilcoxon(get(a,k),get(b,k)) if np.any(x!=0) else None
    print(f"  {label:44s} {k:10s} mean diff {x.mean():8.1f}   (identical in {(x==0).sum():2d}/{len(x)} seeds)   paired-t p={t.pvalue:.3g}")
for a,b,l in [("uru","k24","Urushibara vs evacrl(kochi2024)"),("k24","k24_ceil","segment sizing: round vs ceil (2024 options)"),("legacy_round","legacy","segment sizing: round vs ceil (2021 options)"),
              ("k24","legacy","2024 options vs 2021 options (SP)")]:
    if a in V and b in V:
        for k in ("evac_time","at30"): paired(a,b,k,l)
print()
for a,b,l in [("legacy_clamp","legacy","far-end fix alone (2021 options)"),("k24_clamp","k24","far-end fix alone (2024 options)"),("k24_clamp","legacy_clamp","2024 options vs 2021 options, both fixed")]:
    if a in V and b in V:
        for k in ("evac_time","at30"): paired(a,b,k,l)
