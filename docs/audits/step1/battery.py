# -*- coding: utf-8 -*-
"""Battery of audits (see README.md in this folder).

    python battery.py exact N                       Urushibara engine vs evacrl(2024 options) on N seeds, shortest path
    python battery.py sp N out.json variant ...     paired shortest-path runs; variants: see VARIANTS below
    python battery.py learn NSIMS NSEEDS            SARSA training ablation of the options
    python battery.py learn_exact NSIMS NSEEDS      SARSA training, Urushibara engine vs evacrl(2024 options)
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

import harness as H
from evacrl.options import ModelOptions

T_SP = 120 * 60
_, NEXT = H.files()


VARIANTS = {
    "uru": None,                                                            # the Urushibara engine itself
    "k24": ModelOptions.kochi2024(),                                        # its behaviour, in evacrl
    "k24_ceil": ModelOptions.kochi2024().replace(segmentSizing="ceil"),
    "legacy": ModelOptions(),                                               # evacrl as of 2021
    "legacy_round": ModelOptions(segmentSizing="round"),
    "legacy_clamp": ModelOptions(segmentIndex="clamped"),                   # + the far-end fix
    "k24_clamp": ModelOptions.kochi2024().replace(segmentIndex="clamped"),
    "k24_ceil_clamp": ModelOptions.kochi2024().replace(segmentSizing="ceil", segmentIndex="clamped"),
}


def sp_task(args):
    seed, variant = args
    np.random.seed(seed)
    m = H.make("uru") if variant == "uru" else H.make("tsu", options=VARIANTS[variant])
    t0 = int(min(m.pedDB[:, 9]))
    flag, node = H.run_sp(m, NEXT, T_SP)
    return dict(seed=seed, variant=variant, flag=flag.tolist()[::10], flag_final=int(flag[-1]), node_final=int(node[-1]),
                evac_time=H.evac_time(flag, t0), at30=int(flag[30 * 60 - t0 - 1]), t0=t0,
                flag_equals_node=bool(np.array_equal(flag, node)))


def exact(nseeds):
    jobs = [(s, v) for s in range(nseeds) for v in ("uru", "k24")]
    with Pool(4) as p:
        out = p.map(sp_task, jobs)
    by = {(r["seed"], r["variant"]): r for r in out}
    ok = [by[(s, "uru")]["flag"] == by[(s, "k24")]["flag"] for s in range(nseeds)]
    print(f"SP: Urushibara engine vs evacrl(kochi2024) identical curves: {sum(ok)}/{nseeds} seeds")
    print("flag == arrived-by-node for every run:", all(r["flag_equals_node"] for r in out))
    json.dump(out, open("exact_sp.json", "w"))


def sp(nseeds, variants, out_json):
    jobs = [(s, v) for s in range(nseeds) for v in variants]
    t0 = time.time()
    with Pool(4) as p:
        out = p.map(sp_task, jobs, chunksize=1)
    print("elapsed", round(time.time() - t0), "s")
    json.dump(out, open(out_json, "w"))


# ---------- learning ----------
CONFIGS = {
    "legacy_raw": ModelOptions(),                                                    # the 2021 code as it is today
    "base": ModelOptions(segmentIndex="clamped"),                                    # 2021 + the far-end fix
    "reward1e7": ModelOptions(surviveReward=10000000, segmentIndex="clamped"),
    "density_segment": ModelOptions(densityLevel="segment", segmentIndex="clamped"),
    "entry_position": ModelOptions(entrySpeed="position", segmentIndex="clamped"),
    "all2024": ModelOptions.kochi2024().replace(segmentIndex="clamped"),             # the 2024 choices + the fix
}


def learn_task(args):
    cfg, seed, nsims, T, neval = args
    np.random.seed(seed)
    glee = 1.0 / nsims
    state, curve = None, []
    t0 = time.time()
    for s in range(nsims):
        eps = 1.0 / (glee * s + 1.0)
        m = H.make("uru") if cfg == "uru" else H.make("tsu", options=CONFIGS[cfg])
        if state is not None:
            m.stateMat = state
        curve.append(H.run_rl(m, T, eps))
        state = m.stateMat
    greedy = []
    for k in range(neval):
        m = H.make("uru") if cfg == "uru" else H.make("tsu", options=CONFIGS[cfg])
        m.stateMat = state.copy()
        greedy.append(H.run_rl(m, T, 0.0))
    return dict(cfg=cfg, seed=seed, curve=curve, greedy=greedy, nstates=int(state.shape[0]),
                total=int(m.numPedestrian), secs=round(time.time() - t0))


def learn(nsims, nseeds):
    T = 30 * 60
    jobs = [(c, s, nsims, T, 5) for s in range(nseeds) for c in CONFIGS]
    with Pool(4) as p:
        out = p.map(learn_task, jobs, chunksize=1)
    json.dump(out, open(f"learn_{nsims}.json", "w"))
    for c in CONFIGS:
        g = [np.mean(r["greedy"]) for r in out if r["cfg"] == c]
        last = [np.mean(r["curve"][-10:]) for r in out if r["cfg"] == c]
        print(f"{c:16s} greedy survivors@30min {np.mean(g):6.1f}  train(last10) {np.mean(last):6.1f}  states {np.mean([r['nstates'] for r in out if r['cfg']==c]):.0f}")


def learn_exact(nsims, nseeds):
    """Urushibara SARSA vs evacrl(kochi2024 + round): same seed -> same training?"""
    res = []
    for seed in range(nseeds):
        mats = {}
        for kind in ("uru", "tsu"):
            np.random.seed(seed)
            state, surv = None, []
            for s in range(nsims):
                m = H.make("uru") if kind == "uru" else H.make("tsu", options=ModelOptions.kochi2024())
                if state is not None:
                    m.stateMat = state
                surv.append(H.run_rl(m, 30 * 60, 1.0 / ((1.0 / nsims) * s + 1.0)))
                state = m.stateMat
            mats[kind] = (state, surv)
        same = np.array_equal(mats["uru"][0], mats["tsu"][0])
        print(f"seed {seed}: state matrices identical={same}; survivors uru={mats['uru'][1]} tsu={mats['tsu'][1]}")
        res.append(same)
    print("learning exactness:", sum(res), "/", len(res))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "exact":
        exact(int(sys.argv[2]))
    elif cmd == "sp":
        sp(int(sys.argv[2]), sys.argv[4:], sys.argv[3])
    elif cmd == "learn":
        learn(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "learn_exact":
        learn_exact(int(sys.argv[2]), int(sys.argv[3]))
