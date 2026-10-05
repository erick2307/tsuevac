# -*- coding: utf-8 -*-
"""Command line of the experiments.

    python -m evacrl.experiment sp        CASE --runs 100 | --until-converged   --out DIR   shortest-path runs (the baseline)
    python -m evacrl.experiment calibrate CASE --method qlearning --sims 300    --out DIR   training, checkpoints evaluated greedily
    python -m evacrl.experiment evaluate  CASE --state DIR/best_state.csv --runs 50 --out DIR2   greedy runs of a stored policy
    python -m evacrl.experiment compare   --sp DIR --rl DIR2 --out figure.png                  the two evacuation curves
    python -m evacrl.experiment policy    CASE --state DIR/best_state.csv --out map.png         the policy's walks against the shortest path

CASE is the name of a folder of `cases/` (`kochi_area2`) or a path to a case folder. Times are in minutes. `sp`, `calibrate` and `evaluate` write a
`manifest.json` (case checksums, options, seeds, versions) next to their results; the same `--seed` gives the same results for any
`--workers`.
"""
import argparse
import dataclasses
import os
import sys
import time

import numpy as np

from evacrl.experiment import output
from evacrl.experiment.training import SCHEDULES, calibrate, evaluate_policy
from evacrl.experiment.case import Case
from evacrl.experiment.manifest import build_manifest, write_manifest
from evacrl.experiment.plots import plot_comparison, plot_learning, plot_policy
from evacrl.experiment.policy import compare_with_shortest_path, greedy_next_nodes
from evacrl.experiment.runs import METHODS
from evacrl.experiment.sp import repeat_shortest_path
from evacrl.options import ModelOptions

PRESETS = {"default": ModelOptions, "legacy": ModelOptions.legacy, "kochi2024": ModelOptions.kochi2024}


def options_from(args):
    options = PRESETS[args.preset]()
    changes = {}
    types = {f.name: f.type for f in dataclasses.fields(ModelOptions)}
    for item in args.set or []:
        key, _, value = item.partition("=")
        if key not in types or not value:
            raise SystemExit(f"--set expects KEY=VALUE with KEY one of {', '.join(types)}")
        changes[key] = float(value) if types[key] is float else value
    return options.replace(**changes) if changes else options


def _common(p, kind):
    p.add_argument("case", help="a folder of cases/ or a path to a case folder")
    p.add_argument("--out", required=True, help="folder for the results (created)")
    p.add_argument("--seed", type=int, default=0, help="base seed: every run's seed is derived from it (default 0)")
    p.add_argument("--workers", type=int, default=1, help="processes; the results do not depend on it")
    p.add_argument("--departure", type=float, default=5.0, help="mean departure time in minutes (default 5)")
    p.add_argument("--preset", choices=sorted(PRESETS), default="default", help="model options: the recommended ones, 2021 or the 2024 study")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="change one model option, e.g. --set discounting=second")
    p.add_argument("--popfile", type=int, default=1, help="population file number of a 2024-layout folder")
    if kind != "sp":
        p.add_argument("--method", choices=sorted(METHODS), default="qlearning")
        p.add_argument("--discount", type=float, default=None, help="discount factor (default: the one of the model options, 0.999 per second)")


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m evacrl.experiment", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sp", help="repeated shortest-path runs")
    _common(sp, "sp")
    sp.add_argument("--runs", type=int, help="a fixed number of runs")
    sp.add_argument("--until-converged", action="store_true", help="add batches of runs until the running mean and CV settle")
    sp.add_argument("--max-runs", type=int, default=1000)
    sp.add_argument("--min-runs", type=int, default=30)
    sp.add_argument("--batch", type=int, default=10)
    sp.add_argument("--tol", type=float, default=0.01, help="relative standard error and change of the running mean (default 1 %%)")
    sp.add_argument("--time", type=float, default=120.0, help="minutes simulated (default 120)")
    sp.add_argument("--horizon", type=float, default=30.0, help="minutes at which 'safe' is reported (default 30)")

    cal = sub.add_parser("calibrate", help="train a policy, evaluating checkpoints greedily")
    _common(cal, "calibrate")
    cal.add_argument("--sims", type=int, default=300)
    cal.add_argument("--schedule", choices=SCHEDULES, default="calibration")
    cal.add_argument("--eval-every", type=int, default=25)
    cal.add_argument("--eval-runs", type=int, default=5)
    cal.add_argument("--time", type=float, default=30.0, help="minutes simulated per episode (default 30)")
    cal.add_argument("--restart-from-best", action="store_true",
                     help="after a checkpoint that is not better than the best, train on from the best checkpoint rather than the latest state")
    cal.add_argument("--eval-keep-learning", action="store_true", help="the agents go on learning during the evaluation runs (by default the policy is frozen)")
    cal.add_argument("--sp", help="folder of an `sp` run: its mean is drawn as the reference of the learning curve")

    ev = sub.add_parser("evaluate", help="greedy runs of a stored policy")
    _common(ev, "evaluate")
    ev.add_argument("--state", required=True, help="state matrix file (best_state.csv of a calibration)")
    ev.add_argument("--runs", type=int, default=50)
    ev.add_argument("--time", type=float, default=30.0)
    ev.add_argument("--keep-learning", action="store_true",
                    help="the agents go on learning during the run (on a copy): the policy then adapts as it goes; by default it is frozen")

    pol = sub.add_parser("policy", help="the walks of a stored policy against the shortest path, and a map of its choices")
    pol.add_argument("case")
    pol.add_argument("--state", required=True, help="state matrix file (best_state.csv of a calibration)")
    pol.add_argument("--out", help="figure file (.png): an arrow at every node, orange where the choice differs from the shortest path")
    pol.add_argument("--popfile", type=int, default=1)

    cmp_ = sub.add_parser("compare", help="draw the shortest-path and the learned evacuation curves")
    cmp_.add_argument("--sp", required=True, help="folder written by `sp`")
    cmp_.add_argument("--rl", required=True, help="folder written by `evaluate`")
    cmp_.add_argument("--out", required=True, help="figure file (.png)")
    cmp_.add_argument("--horizon", type=float, default=30.0, help="minutes shown (default 30)")
    cmp_.add_argument("--title")
    return parser


def _prepare_out(folder):
    """Make the result folder, or stop with a message if that cannot be done (a file of that name, no permission)."""
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"error: cannot use --out {folder}: {exc.strerror or exc}")


def _seconds(minutes):
    """Minutes as a whole number of seconds, rounded (`int(2.05 * 60)` would be 122)."""
    return int(round(minutes * 60))


def _sd(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else None


def _sha256(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _summary(label, values, unit="s"):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return f"{label}: {v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else float('nan'):.1f} {unit} (n = {len(v)})" if len(v) else f"{label}: none"


def main(argv=None):
    """Run one command; a value the experiments cannot use (a negative seed, no runs, a state of another case) ends with a message."""
    try:
        return _main(argv)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")


def _main(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    started = time.time()
    command = "python -m evacrl.experiment " + " ".join(argv)

    if args.command == "compare":
        sp_time, sp_safe, sp_agents = output.read_curves(args.sp)
        rl_time, rl_safe, rl_agents = output.read_curves(args.rl)
        plot_comparison(args.out, (sp_time, sp_safe), (rl_time, rl_safe), agents=int(sp_agents[0]) if sp_agents[0] == rl_agents[0] else None,
                        horizon=_seconds(args.horizon), title=args.title)
        print(f"wrote {args.out}")
        return 0

    case = Case.load(args.case, popfile=args.popfile)
    if args.command == "policy":
        state = np.loadtxt(args.state, delimiter=",")
        c = compare_with_shortest_path(case, state)
        print(f"{case.name}: {c.agents} agents; the policy's first choice is the shortest path's for {100 * c.agreement:.1f} %; "
              f"walk {c.metres:.0f} m against {c.metres_sp:.0f} m ({c.longer:+.1f} %), {c.hops:.1f} nodes against {c.hops_sp:.1f}; "
              f"{c.never_arrive} agents never arrive")
        if args.out:
            from evacrl.tables import load_table
            plot_policy(args.out, case, greedy_next_nodes(case, state), None if case.nextnode is None else load_table(case.nextnode, dtype=int)[:, 1],
                        title=f"{case.name}: greedy choice at every node")
            print(f"wrote {args.out}")
        return 0
    options = options_from(args)
    if args.command in ("sp", "calibrate", "evaluate"):
        _prepare_out(args.out)                     # a result folder that cannot be made is found out before the computation, not after it
    sp_curves = None
    if args.command == "calibrate" and args.sp:
        try:
            sp_curves = output.read_curves(args.sp)
        except OSError as exc:
            raise SystemExit(f"error: --sp {args.sp} is not the folder of an `sp` run ({exc})")

    if args.command == "sp":
        if args.runs is None and not args.until_converged:
            raise SystemExit("give --runs N or --until-converged")
        horizon = _seconds(args.horizon)
        if horizon > _seconds(args.time):
            raise SystemExit(f"error: --horizon ({args.horizon:g} min) is later than the end of the simulation (--time {args.time:g} min)")
        res = repeat_shortest_path(case, args.runs, until_converged=args.until_converged, max_runs=args.max_runs, min_runs=args.min_runs,
                                   batch=args.batch, tol=args.tol, horizon=horizon, seed=args.seed, workers=args.workers, options=options,
                                   sim_time=_seconds(args.time), mean_departure=args.departure)
        output.write_runs(args.out, res.runs, horizon)
        output.write_convergence(args.out, res.trace)
        last, safe = res.metric("last_evacuee"), res.metric("safe", horizon)
        arrived = last[~np.isnan(last)]
        mean_or_none = lambda x, f: float(f(x)) if len(x) else None
        summary = dict(runs=len(res.runs), converged=res.converged, agents=res.runs[0].agents, incomplete_runs=res.incomplete,
                       last_evacuee_mean_s=mean_or_none(arrived, np.mean), last_evacuee_sd_s=_sd(arrived),
                       safe_mean=float(safe.mean()), safe_sd=_sd(safe), horizon_s=horizon,
                       cv_last_evacuee=res.trace["last_evacuee"][-1].cv)
        write_manifest(args.out, build_manifest("shortest_path", case, dict(vars(args), tolerance=args.tol), options=options,
                                                seeds=dict(base=args.seed, runs=res.seeds), results=summary, started=started, argv=[command]))
        print(f"{case.name}: {len(res.runs)} shortest-path runs, {res.runs[0].agents} agents, converged: {res.converged}")
        print(_summary("last evacuee", last))
        if res.incomplete:
            print(f"warning: {res.incomplete} of {len(res.runs)} runs ended with agents still walking after {args.time:g} min: for those the evacuation "
                  f"time is the last arrival within the simulation, not the time of the last agent; simulate longer (--time)")
        print(_summary(f"safe at {args.horizon:g} min", safe, "agents"))
        cv = summary["cv_last_evacuee"]
        print(f"CV of the evacuation time: {'-' if cv is None or np.isnan(cv) else format(cv, '.4f')}; results in {args.out}")
        return 0

    if args.command == "calibrate":
        log = lambda c: print(f"sim {c.sim:5d}  epsilon {c.epsilon:.3f}  exploring {c.train_safe:7.1f}  greedy {c.eval_mean:7.1f} ± {c.eval_sd:5.1f}{'  *' if c.best else ''}", flush=True)
        res = calibrate(case, method=args.method, sims=args.sims, schedule=args.schedule, eval_every=args.eval_every, eval_runs=args.eval_runs,
                        sim_time=_seconds(args.time), mean_departure=args.departure, options=options, discount=args.discount, seed=args.seed,
                        workers=args.workers, restart_from_best=args.restart_from_best, eval_learn=args.eval_keep_learning, progress=log)
        output.write_history(args.out, res.history)
        output.write_state(os.path.join(args.out, "best_state.csv"), res.best_state)
        output.write_state(os.path.join(args.out, "final_state.csv"), res.final_state)
        reference = None
        if sp_curves is not None:
            _, safe, _ = sp_curves
            reference = float(safe[:, min(_seconds(args.time), safe.shape[1]) - 1].mean())
        discount = options.discount if args.discount is None else args.discount
        plot_learning(os.path.join(args.out, "learning.png"), output.read_history(args.out), reference=reference,
                      title=f"{case.name}: {args.method}, discount {discount:g} ({options.discounting})")
        summary = dict(best_sim=res.best_sim, best_eval=res.best_eval, agents=res.agents, states=int(res.best_state.shape[0]), sp_reference=reference,
                       discount=discount, discounting=options.discounting)
        write_manifest(args.out, build_manifest("calibration", case, vars(args), options=options, seeds=res.seeds, results=summary, started=started, argv=[command]))
        print(f"best checkpoint: after {res.best_sim} simulations, {res.best_eval:.1f} of {res.agents} safe (greedy); results in {args.out}")
        return 0

    if args.command == "evaluate":
        state = np.loadtxt(args.state, delimiter=",")
        runs = evaluate_policy(case, args.method, state, args.runs, seed=args.seed, workers=args.workers, options=options,
                               sim_time=_seconds(args.time), mean_departure=args.departure, discount=args.discount, learn=args.keep_learning)
        output.write_runs(args.out, runs, _seconds(args.time))
        safe = np.array([r.evacuated for r in runs], dtype=float)
        write_manifest(args.out, build_manifest("evaluation", case, dict(vars(args), state_sha256=_sha256(args.state)), options=options,
                                                seeds=dict(base=args.seed, runs=[r.seed for r in runs]),
                                                results=dict(runs=len(runs), agents=runs[0].agents, safe_mean=float(safe.mean()), safe_sd=_sd(safe)),
                                                started=started, argv=[command]))
        print(f"{case.name}: {len(runs)} greedy runs of {args.state} ({'learning on' if args.keep_learning else 'frozen'}): " + _summary("safe", safe, "agents") + f"; results in {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
