# -*- coding: utf-8 -*-
"""Figures of the experiments: the evacuation curve of a learned policy against the shortest-path baseline, and the learning curve."""
import numpy as np


# Slots 1-3 of the reference categorical palette (validated for colour-vision deficiency, also all pairs): the shortest path is always
# blue, the learned policy orange; the third (exploring training runs) is aqua, which is below 3:1 on white, hence the legend.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt


def _style(ax, xlabel, ylabel, title=None):
    """Recessive grid and axes, text in ink and not in the series colours."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK)


def plot_comparison(path, sp, rl, *, agents=None, horizon=1800, title=None, sp_label="shortest path", rl_label="learned policy"):
    """Safe agents against time: for each of `sp` and `rl` (`(time, safe)` as `curves_matrix` makes them) the mean over its runs
    and the band between the 5th and 95th percentile. Time in minutes up to `horizon` seconds; with `agents`, in % of the agents."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    scale = 100.0 / agents if agents else 1.0
    end = int(horizon)
    for (time, safe), label, color in ((sp, sp_label, BLUE), (rl, rl_label, ORANGE)):
        x = np.asarray(time)[:end] / 60.0
        y = np.asarray(safe, dtype=float)[:, :end] * scale
        ax.plot(x, y.mean(axis=0), color=color, lw=1.8, label=f"{label}: mean of {len(y)} runs, band 5-95 %")
        ax.fill_between(x, np.percentile(y, 5, axis=0), np.percentile(y, 95, axis=0), color=color, alpha=0.18, lw=0)
    ax.set_xlim(0, end / 60.0)
    _style(ax, "time (min)", "safe (% of the agents)" if agents else "safe agents", title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_learning(path, history, *, reference=None, reference_label="shortest path", title=None):
    """The greedy evaluation of the checkpoints (mean, and +-1 sd) and the exploring training runs against the training simulations;
    `reference`, a number of safe agents, as a dashed line."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    sim, mean, sd = history["sim"], history["eval_mean"], np.nan_to_num(history["eval_sd"])
    ax.plot(sim, history["train_safe"], color=AQUA, lw=1.8, marker="o", ms=4, mec="white", mew=1, label="exploring runs (training)")
    ax.plot(sim, mean, color=ORANGE, lw=1.8, marker="o", ms=4, mec="white", mew=1, label="greedy policy (evaluation)")
    ax.fill_between(sim, mean - sd, mean + sd, color=ORANGE, alpha=0.18, lw=0)
    if "best" in history and history["best"].any():
        i = int(np.where(history["best"] > 0)[0][-1])
        ax.scatter([sim[i]], [mean[i]], s=80, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=5, label="best checkpoint")
    if reference is not None:
        ax.axhline(reference, color=BLUE, ls="--", lw=1.5, label=reference_label)
    _style(ax, "training simulations", "safe agents at the end of the simulation", title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def policy_arrows(nodes, next_nodes, next_sp=None):
    """`(move, differs)`: the nodes that have an arrow (not the shelters, and not those without a way on) and, for each, whether its
    next node is not the shortest path's (all False without `next_sp`)."""
    next_nodes = np.asarray(next_nodes)
    move = np.array([i for i in range(len(nodes)) if nodes[i, 3] != 1 and next_nodes[i] >= 0], dtype=int)
    differs = np.zeros(len(move), dtype=bool) if next_sp is None else (np.asarray(next_sp)[move] != next_nodes[move])
    return move, differs


def plot_policy(path, case, next_nodes, next_sp=None, *, title=None):
    """The network with an arrow at every node towards its next node. With `next_sp` (the shortest-path table) the arrows that
    differ from it are orange and the others blue; shelters are drawn as dark squares."""
    plt = _pyplot()
    from evacrl.tables import load_table
    nodes = load_table(case.kwargs["nodesdbFile"])
    links = load_table(case.kwargs["linksdbFile"], dtype=int)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for _, a, b, _, _ in links:
        ax.plot([nodes[a, 1], nodes[b, 1]], [nodes[a, 2], nodes[b, 2]], color=GRID, lw=0.8, zorder=1)
    move, differs = policy_arrows(nodes, next_nodes, next_sp)
    if len(move):
        dx = 0.35 * (nodes[next_nodes[move], 1] - nodes[move, 1])
        dy = 0.35 * (nodes[next_nodes[move], 2] - nodes[move, 2])
        for flag, color, label in ((False, BLUE, "as the shortest path" if next_sp is not None else "choice"), (True, ORANGE, "differs from the shortest path")):
            sel = differs == flag
            if sel.any():
                ax.quiver(nodes[move[sel], 1], nodes[move[sel], 2], dx[sel], dy[sel], color=color, angles="xy", scale_units="xy", scale=1,
                          width=0.0035, headwidth=4, headlength=4, zorder=3, label=f"{label} ({int(sel.sum())})")
    shelters = nodes[:, 3] == 1
    ax.scatter(nodes[shelters, 1], nodes[shelters, 2], marker="s", s=45, color=INK, zorder=4, label=f"shelter ({int(shelters.sum())})")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
