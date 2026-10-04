# -*- coding: utf-8 -*-
"""The learning-curve figure of the Step 4 audit, from gamma_learning_kochi2.json (run gamma_learning.py first)."""
import json
import os

import numpy as np
from common import HERE

from evacrl.experiment.plots import AQUA, BLUE, INK, ORANGE, _pyplot, _style

if __name__ == "__main__":
    runs = json.load(open(os.path.join(HERE, "gamma_learning_kochi2.json")))
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    labels = {("decision", 0.9): ("discount 0.9 per decision (2021, 2024)", ORANGE), ("second", 0.999): ("discount 0.999 per second", AQUA)}
    seen = set()
    for r in runs:
        label, color = labels[(r["discounting"], r["discount"])]
        h = np.array(r["history"])
        ax.plot(h[:, 0], h[:, 3], color=color, lw=1.8, marker="o", ms=4, mec="white", mew=1, label=None if label in seen else label)
        seen.add(label)
    ax.axhline(529.0, color=BLUE, ls="--", lw=1.5, label="shortest path")
    ax.set_ylim(400, 540)
    _style(ax, "training simulations", "safe agents at 30 min (greedy policy, 5 runs)", "kochi2, Q-learning: the greedy policy after training (622 agents)")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "learning_curves.png"), dpi=150)
