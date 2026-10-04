#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-learning entry of the model (see `QLearning` for what that means here)."""

from evacrl.sarsa import SARSA


class QLearning(SARSA):
    """Identical to `SARSA`: the update rule is `SARSA.tdControl`, whose target is Q(S, A) of the action that
    was actually chosen next. That equals Q-learning's max over the actions only when the choice is greedy,
    which is how scripts/main_ql.py uses this class (randomChoiceRate = 0); scripts/main_ql_mod.py and
    scripts/main_sarsa.py explore (randomChoiceRate = 0.99). Override `tdControl` here to make the update
    off-policy (target: max over the actions at S) without touching SARSA.
    """
