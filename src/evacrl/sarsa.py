#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SARSA: the evacuation engine of `evacrl.core` with the on-policy temporal-difference update."""

from evacrl.td import TemporalDifference


class SARSA(TemporalDifference):
    """Learns during the episode, bootstrapping from the action the agent actually chose (see `evacrl.td`)."""

    def bootstrapValue(self, state, action, terminal):
        """Q(S, A) of the action that was chosen at S, exploration included."""
        return self.stateMat[state, 11 + action]
