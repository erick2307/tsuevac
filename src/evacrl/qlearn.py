#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Q-learning: the evacuation engine of `evacrl.core` with the off-policy temporal-difference update."""

from evacrl.td import TemporalDifference


class QLearning(TemporalDifference):
    """Learns during the episode, bootstrapping from the best action at the next state (see `evacrl.td`).

    The target ignores which action the agent chose at S: it is the maximum of Q(S, .) over the actions that exist
    at that node. The unused slots of the state matrix hold 0, which would beat every negative value, so only the
    first `transLinkdb[node, 1]` slots are looked at. When the agents always choose the best action (no
    exploration) this is the SARSA update, to the last digit; with exploration the two differ.

    On reaching an evacuation node the episode ends: there is no "best next action", the value is that of the
    self-loop the agent chose (the only action of an evacuation node in every case table of this repository, but
    that is the data's convention, not the update's).
    """

    def bootstrapValue(self, state, action, terminal):
        """max over the actions available at S of Q(S, a); `action`, the one chosen, plays no role, except at the end."""
        if terminal:
            return self.stateMat[state, 11 + action]
        node = int(self.stateMat[state, 0])
        numActions = int(self.transLinkdb[node, 1])
        if numActions == 0:
            return 0.0
        return self.stateMat[state, 11:11 + numActions].max()
