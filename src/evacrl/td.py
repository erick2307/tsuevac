#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The temporal-difference update shared by SARSA and Q-learning.

Both learn while the simulation runs: every time an agent chooses its next node, the value of its *previous*
choice is moved towards what that choice turned out to be worth. With S0, A0 the previous state and action,
S, A the current ones and dt the seconds between the two choices,

    Q(S0, A0) += alpha * (stepReward * dt + gamma * V(S, A) - Q(S0, A0))

and, if the agent has just reached an evacuation node, first Q(S, A) += alpha * (surviveReward - Q(S, A)): arriving
is the end of the episode, there is nothing to bootstrap from beyond it. The two methods differ only in V(S, A),
the value they bootstrap from, which is the `bootstrapValue` hook:

* SARSA (on-policy)     V = Q(S, A): the action the agent actually chose, exploration included
* Q-learning (off-policy) V = max over the actions available at S: what the best choice would be worth,
                          whatever the agent chose
"""
import numpy as np

from evacrl.core import EvacuationModel


class TemporalDifference(EvacuationModel):
    """Evacuation engine with the temporal-difference update; use `SARSA` or `QLearning`, not this class."""

    defaultDiscounting = "decision"

    def bootstrapValue(self, state, action, terminal):
        """V(S, A): the value the previous choice is moved towards, besides the reward (see the module docstring).
        `terminal`: the agent has just reached an evacuation node, `action` is its self-loop there."""
        raise NotImplementedError("use SARSA or QLearning")

    def tdControl(self, pedIndx, alpha= 0.05):
        """
        Called right after pedestrian pedIndx chose its next node (its experience, including this choice, is the
        last item of `self.expeStat[pedIndx]`; the one before is the previous choice).
        Each updated action's visit count is incremented.
        """
        trackPed = np.array(self.expeStat[pedIndx][-2:])  # only the previous and the current choice are needed
        # current and previous states
        current_S= trackPed[-1,0]
        pre_S= trackPed[-2,0]
        # current and previous actions
        current_A= trackPed[-1,1]
        pre_A= trackPed[-2,1]
        # current and pre time
        current_t = trackPed[-1,2]
        pre_t = trackPed[-2,2]

        terminal= bool(self.pedDB[pedIndx,10])
        if terminal:
            currentReward= self.surviveReward
            self.stateMat[current_S, 11 + current_A] += alpha * (currentReward - self.stateMat[current_S, 11 + current_A])
            self.stateMat[current_S, 21 + current_A] += 1

        preReward= self.stepReward * (current_t - pre_t)
        target= preReward + self.discountFactor(current_t - pre_t) * self.bootstrapValue(current_S, current_A, terminal)
        self.stateMat[pre_S, 11 + pre_A] += alpha * (target - self.stateMat[pre_S, 11 + pre_A])
        self.stateMat[pre_S, 21 + pre_A] += 1
        return
