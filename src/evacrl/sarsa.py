#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SARSA: the evacuation engine of `evacrl.core` with the temporal-difference update of its hook."""

import numpy as np

from evacrl.core import EvacuationModel


class SARSA(EvacuationModel):
    """Learns during the episode: every choice of a next node updates the action value of the previous choice."""

    def tdControl(self, pedIndx, alpha= 0.05):
        """
        On-policy temporal-difference update of "stateMat" during the episode, rather than at the end
        (the main change between SARSA and Monte Carlo). Called right after pedestrian pedIndx chose
        its next node: with S, A the state and action just chosen and S0, A0 the previous ones,

            Q(S0, A0) += alpha * (stepReward * dt + discount * Q(S, A) - Q(S0, A0))

        and, when the pedestrian has just reached an evacuation node, also
        Q(S, A) += alpha * (surviveReward - Q(S, A)). Each updated action's visit count is incremented.
        """
        trackPed = np.array(self.expeStat[pedIndx])
        # current and previous states
        current_S= trackPed[-1,0]
        pre_S= trackPed[-2,0]
        # current and previous actions
        current_A= trackPed[-1,1]
        pre_A= trackPed[-2,1]
        # current and pre time
        current_t = trackPed[-1,2]
        pre_t = trackPed[-2,2]
        
        if self.pedDB[pedIndx,10]:
            currentReward= self.surviveReward
            self.stateMat[current_S, 11 + current_A] += alpha * (currentReward - self.stateMat[current_S, 11 + current_A])
            self.stateMat[current_S, 21 + current_A] += 1
        
        preReward= self.stepReward * (current_t - pre_t)
        self.stateMat[pre_S, 11 + pre_A] += alpha * (preReward + self.discount * self.stateMat[current_S, 11 + current_A] - self.stateMat[pre_S, 11 + pre_A])
        self.stateMat[pre_S, 21 + pre_A] += 1
        return
