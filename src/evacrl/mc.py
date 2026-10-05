#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Monte Carlo: the evacuation engine of `evacrl.core` without a temporal-difference update.

The action values are updated once per simulation, from the experience of every agent, by
`updateValueFunctionDB` (the `tdControl` hook of the engine does nothing here).

Created on Fri Jan 11 16:31:19 2019
@author: Erick Mas
"""

from evacrl.core import EvacuationModel


class MonteCarlo(EvacuationModel):
    """Learns at the end of the simulation (see `updateValueFunctionDB`)."""
