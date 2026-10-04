# -*- coding: utf-8 -*-
"""Model settings that differ between the 2021 code (this repository) and the 2024 Kochi study.

`ModelOptions()` holds the recommended settings. `ModelOptions.legacy()` is the 2021 behaviour and
`ModelOptions.kochi2024()` is `EVACMODEL3_FocalPoints/SARSA2024.py` bit for bit (as `SARSA`: the 2024 code has no other
learning method, so `QLearning` and `MonteCarlo` get its dynamics and their own update); both are fully explicit, so they
do not change when the defaults do, and the golden tests pin them. `docs/engine-reconciliation.md` records what each
setting does to the results and why the defaults are what they are.
"""
from dataclasses import dataclass, replace

DENSITY_LEVELS = ("link", "segment")
ENTRY_SPEEDS = ("first_segment", "position")
SEGMENT_INDEXES = ("raw", "clamped")
SEGMENT_SIZINGS = ("ceil", "round")
DISCOUNTINGS = ("method", "decision", "second")


@dataclass(frozen=True)
class ModelOptions:
    """
    surviveReward
        Reward credited when an agent reaches an evacuation node (2021: 1e5, 2024: 1e7; default 1e5). The step
        reward is -1 per second, so this only sets how strongly arriving outweighs the time spent getting there.
    densityLevel
        How the density code of a link in the *state* is computed.
        "link"    population on the link / (2 m x length): one number per link, the 2 m width is fixed
                  (2021, default).
        "segment" the largest of the per-segment density levels of the link, from the same histogram that
                  drives the walking speed, with the real link width, refreshed every 10 s (2024).
    entrySpeed
        Speed given to an agent when it enters a link.
        "first_segment" speed of the first segment of the link, whichever end the agent enters from (2021).
        "position"      speed of the segment where the agent actually is (2024, default).
    discounting
        What the discount factor `discount` is applied to, in the return that the learning methods estimate.
        "decision" once per choice of a next node, whatever the time it takes (the temporal-difference methods, SARSA
                   and Q-learning, did this in 2021 and 2024).
        "second"   once per second: `discount ** seconds` (Monte Carlo did this in 2021). With discount = 0.9 a
                   reward 100 s ahead is weighted by 3e-5.
        "method"   each method keeps what it did before: "decision" for SARSA and Q-learning, "second" for Monte Carlo.
                   Default, until the choice is made (docs/audits/step2).
    segmentSizing
        Number of segments (about 2 m each) a link is divided into for the density and speed histograms.
        "ceil"  ceil(length / 2) (2021, default).
        "round" round(length / 2), at least 1 (2024). The two differ on about a quarter of the links; on the Kochi
                shortest-path runs the effect is not distinguishable from noise.
    segmentIndex
        Which segment of the link an agent is in, from its distance to the link's first node.
        "raw"     floor(distance / segment length), as in 2021 and 2024. An agent standing at the far end of a
                  link whose straight-line length is at least its stored length gets index == number of
                  segments, one past the last one, where the speed array holds 0 (or 1.19, the free speed,
                  if the link was ever empty). With 0 the agent stops: it stays at that end of the link
                  until noise moves it, and every later agent entering from that end stops too.
        "clamped" the index is limited to the last segment: the fix of that defect (default).
    """
    surviveReward: float = 100000
    densityLevel: str = "link"
    entrySpeed: str = "position"
    segmentSizing: str = "ceil"
    segmentIndex: str = "clamped"
    discounting: str = "method"

    def __post_init__(self):
        if self.densityLevel not in DENSITY_LEVELS:
            raise ValueError(f"densityLevel must be one of {DENSITY_LEVELS}, not {self.densityLevel!r}")
        if self.entrySpeed not in ENTRY_SPEEDS:
            raise ValueError(f"entrySpeed must be one of {ENTRY_SPEEDS}, not {self.entrySpeed!r}")
        if self.discounting not in DISCOUNTINGS:
            raise ValueError(f"discounting must be one of {DISCOUNTINGS}, not {self.discounting!r}")
        if self.segmentSizing not in SEGMENT_SIZINGS:
            raise ValueError(f"segmentSizing must be one of {SEGMENT_SIZINGS}, not {self.segmentSizing!r}")
        if self.segmentIndex not in SEGMENT_INDEXES:
            raise ValueError(f"segmentIndex must be one of {SEGMENT_INDEXES}, not {self.segmentIndex!r}")

    @classmethod
    def legacy(cls):
        """The 2021 behaviour: what the code did before the options existed (it contains the segment-index defect)."""
        return cls(surviveReward=100000, densityLevel="link", entrySpeed="first_segment", discounting="method",
                   segmentSizing="ceil", segmentIndex="raw")

    @classmethod
    def kochi2024(cls):
        """The dynamics of `SARSA2024.py` (EVACMODEL3_FocalPoints), including its segment-index defect; bit for bit with `SARSA`."""
        return cls(surviveReward=10000000, densityLevel="segment", entrySpeed="position", discounting="method",
                   segmentSizing="round", segmentIndex="raw")

    def replace(self, **changes):
        return replace(self, **changes)
