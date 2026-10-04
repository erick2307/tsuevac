# -*- coding: utf-8 -*-
"""Model settings that differ between the 2021 code (this repository) and the 2024 Kochi study.

The defaults reproduce the 2021 behaviour exactly (the golden tests pin it). Four fields name a behaviour that
was changed in `EVACMODEL3_FocalPoints/SARSA2024.py`; `ModelOptions.kochi2024()` switches those four and so
reproduces that code bit for bit. The fifth, `segmentIndex`, is a defect present in both codes.
`docs/engine-reconciliation.md` records what each one does to the results and which setting is recommended.
"""
from dataclasses import dataclass, replace

DENSITY_LEVELS = ("link", "segment")
ENTRY_SPEEDS = ("first_segment", "position")
SEGMENT_INDEXES = ("raw", "clamped")
SEGMENT_SIZINGS = ("ceil", "round")


@dataclass(frozen=True)
class ModelOptions:
    """
    surviveReward
        Reward credited when an agent reaches an evacuation node (2021: 1e5, 2024: 1e7). The step reward is
        -1 per second, so this only sets how strongly arriving outweighs the time spent getting there.
    densityLevel
        How the density code of a link in the *state* is computed.
        "link"    population on the link / (2 m x length): one number per link, the 2 m width is fixed
                  (2021).
        "segment" the largest of the per-segment density levels of the link, from the same histogram that
                  drives the walking speed, with the real link width, refreshed every 10 s (2024).
    entrySpeed
        Speed given to an agent when it enters a link.
        "first_segment" speed of the first segment of the link, whichever end the agent enters from (2021).
        "position"      speed of the segment where the agent actually is (2024).
    segmentSizing
        Number of segments (about 2 m each) a link is divided into for the density and speed histograms.
        "ceil"  ceil(length / 2) (2021).
        "round" round(length / 2), at least 1 (2024). The two differ on about a quarter of the links; on the Kochi
                shortest-path runs the effect is not distinguishable from noise.
    segmentIndex
        Which segment of the link an agent is in, from its distance to the link's first node.
        "raw"     floor(distance / segment length), as in 2021 and 2024. An agent standing at the far end of a
                  link whose straight-line length is at least its stored length gets index == number of
                  segments, one past the last one, where the speed array holds 0 (or 1.19, the free speed,
                  if the link was ever empty). With 0 the agent stops: it stays at that end of the link
                  until noise moves it, and every later agent entering from that end stops too.
        "clamped" the index is limited to the last segment. Recommended; it is the fix of that defect.
    """
    surviveReward: float = 100000
    densityLevel: str = "link"
    entrySpeed: str = "first_segment"
    segmentSizing: str = "ceil"
    segmentIndex: str = "raw"

    def __post_init__(self):
        if self.densityLevel not in DENSITY_LEVELS:
            raise ValueError(f"densityLevel must be one of {DENSITY_LEVELS}, not {self.densityLevel!r}")
        if self.entrySpeed not in ENTRY_SPEEDS:
            raise ValueError(f"entrySpeed must be one of {ENTRY_SPEEDS}, not {self.entrySpeed!r}")
        if self.segmentSizing not in SEGMENT_SIZINGS:
            raise ValueError(f"segmentSizing must be one of {SEGMENT_SIZINGS}, not {self.segmentSizing!r}")
        if self.segmentIndex not in SEGMENT_INDEXES:
            raise ValueError(f"segmentIndex must be one of {SEGMENT_INDEXES}, not {self.segmentIndex!r}")

    @classmethod
    def legacy(cls):
        """The 2021 behaviour (the default)."""
        return cls()

    @classmethod
    def kochi2024(cls):
        """The behaviour of `SARSA2024.py` (EVACMODEL3_FocalPoints), including its segment-index defect."""
        return cls(surviveReward=10000000, densityLevel="segment", entrySpeed="position", segmentSizing="round")

    def replace(self, **changes):
        return replace(self, **changes)
