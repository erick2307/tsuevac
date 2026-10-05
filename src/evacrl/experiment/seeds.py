# -*- coding: utf-8 -*-
"""Reproducible seeds: every run of an experiment gets its own seed, derived from one base seed.

The engine draws from NumPy's global generator (`np.random`), so a run is made reproducible by seeding that generator at its
start. The seeds are the words of a `numpy.random.SeedSequence`, which makes them (1) the same for a given base seed whatever
the number of workers or of runs asked for (the first n seeds of a longer list are the n seeds of a shorter one) and (2) independent
between the *streams* of an experiment: the departure times of the shortest-path runs, the training episodes, the evaluation
episodes never share a seed.
"""
import contextlib

import numpy as np

STREAMS = {"shortest_path": 0, "training": 1, "evaluation": 2, "policy": 3}


@contextlib.contextmanager
def seeded(seed):
    """Seed NumPy's global generator, which the engine draws from, for the duration of the block, and put it back as it was: a run of the
    experiment layer does not change the random numbers of the program that calls it (in a worker process nobody would see the difference)."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def derive_seeds(base_seed, n, stream="shortest_path"):
    """`n` integer seeds (below 2**32, what `np.random.seed` takes) for `stream` of the experiment with `base_seed`."""
    if int(base_seed) < 0:
        raise ValueError("the base seed must not be negative")
    key = STREAMS[stream] if isinstance(stream, str) else int(stream)
    return [int(s) for s in np.random.SeedSequence([int(base_seed), key]).generate_state(n, dtype=np.uint32)]
