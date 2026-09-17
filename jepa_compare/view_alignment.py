"""Validation of the node-benchmark view lists.

This module deliberately imports nothing from the rest of the package and does
not import torch, so the configuration guard it holds can be unit-tested in an
environment with no deep-learning stack installed.
"""
from __future__ import annotations

from typing import Iterable


def require_aligned_views(
    checkpoint_views: Iterable[str], reported_views: Iterable[str]
) -> None:
    """Refuse a configuration whose checkpoint views differ from what is reported.

    Checkpoint selection and the reported test number must optimise the same
    functional.  An earlier version selected the epoch by the better of the two
    views scored separately while reporting their logit ensemble, so the chosen
    epoch was optimal for a quantity that never appeared in the paper.  Failing
    loudly here is what stops that from coming back silently.
    """
    checkpoint = tuple(checkpoint_views)
    reported = tuple(reported_views)
    if checkpoint != reported:
        raise ValueError(
            "rcps_checkpoint_views must equal rcps_multiscale_views so that the "
            "selected epoch is optimal for the reported ensemble; got "
            f"{checkpoint!r} vs {reported!r}"
        )
