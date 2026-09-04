from __future__ import annotations

from typing import Any

from .dyglib_baselines import DyGLibLinkBaseline


class DyRepLinkBaseline(DyGLibLinkBaseline):
    """Compatibility name for the vendored, author-maintained DyGLib DyRep.

    New comparison code constructs :class:`DyGLibLinkBaseline` directly. This
    wrapper keeps older imports working while ensuring they run the same
    official ``MemoryModel(model_name="DyRep")`` implementation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "model_name" in kwargs:
            raise TypeError("DyRepLinkBaseline fixes model_name to 'dyrep'")
        super().__init__("dyrep", *args, **kwargs)
