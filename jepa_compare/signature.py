from __future__ import annotations

import torch
from torch import Tensor


def truncated_signature(increments: Tensor, depth: int = 2) -> Tensor:
    """Exact depth-1/2 signature of a concatenated piecewise-linear path.

    ``increments`` has shape ``[batch, steps, channels]``. The depth-two
    update follows Chen's identity: S2 <- S2 + S1⊗dx + 1/2 dx⊗dx.
    """
    if increments.ndim != 3:
        raise ValueError("increments must have shape [batch, steps, channels]")
    if depth not in {1, 2}:
        raise ValueError("this implementation supports signature depth 1 or 2")
    batch, _, channels = increments.shape
    first = torch.zeros(batch, channels, dtype=increments.dtype, device=increments.device)
    second = None
    if depth == 2:
        second = torch.zeros(
            batch, channels, channels, dtype=increments.dtype, device=increments.device
        )
    for delta in increments.unbind(dim=1):
        if second is not None:
            second = second + torch.einsum("bi,bj->bij", first, delta)
            second = second + 0.5 * torch.einsum("bi,bj->bij", delta, delta)
        first = first + delta
    if second is None:
        return first
    return torch.cat([first, second.flatten(start_dim=1)], dim=-1)


def signature_dimension(channels: int, depth: int) -> int:
    if depth == 1:
        return channels
    if depth == 2:
        return channels + channels * channels
    raise ValueError("this implementation supports signature depth 1 or 2")

