from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_time_encoding(time: int, dim: int, device: torch.device) -> Tensor:
    """Equation (7): deterministic sinusoidal encoding for one snapshot."""
    if dim <= 0:
        return torch.empty(0, device=device)
    positions = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    rates = torch.exp(-math.log(10_000.0) * positions / max(dim, 1))
    out = torch.zeros(dim, device=device)
    out[0::2] = torch.sin(float(time) * rates)
    if dim > 1:
        out[1::2] = torch.cos(float(time) * rates[: out[1::2].numel()])
    return out


def random_walk_positional_encoding(
    edge_index: Tensor, num_nodes: int, dim: int, walks: int = 16, seed: int = 42
) -> Tensor:
    """Monte-Carlo RWPE return probabilities without an N x N matrix.

    Each feature estimates ``diag(P^k)`` from ``walks`` paths per node. This is a
    scalable estimator of the paper's random-walk positional encoding; isolated
    nodes receive zero encodings.
    """
    device = edge_index.device
    if dim <= 0:
        return torch.empty(num_nodes, 0, device=device)
    if edge_index.numel() == 0 or walks <= 0:
        return torch.zeros(num_nodes, dim, device=device)
    src, dst = edge_index
    order = torch.argsort(src)
    src, dst = src[order], dst[order]
    degree = torch.bincount(src, minlength=num_nodes)
    offsets = torch.zeros(num_nodes, dtype=torch.long, device=device)
    offsets[1:] = degree.cumsum(0)[:-1]
    start = torch.arange(num_nodes, device=device).repeat_interleave(walks)
    current = start.clone()
    generator = torch.Generator(device=device).manual_seed(seed)
    features = []
    for _ in range(dim):
        deg = degree[current]
        movable = deg > 0
        choice = (
            torch.rand(current.numel(), device=device, generator=generator)
            * deg.clamp_min(1)
        ).long()
        next_node = current.clone()
        next_node[movable] = dst[offsets[current[movable]] + choice[movable]]
        current = next_node
        returned = (current == start).view(num_nodes, walks).float().mean(dim=1)
        returned[degree == 0] = 0
        features.append(returned)
    return torch.stack(features, dim=-1)
