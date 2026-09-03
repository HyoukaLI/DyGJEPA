from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.shape[0]
        if edge_index.numel() == 0:
            mean = torch.zeros_like(x)
        else:
            src, dst = edge_index
            mean = torch.zeros_like(x)
            mean.index_add_(0, dst, x[src])
            degree = torch.bincount(dst, minlength=n).to(x.dtype).clamp_min_(1.0)
            mean = mean / degree.unsqueeze(-1)
        return self.linear(torch.cat([x, mean], dim=-1))


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GraphSAGE requires at least one layer")
        dims = [in_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(GraphSAGELayer(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i + 1 < len(self.layers):
                x = F.relu(x)
        return x


class _SurrogateStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)  # type: ignore[attr-defined]
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor]:
        (x,) = ctx.saved_tensors  # type: ignore[attr-defined]
        # Arctangent surrogate derivative, stable and commonly used for SNN BPTT.
        return (grad_output / (1.0 + (torch.pi * x).square()),)


class PLIF(nn.Module):
    """Parametric LIF implementing paper equations (1)-(3)."""

    def __init__(self, dim: int, threshold: float = 0.5, reset: float = 0.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(dim))
        self.threshold = threshold
        self.reset = reset

    def forward(self, sequence: Tensor) -> Tensor:
        """Return spikes [time, nodes, dim] for input with the same shape."""
        voltage = torch.full_like(sequence[0], self.reset)
        spikes = []
        decay = torch.sigmoid(self.beta)
        for current in sequence:
            voltage = voltage + decay * (current - (voltage - self.reset))
            spike = _SurrogateStep.apply(voltage - self.threshold)
            voltage = voltage * (1.0 - spike) + self.reset * spike
            spikes.append(spike)
        return torch.stack(spikes)

