from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import DynamicGraph, Snapshot
from .encoding import random_walk_positional_encoding, sinusoidal_time_encoding
from .layers import GraphSAGE, PLIF


@dataclass
class WindowOutput:
    prediction: Tensor
    target: Tensor
    target_nodes: Tensor
    spike_rate: Tensor


class SGJEPA(nn.Module):
    """Paper-faithful implementation of SG-JEPA equations (5)-(16)."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        rwpe_dim: int = 8,
        rwpe_walks: int = 16,
        time_dim: int = 16,
        gnn_layers: int = 2,
        window_size: int = 4,
        threshold: float = 0.05,
        reset_potential: float = 0.0,
        temperature: float = 0.2,
        predictor_hidden_dim: int = 128,
        rwpe_seed: int = 42,
        cache_rwpe: bool = True,
    ) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.hidden_dim = hidden_dim
        self.rwpe_dim = rwpe_dim
        self.rwpe_walks = rwpe_walks
        self.rwpe_seed = rwpe_seed
        self.cache_rwpe = cache_rwpe
        self._rwpe_cache: dict[tuple[int, int, str], Tensor] = {}
        self.time_dim = time_dim
        self.window_size = window_size
        self.temperature = temperature
        self.encoder = GraphSAGE(feature_dim + rwpe_dim + time_dim, hidden_dim, gnn_layers)
        self.plif = PLIF(hidden_dim, threshold, reset_potential)
        # Eq. (12), W in R^((w-1)d x d). Prefix slices are shared by construction.
        self.prefix_projection = nn.Parameter(
            torch.empty((window_size - 1) * hidden_dim, hidden_dim)
        )
        nn.init.xavier_uniform_(self.prefix_projection)
        self.tokens = nn.Parameter(torch.randn(window_size - 1, hidden_dim) * 0.02)
        predictor_in = hidden_dim + time_dim + hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in, predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )
        self.pool_logits = nn.Parameter(torch.zeros(window_size - 1))

    def encode_snapshot(self, snapshot: Snapshot) -> Tensor:
        n = snapshot.x.shape[0]
        cache_key = (snapshot.time, snapshot.edge_index.shape[1], str(snapshot.x.device))
        rwpe = self._rwpe_cache.get(cache_key) if self.cache_rwpe else None
        if rwpe is None:
            rwpe = random_walk_positional_encoding(
                snapshot.edge_index, n, self.rwpe_dim, self.rwpe_walks,
                seed=self.rwpe_seed + snapshot.time,
            )
            if self.cache_rwpe:
                self._rwpe_cache[cache_key] = rwpe.detach()
        te = sinusoidal_time_encoding(snapshot.time, self.time_dim, snapshot.x.device)
        te = te.expand(n, -1)
        return self.encoder(torch.cat([snapshot.x, rwpe, te], dim=-1), snapshot.edge_index)

    def forward_window(self, window: list[Snapshot], precision: int | None = None) -> WindowOutput:
        if len(window) != self.window_size:
            raise ValueError(f"expected {self.window_size} snapshots")
        steps = self.window_size - 1 if precision is None else precision
        if not 1 <= steps <= self.window_size - 1:
            raise ValueError("precision must be in [1, window_size - 1]")
        embeddings = torch.stack([self.encode_snapshot(s) for s in window])
        target_nodes = window[-1].active.nonzero(as_tuple=False).flatten()
        if target_nodes.numel() == 0:
            raise ValueError("target snapshot has no active nodes")
        context = embeddings[:-1, target_nodes]
        spikes = self.plif(context)
        counts = spikes.cumsum(dim=0)
        candidates = []
        target_time = sinusoidal_time_encoding(window[-1].time, self.time_dim, embeddings.device)
        for t in range(1, steps + 1):
            nested = counts[:t].transpose(0, 1).reshape(target_nodes.numel(), t * self.hidden_dim)
            projected = nested @ self.prefix_projection[: t * self.hidden_dim]
            u = torch.cat(
                [projected, target_time.expand(target_nodes.numel(), -1), self.tokens[t - 1].expand(target_nodes.numel(), -1)],
                dim=-1,
            )
            candidates.append(self.predictor(u))
        weights = F.softmax(self.pool_logits[:steps], dim=0)
        prediction = (torch.stack(candidates, dim=0) * weights[:, None, None]).sum(dim=0)
        return WindowOutput(prediction, embeddings[-1, target_nodes], target_nodes, spikes.mean())

    def loss(
        self, graph: DynamicGraph, batch_size: int | None = None
    ) -> tuple[Tensor, dict[str, float]]:
        losses, rates = [], []
        for window in graph.windows(self.window_size):
            out = self.forward_window(window)
            pred = F.normalize(out.prediction, dim=-1)
            target = F.normalize(out.target.detach(), dim=-1)
            permutation = torch.randperm(pred.shape[0], device=pred.device)
            pred, target = pred[permutation], target[permutation]
            size = pred.shape[0] if batch_size is None else batch_size
            # Eq. (15) with in-batch target nodes bounds similarity at B x B.
            for start in range(0, pred.shape[0], size):
                p = pred[start : start + size]
                z = target[start : start + size]
                logits = p @ z.T / self.temperature
                labels = torch.arange(logits.shape[0], device=logits.device)
                losses.append(F.cross_entropy(logits, labels))
            rates.append(out.spike_rate.detach())
        if not losses:
            raise ValueError("graph does not contain one complete temporal window")
        loss = torch.stack(losses).mean()
        return loss, {"loss": loss.item(), "spike_rate": torch.stack(rates).mean().item()}

    @torch.no_grad()
    def infer(
        self,
        graph: DynamicGraph,
        precision: int | None = None,
        representation: str = "encoder",
    ) -> tuple[Tensor, Tensor]:
        """Return node embeddings for the frozen downstream probe.

        Self-supervised training remains predictive (context spikes → target).
        For DBLP-style node classification the final snapshot is available, so the
        default readout is the trained GraphSAGE encoder on that last snapshot.
        Pass ``representation="prediction"`` to use the spike-pooled JEPA output
        from the last temporal window instead.
        """
        if representation not in {"encoder", "prediction"}:
            raise ValueError("representation must be 'encoder' or 'prediction'")
        if len(graph.snapshots) < self.window_size:
            raise ValueError("not enough snapshots for inference")
        if representation == "encoder":
            last = graph.snapshots[-1]
            embeddings = self.encode_snapshot(last)
            node_ids = last.active.nonzero(as_tuple=False).flatten()
            if node_ids.numel() == 0:
                raise ValueError("target snapshot has no active nodes")
            return embeddings, node_ids
        window = graph.snapshots[-self.window_size :]
        out = self.forward_window(window, precision=precision)
        return out.prediction, out.target_nodes

    @torch.no_grad()
    def infer_node_views(self, graph: DynamicGraph) -> tuple[dict[str, Tensor], Tensor]:
        encoder, node_ids = self.infer(graph, representation="encoder")
        prediction, pred_ids = self.infer(graph, representation="prediction")
        if not torch.equal(node_ids, pred_ids):
            raise RuntimeError("encoder and prediction readouts disagree on active nodes")
        return {"encoder": encoder[node_ids], "prediction": prediction}, node_ids
