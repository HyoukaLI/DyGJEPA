from __future__ import annotations

"""Discrete-time self-supervised baselines with a shared frozen link probe.

CLDG is a dependency-free PyTorch port of the mechanisms in the authors'
public implementation: timespan views, a two-layer GCN, an MLP projection
head, and symmetric cross-view InfoNCE.  MaskDGNN and DVGMAE are paper-level
reimplementations because no publicly accessible author implementation could
be verified when this adapter was written.  Their provenance is deliberately
exposed in the result JSON; they must not be described as official-code runs.
"""

from abc import ABC, abstractmethod
import math
import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import Snapshot
from .link_prediction import (
    binary_average_precision,
    binary_roc_auc,
    canonical_pairs,
    grouped_ranking_metrics,
    sample_link_queries,
)


def _bidirected_unique(edge_index: Tensor, num_nodes: int) -> Tensor:
    pairs = canonical_pairs(edge_index, num_nodes, undirected=True)
    if pairs.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=edge_index.device)
    return torch.stack(
        [
            torch.cat([pairs[:, 0], pairs[:, 1]]),
            torch.cat([pairs[:, 1], pairs[:, 0]]),
        ]
    )


def _merge_snapshots(snapshots: Sequence[Snapshot]) -> tuple[Tensor, Tensor, Tensor]:
    if not snapshots:
        raise ValueError("at least one snapshot is required")
    num_nodes = snapshots[0].x.shape[0]
    device = snapshots[0].x.device
    keys = []
    for snapshot in snapshots:
        pairs = canonical_pairs(snapshot.edge_index, num_nodes, undirected=True)
        if pairs.numel():
            keys.append(pairs[:, 0] * num_nodes + pairs[:, 1])
    if keys:
        merged = torch.unique(torch.cat(keys), sorted=True)
        pairs = torch.stack([merged // num_nodes, merged % num_nodes], dim=-1)
        edge_index = torch.stack(
            [
                torch.cat([pairs[:, 0], pairs[:, 1]]),
                torch.cat([pairs[:, 1], pairs[:, 0]]),
            ]
        )
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    features = torch.stack([snapshot.x for snapshot in snapshots]).mean(dim=0)
    active = torch.stack([snapshot.active for snapshot in snapshots]).any(dim=0)
    return features, edge_index, active


class NormalizedGraphConv(nn.Module):
    """Sparse D^{-1/2}(A+I)D^{-1/2}XW without a PyG/DGL dependency."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = x.shape[0]
        loop = torch.arange(num_nodes, device=x.device)
        if edge_index.numel():
            src = torch.cat([edge_index[0], loop])
            dst = torch.cat([edge_index[1], loop])
        else:
            src = dst = loop
        out_degree = torch.bincount(src, minlength=num_nodes).to(x.dtype).clamp_min_(1)
        in_degree = torch.bincount(dst, minlength=num_nodes).to(x.dtype).clamp_min_(1)
        weight = out_degree[src].rsqrt() * in_degree[dst].rsqrt()
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, x[src] * weight.unsqueeze(-1))
        return self.linear(aggregated)


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("GCN needs at least one layer")
        dims = [in_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            NormalizedGraphConv(source, target)
            for source, target in zip(dims, dims[1:])
        )
        self.dropout = float(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for index, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if index + 1 < len(self.layers):
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return x


class PairMLP(nn.Module):
    """Common link probe used for every frozen SSL representation."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        left, right = embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]
        pair = torch.cat([left, right, left * right, (left - right).abs()], dim=-1)
        return self.network(pair).squeeze(-1)


class ConcatPairMLP(nn.Module):
    """Native masked-edge decoder used by the generative SSL objectives."""

    def __init__(self, embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: Tensor, pairs: Tensor) -> Tensor:
        endpoints = torch.cat(
            [embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]], dim=-1
        )
        return self.network(endpoints).squeeze(-1)


def _sample_negative_pairs(
    positive_pairs: Tensor,
    all_edges: Tensor,
    num_nodes: int,
    count: int,
    seed: int,
    bipartite_source_count: int | None,
) -> Tensor:
    if count < 1:
        return torch.empty(0, 2, dtype=torch.long, device=positive_pairs.device)
    forbidden_pairs = canonical_pairs(all_edges, num_nodes, undirected=True).cpu()
    forbidden = {
        int(left) * num_nodes + int(right)
        for left, right in forbidden_pairs.tolist()
    }
    generator = torch.Generator().manual_seed(seed)
    rows: list[tuple[int, int]] = []
    seen: set[int] = set()
    attempts = 0
    max_attempts = max(2_000, count * 100)
    while len(rows) < count and attempts < max_attempts:
        if bipartite_source_count is None:
            left = int(torch.randint(num_nodes, (1,), generator=generator).item())
            right = int(torch.randint(num_nodes, (1,), generator=generator).item())
        else:
            left = int(
                torch.randint(bipartite_source_count, (1,), generator=generator).item()
            )
            right = int(
                torch.randint(
                    bipartite_source_count, num_nodes, (1,), generator=generator
                ).item()
            )
        attempts += 1
        if left == right:
            continue
        lo, hi = min(left, right), max(left, right)
        key = lo * num_nodes + hi
        if key in forbidden or key in seen:
            continue
        seen.add(key)
        rows.append((left, right))
    if len(rows) < count:
        raise RuntimeError("unable to sample SSL reconstruction negatives")
    return torch.tensor(rows, dtype=torch.long, device=positive_pairs.device)


def _subsample_rows(rows: Tensor, count: int, seed: int) -> Tensor:
    if rows.shape[0] <= count:
        return rows
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(rows.shape[0], generator=generator)[:count]
    return rows[order.to(rows.device)]


class SnapshotSSLLinkBaseline(nn.Module, ABC):
    implementation: str = "paper_reimplementation"

    def __init__(
        self,
        embedding_dim: int,
        probe_hidden_dim: int,
        negative_ratio: float,
        max_positive_pairs: int | None,
        new_edges_only: bool,
        undirected: bool,
        bipartite_source_count: int | None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.probe = PairMLP(self.embedding_dim, int(probe_hidden_dim))
        self.negative_ratio = float(negative_ratio)
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = bool(new_edges_only)
        self.undirected = bool(undirected)
        self.bipartite_source_count = bipartite_source_count
        self.encoder_frozen = False

    @abstractmethod
    def pretrain_epoch(
        self,
        snapshots: Sequence[Snapshot],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int,
    ) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def encode_context(self, snapshots: Sequence[Snapshot]) -> Tensor:
        raise NotImplementedError

    def pretrain_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("probe.") and parameter.requires_grad
        ]

    def freeze_encoder(self) -> None:
        for name, parameter in self.named_parameters():
            if not name.startswith("probe."):
                parameter.requires_grad_(False)
        self.encoder_frozen = True

    def sample_queries(self, window: Sequence[Snapshot], seed: int):
        return sample_link_queries(
            window[-1],
            window[-2] if len(window) > 1 else None,
            negative_ratio=self.negative_ratio,
            max_positive=self.max_positive_pairs,
            seed=seed,
            new_edges_only=self.new_edges_only,
            undirected=self.undirected,
            bipartite_source_count=self.bipartite_source_count,
        )

    def train_probe_epoch(
        self,
        windows: Sequence[Sequence[Snapshot]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        pair_batch_size: int,
        seed: int,
    ) -> dict[str, float]:
        if not self.encoder_frozen:
            raise RuntimeError("freeze the SSL encoder before fitting the link probe")
        self.eval()
        self.probe.train()
        total_loss = 0.0
        examples = 0
        probability_sum = 0.0
        for window_index, window in enumerate(windows):
            with torch.no_grad():
                embeddings = self.encode_context(window[:-1]).detach()
            queries = self.sample_queries(window, seed + window_index)
            for start in range(0, queries.pairs.shape[0], pair_batch_size):
                pairs = queries.pairs[start : start + pair_batch_size]
                labels = queries.labels[start : start + pair_batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = self.probe(embeddings, pairs)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.probe.parameters(), grad_clip)
                optimizer.step()
                count = labels.numel()
                total_loss += float(loss.detach().item()) * count
                probability_sum += float(logits.detach().sigmoid().sum().item())
                examples += count
        if not examples:
            raise ValueError("no frozen-probe training examples were produced")
        return {
            "loss": total_loss / examples,
            "mean_probability": probability_sum / examples,
            "examples": float(examples),
        }

    @torch.no_grad()
    def evaluate_windows(
        self,
        windows: Sequence[Sequence[Snapshot]],
        pair_batch_size: int | None = None,
        query_seed: int = 42,
    ) -> dict[str, float]:
        self.eval()
        batch_size = int(pair_batch_size or 4096)
        probabilities, labels, groups = [], [], []
        group_offset = 0
        for window_index, window in enumerate(windows):
            embeddings = self.encode_context(window[:-1])
            queries = self.sample_queries(window, query_seed + window_index)
            for start in range(0, queries.pairs.shape[0], batch_size):
                pairs = queries.pairs[start : start + batch_size]
                probabilities.append(self.probe(embeddings, pairs).sigmoid())
                labels.append(queries.labels[start : start + batch_size])
                groups.append(
                    queries.group_ids[start : start + batch_size] + group_offset
                )
            group_offset += int(queries.group_ids.max().item()) + 1
        probability = torch.cat(probabilities)
        target = torch.cat(labels)
        group_ids = torch.cat(groups)
        mrr, recall_at_10 = grouped_ranking_metrics(
            target, probability, group_ids, recall_k=10
        )
        return {
            "ap": binary_average_precision(target, probability),
            "auc": binary_roc_auc(target, probability),
            "mrr": mrr,
            "recall_at_10": recall_at_10,
            "mean_probability": float(probability.mean().item()),
            "examples": float(target.numel()),
        }


class CLDGLinkBaseline(SnapshotSSLLinkBaseline):
    """CLDG official-mechanism port followed by a common frozen link probe."""

    implementation = "official_mechanism_port@bdbc1eb9"

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_spans: int = 4,
        num_views: int = 4,
        view_strategy: str = "sequential",
        temperature: float = 0.07,
        contrastive_batch_size: int = 1024,
        probe_hidden_dim: int = 128,
        **link_kwargs: object,
    ) -> None:
        super().__init__(embedding_dim, probe_hidden_dim, **link_kwargs)
        self.encoder = GCNEncoder(feature_dim, hidden_dim, num_layers, dropout)
        self.readout = nn.Linear(hidden_dim, embedding_dim)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2),
        )
        self.num_spans = int(num_spans)
        self.num_views = int(num_views)
        self.view_strategy = str(view_strategy)
        self.temperature = float(temperature)
        self.contrastive_batch_size = int(contrastive_batch_size)

    def _encode(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = self.encoder(features, edge_index)
        return F.leaky_relu(F.normalize(self.readout(hidden), dim=-1), 0.2)

    def _project(self, embeddings: Tensor) -> Tensor:
        first = self.projector[0](embeddings)
        first = self.projector[1](F.normalize(first, dim=-1))
        second = self.projector[2](first)
        return self.projector[3](F.normalize(second, dim=-1))

    def _view_ranges(self, length: int, seed: int) -> list[tuple[int, int]]:
        if length < 1:
            raise ValueError("CLDG needs at least one training snapshot")
        spans = min(self.num_spans, length)
        views = min(self.num_views, spans)
        width = max(1, math.ceil(length / spans))
        rng = random.Random(seed)
        if self.view_strategy == "sequential":
            starts = list(range(0, length, width))[:spans]
            starts = rng.sample(starts, views)
        elif self.view_strategy == "random":
            starts = [rng.randrange(max(1, length - width + 1)) for _ in range(views)]
        elif self.view_strategy in {"low_overlap", "high_overlap"}:
            step_ratio = 0.75 if self.view_strategy == "low_overlap" else 0.25
            step = max(1, round(width * step_ratio))
            required = width + step * (views - 1)
            first = rng.randrange(max(1, length - required + 1))
            starts = [min(length - 1, first + index * step) for index in range(views)]
        else:
            raise ValueError(f"unknown CLDG view strategy: {self.view_strategy}")
        return [(start, min(length, start + width)) for start in starts]

    def pretrain_epoch(
        self,
        snapshots: Sequence[Snapshot],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int,
    ) -> dict[str, float]:
        self.train()
        encoded, node_sets = [], []
        for start, end in self._view_ranges(len(snapshots), seed):
            features, edges, _ = _merge_snapshots(snapshots[start:end])
            encoded.append(self._project(self._encode(features, edges)))
            node_sets.append(set(torch.unique(edges).detach().cpu().tolist()))
        common = set.intersection(*node_sets) if node_sets else set()
        if len(common) < 2:
            common = set(range(snapshots[0].x.shape[0]))
        nodes = torch.tensor(sorted(common), dtype=torch.long)
        if nodes.numel() > self.contrastive_batch_size:
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(nodes.numel(), generator=generator)
            nodes = nodes[order[: self.contrastive_batch_size]]
        nodes = nodes.to(snapshots[0].x.device)
        labels = torch.arange(nodes.numel(), device=nodes.device)
        losses = []
        for left in range(len(encoded)):
            for right in range(left + 1, len(encoded)):
                first, second = encoded[left][nodes], encoded[right][nodes]
                logits = first @ second.T / self.temperature
                reverse = second @ first.T / self.temperature
                losses.append(
                    0.5
                    * (F.cross_entropy(logits, labels) + F.cross_entropy(reverse, labels))
                )
        if not losses:
            raise ValueError("CLDG requires at least two temporal views")
        loss = torch.stack(losses).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pretrain_parameters(), grad_clip)
        optimizer.step()
        return {
            "loss": float(loss.detach().item()),
            "view_pairs": float(len(losses)),
            "contrastive_nodes": float(nodes.numel()),
        }

    def encode_context(self, snapshots: Sequence[Snapshot]) -> Tensor:
        features, edges, _ = _merge_snapshots(snapshots)
        return self._encode(features, edges)


def _page_rank(edge_index: Tensor, num_nodes: int, damping: float, steps: int) -> Tensor:
    device = edge_index.device
    rank = torch.full((num_nodes,), 1.0 / num_nodes, device=device)
    if edge_index.numel() == 0:
        return rank
    src, dst = edge_index
    degree = torch.bincount(src, minlength=num_nodes).to(rank.dtype).clamp_min_(1)
    teleport = (1.0 - damping) / num_nodes
    for _ in range(steps):
        updated = torch.full_like(rank, teleport)
        updated.index_add_(0, dst, damping * rank[src] / degree[src])
        rank = updated
    return rank


def _minmax(values: Tensor) -> Tensor:
    minimum, maximum = values.min(), values.max()
    return (values - minimum) / (maximum - minimum).clamp_min(1e-8)


class MaskDGNNLinkBaseline(SnapshotSSLLinkBaseline):
    """Paper-level MaskDGNN reimplementation for DTDG snapshots."""

    implementation = "paper_reimplementation_ijcai2025"

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 4,
        mask_ratio: float = 0.3,
        dynamics_ratio: float = 0.7,
        dynamics_weight: float = 1.0,
        existing_offset: float = 2.0,
        new_node_offset: float = -0.5,
        pagerank_damping: float = 0.85,
        pagerank_steps: int = 10,
        pretrain_pair_limit: int = 4096,
        probe_hidden_dim: int = 128,
        **link_kwargs: object,
    ) -> None:
        super().__init__(hidden_dim, probe_hidden_dim, **link_kwargs)
        if not 0 < mask_ratio < 1:
            raise ValueError("mask_ratio must be in (0, 1)")
        self.encoder = GCNEncoder(feature_dim, hidden_dim, num_layers, dropout)
        frequency_count = window_size // 2 + 1
        self.frequency_real = nn.Parameter(torch.ones(frequency_count, hidden_dim))
        self.frequency_imag = nn.Parameter(torch.zeros(frequency_count, hidden_dim))
        self.reconstruction = ConcatPairMLP(hidden_dim, hidden_dim)
        self.window_size = int(window_size)
        self.mask_ratio = float(mask_ratio)
        self.dynamics_ratio = float(dynamics_ratio)
        self.dynamics_weight = float(dynamics_weight)
        self.existing_offset = float(existing_offset)
        self.new_node_offset = float(new_node_offset)
        self.pagerank_damping = float(pagerank_damping)
        self.pagerank_steps = int(pagerank_steps)
        self.pretrain_pair_limit = int(pretrain_pair_limit)

    def _activeness(self, current: Snapshot, previous: Snapshot | None) -> Tensor:
        num_nodes = current.x.shape[0]
        current_edges = _bidirected_unique(current.edge_index, num_nodes)
        current_degree = torch.bincount(
            current_edges[0], minlength=num_nodes
        ).to(current.x.dtype)
        if previous is None:
            previous_degree = torch.zeros_like(current_degree)
            changed = current_degree
            existed = torch.zeros(num_nodes, dtype=torch.bool, device=current.x.device)
        else:
            previous_pairs = canonical_pairs(previous.edge_index, num_nodes, True)
            current_pairs = canonical_pairs(current.edge_index, num_nodes, True)
            previous_keys = set(
                (previous_pairs[:, 0] * num_nodes + previous_pairs[:, 1])
                .detach()
                .cpu()
                .tolist()
            )
            current_keys = set(
                (current_pairs[:, 0] * num_nodes + current_pairs[:, 1])
                .detach()
                .cpu()
                .tolist()
            )
            changed_degree = torch.zeros(num_nodes, device=current.x.device)
            for key in previous_keys.symmetric_difference(current_keys):
                left, right = divmod(int(key), num_nodes)
                changed_degree[left] += 1
                changed_degree[right] += 1
            changed = changed_degree
            previous_edges = _bidirected_unique(previous.edge_index, num_nodes)
            previous_degree = torch.bincount(
                previous_edges[0], minlength=num_nodes
            ).to(current.x.dtype)
            existed = previous.active
        dynamics = self.dynamics_weight * changed / (
            previous_degree.sqrt() + self.existing_offset
        )
        new_dynamics = self.dynamics_weight * changed / (
            changed.sqrt().clamp_min(1e-6) + self.new_node_offset
        ).abs().clamp_min(1e-6)
        dynamics = torch.where(existed, dynamics, new_dynamics)
        significance = _page_rank(
            current_edges, num_nodes, self.pagerank_damping, self.pagerank_steps
        )
        return self.dynamics_ratio * _minmax(dynamics) + (
            1.0 - self.dynamics_ratio
        ) * _minmax(significance)

    def _mask_snapshot(
        self, current: Snapshot, previous: Snapshot | None, seed: int
    ) -> tuple[Tensor, Tensor]:
        num_nodes = current.x.shape[0]
        pairs = canonical_pairs(current.edge_index, num_nodes, undirected=True)
        if pairs.shape[0] < 2:
            return _bidirected_unique(current.edge_index, num_nodes), pairs
        activity = self._activeness(current, previous)
        probability = 1.0 - 0.5 * (
            activity[pairs[:, 0]] + activity[pairs[:, 1]]
        )
        desired = max(1, round(pairs.shape[0] * self.mask_ratio))
        probability = (probability * desired / probability.sum().clamp_min(1e-8)).clamp(
            0, 1
        )
        generator = torch.Generator().manual_seed(seed)
        mask = torch.rand(pairs.shape[0], generator=generator).to(pairs.device) < probability
        if not mask.any():
            mask[torch.argmax(probability)] = True
        if mask.all():
            mask[torch.argmin(probability)] = False
        retained = pairs[~mask]
        retained_edges = torch.stack(
            [
                torch.cat([retained[:, 0], retained[:, 1]]),
                torch.cat([retained[:, 1], retained[:, 0]]),
            ]
        )
        return retained_edges, pairs[mask]

    def _frequency_enhance(self, sequence: Tensor) -> Tensor:
        spectrum = torch.fft.rfft(sequence, dim=0)
        frequencies = spectrum.shape[0]
        weight = torch.complex(
            self.frequency_real[:frequencies], self.frequency_imag[:frequencies]
        ).unsqueeze(1)
        enhanced = torch.fft.irfft(spectrum * weight, n=sequence.shape[0], dim=0)
        return sequence + F.dropout(enhanced, self.encoder.dropout, self.training)

    def _pretrain_window(
        self, snapshots: Sequence[Snapshot], seed: int
    ) -> tuple[Tensor, int]:
        representations, masked_by_time = [], []
        for index, snapshot in enumerate(snapshots):
            previous = None if index == 0 else snapshots[index - 1]
            retained, masked = self._mask_snapshot(snapshot, previous, seed + index)
            representations.append(self.encoder(snapshot.x, retained))
            masked_by_time.append(masked)
        enhanced = self._frequency_enhance(torch.stack(representations))
        losses, examples = [], 0
        for index, (snapshot, positive) in enumerate(zip(snapshots, masked_by_time)):
            positive = _subsample_rows(
                positive, self.pretrain_pair_limit, seed + 10_000 + index
            )
            if positive.numel() == 0:
                continue
            negative = _sample_negative_pairs(
                positive,
                snapshot.edge_index,
                snapshot.x.shape[0],
                positive.shape[0],
                seed + 20_000 + index,
                self.bipartite_source_count,
            )
            pairs = torch.cat([positive, negative])
            labels = torch.cat(
                [
                    torch.ones(positive.shape[0], device=pairs.device),
                    torch.zeros(negative.shape[0], device=pairs.device),
                ]
            )
            logits = self.reconstruction(enhanced[index], pairs)
            losses.append(F.binary_cross_entropy_with_logits(logits, labels))
            examples += labels.numel()
        if not losses:
            raise ValueError("MaskDGNN produced no masked reconstruction examples")
        return torch.stack(losses).mean(), examples

    def pretrain_epoch(
        self,
        snapshots: Sequence[Snapshot],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int,
    ) -> dict[str, float]:
        self.train()
        total, windows, examples = 0.0, 0, 0
        for start in range(0, len(snapshots), self.window_size):
            window = snapshots[start : start + self.window_size]
            if len(window) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, count = self._pretrain_window(window, seed + start * 100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pretrain_parameters(), grad_clip)
            optimizer.step()
            total += float(loss.detach().item())
            examples += count
            windows += 1
        if not windows:
            raise ValueError("MaskDGNN needs a pretraining window with at least two snapshots")
        return {"loss": total / windows, "examples": float(examples)}

    def encode_context(self, snapshots: Sequence[Snapshot]) -> Tensor:
        selected = snapshots[-self.window_size :]
        encoded = [
            self.encoder(snapshot.x, _bidirected_unique(snapshot.edge_index, snapshot.x.shape[0]))
            for snapshot in selected
        ]
        return self._frequency_enhance(torch.stack(encoded))[-1]


class DVGMAELinkBaseline(SnapshotSSLLinkBaseline):
    """Paper-level dynamic variational graph masked autoencoder."""

    implementation = "paper_reimplementation_tnnls2025"

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 4,
        mask_ratio: float = 0.3,
        history_balance: float = 0.5,
        kl_weight: float = 0.001,
        feature_weight: float = 0.1,
        pretrain_pair_limit: int = 4096,
        probe_hidden_dim: int = 128,
        **link_kwargs: object,
    ) -> None:
        super().__init__(hidden_dim, probe_hidden_dim, **link_kwargs)
        self.encoder = GCNEncoder(feature_dim, hidden_dim, num_layers, dropout)
        self.mean_head = nn.Linear(hidden_dim, hidden_dim)
        self.log_variance_head = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.global_decoder = nn.Linear(hidden_dim, hidden_dim)
        self.edge_decoder = ConcatPairMLP(hidden_dim, hidden_dim)
        self.feature_decoder = nn.Linear(hidden_dim, feature_dim)
        self.window_size = int(window_size)
        self.mask_ratio = float(mask_ratio)
        self.history_balance = float(history_balance)
        self.kl_weight = float(kl_weight)
        self.feature_weight = float(feature_weight)
        self.pretrain_pair_limit = int(pretrain_pair_limit)

    def _temporal_masks(
        self, snapshots: Sequence[Snapshot], seed: int
    ) -> list[tuple[Tensor, Tensor]]:
        num_nodes = snapshots[0].x.shape[0]
        history_total: dict[int, int] = {}
        history_masked: dict[int, int] = {}
        outputs = []
        for index, snapshot in enumerate(snapshots):
            pairs = canonical_pairs(snapshot.edge_index, num_nodes, True)
            if pairs.shape[0] < 2:
                outputs.append((_bidirected_unique(snapshot.edge_index, num_nodes), pairs))
                continue
            keys = (pairs[:, 0] * num_nodes + pairs[:, 1]).detach().cpu().tolist()
            history_rate = torch.tensor(
                [
                    history_masked.get(int(key), 0)
                    / max(1, history_total.get(int(key), 0))
                    for key in keys
                ],
                dtype=snapshot.x.dtype,
                device=snapshot.x.device,
            )
            priority = 1.0 + self.history_balance * (1.0 - history_rate)
            desired = max(1, round(pairs.shape[0] * self.mask_ratio))
            probability = (priority * desired / priority.sum()).clamp(0, 1)
            generator = torch.Generator().manual_seed(seed + index)
            mask = torch.rand(pairs.shape[0], generator=generator).to(pairs.device) < probability
            if not mask.any():
                mask[torch.argmax(probability)] = True
            if mask.all():
                mask[torch.argmin(probability)] = False
            for position, key in enumerate(keys):
                integer_key = int(key)
                history_total[integer_key] = history_total.get(integer_key, 0) + 1
                history_masked[integer_key] = history_masked.get(integer_key, 0) + int(
                    mask[position].item()
                )
            retained = pairs[~mask]
            retained_edges = torch.stack(
                [
                    torch.cat([retained[:, 0], retained[:, 1]]),
                    torch.cat([retained[:, 1], retained[:, 0]]),
                ]
            )
            outputs.append((retained_edges, pairs[mask]))
        return outputs

    def _decode_sequence(self, latent: Tensor, active: Sequence[Tensor]) -> Tensor:
        # GRU sees each node as a temporal sequence.  A graph-level summary is
        # added at every time step, matching DVGMAE's globally enhanced decoder.
        temporal, _ = self.temporal_decoder(latent.transpose(0, 1))
        temporal = temporal.transpose(0, 1)
        global_rows = []
        for index, mask in enumerate(active):
            selected = latent[index][mask]
            global_rows.append(selected.mean(dim=0) if selected.numel() else latent[index].mean(dim=0))
        global_context = self.global_decoder(torch.stack(global_rows)).unsqueeze(1)
        return temporal + global_context

    def _encode_variational(
        self, snapshots: Sequence[Snapshot], edges: Sequence[Tensor], sample: bool
    ) -> tuple[Tensor, Tensor, Tensor]:
        means, log_variances = [], []
        for snapshot, edge_index in zip(snapshots, edges):
            hidden = self.encoder(snapshot.x, edge_index)
            means.append(self.mean_head(hidden))
            log_variances.append(self.log_variance_head(hidden).clamp(-8, 8))
        mean = torch.stack(means)
        log_variance = torch.stack(log_variances)
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
        else:
            latent = mean
        decoded = self._decode_sequence(latent, [snapshot.active for snapshot in snapshots])
        return decoded, mean, log_variance

    def _pretrain_window(
        self, snapshots: Sequence[Snapshot], seed: int
    ) -> tuple[Tensor, dict[str, float]]:
        masks = self._temporal_masks(snapshots, seed)
        decoded, mean, log_variance = self._encode_variational(
            snapshots, [retained for retained, _ in masks], sample=True
        )
        reconstruction_losses, feature_losses, examples = [], [], 0
        for index, (snapshot, (_, positive)) in enumerate(zip(snapshots, masks)):
            positive = _subsample_rows(
                positive, self.pretrain_pair_limit, seed + 10_000 + index
            )
            if positive.numel():
                negative = _sample_negative_pairs(
                    positive,
                    snapshot.edge_index,
                    snapshot.x.shape[0],
                    positive.shape[0],
                    seed + 20_000 + index,
                    self.bipartite_source_count,
                )
                pairs = torch.cat([positive, negative])
                labels = torch.cat(
                    [
                        torch.ones(positive.shape[0], device=pairs.device),
                        torch.zeros(negative.shape[0], device=pairs.device),
                    ]
                )
                logits = self.edge_decoder(decoded[index], pairs)
                reconstruction_losses.append(
                    F.binary_cross_entropy_with_logits(logits, labels)
                )
                examples += labels.numel()
            feature_losses.append(
                F.mse_loss(self.feature_decoder(decoded[index]), snapshot.x)
            )
        reconstruction = torch.stack(reconstruction_losses).mean()
        feature = torch.stack(feature_losses).mean()
        kl = -0.5 * torch.mean(1 + log_variance - mean.square() - log_variance.exp())
        loss = reconstruction + self.feature_weight * feature + self.kl_weight * kl
        return loss, {
            "reconstruction_loss": float(reconstruction.detach().item()),
            "feature_loss": float(feature.detach().item()),
            "kl_loss": float(kl.detach().item()),
            "examples": float(examples),
        }

    def pretrain_epoch(
        self,
        snapshots: Sequence[Snapshot],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int,
    ) -> dict[str, float]:
        self.train()
        sums = {"loss": 0.0, "reconstruction_loss": 0.0, "feature_loss": 0.0, "kl_loss": 0.0}
        windows, examples = 0, 0.0
        for start in range(0, len(snapshots), self.window_size):
            window = snapshots[start : start + self.window_size]
            if len(window) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._pretrain_window(window, seed + start * 100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pretrain_parameters(), grad_clip)
            optimizer.step()
            sums["loss"] += float(loss.detach().item())
            for key in ("reconstruction_loss", "feature_loss", "kl_loss"):
                sums[key] += metrics[key]
            examples += metrics["examples"]
            windows += 1
        if not windows:
            raise ValueError("DVGMAE needs a pretraining window with at least two snapshots")
        return {key: value / windows for key, value in sums.items()} | {
            "examples": examples
        }

    def encode_context(self, snapshots: Sequence[Snapshot]) -> Tensor:
        selected = snapshots[-self.window_size :]
        edges = [
            _bidirected_unique(snapshot.edge_index, snapshot.x.shape[0])
            for snapshot in selected
        ]
        decoded, _, _ = self._encode_variational(selected, edges, sample=False)
        return decoded[-1]
