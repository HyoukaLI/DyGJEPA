from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import Snapshot
from .temporal_event_utils import (
    EventStream,
    HarmonicTimeEncoder,
    MergeLayer,
    SharedLinkProtocol,
    TemporalAttentionLayer,
    TemporalNeighborIndex,
    seeded_torch_generator,
    stream_events,
    unique_snapshots,
)


class TGATLinkBaseline(nn.Module, SharedLinkProtocol):
    """TGAT internals evaluated with the suite's shared link protocol.

    The implementation follows the official TGAT recursion: functional time
    encoding, temporal-neighbor sampling, edge-aware multi-head attention and
    the merge-layer affinity decoder.  Only the temporal split, evaluation
    candidates, metrics and validation checkpoint rule are shared with the
    other models.
    """

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        bipartite_source_count: int | None,
        interaction_feature_dim: int = 172,
        num_layers: int = 2,
        num_heads: int = 2,
        num_neighbors: int = 20,
        dropout: float = 0.1,
        uniform_neighbors: bool = True,
        train_batch_size: int = 200,
        eval_group_batch_size: int = 16,
        negative_ratio: float = 20.0,
        max_positive_pairs: int | None = 1024,
        new_edges_only: bool = False,
        undirected: bool = False,
    ) -> None:
        super().__init__()
        del feature_dim
        if undirected:
            raise ValueError("TGAT event evaluation requires directed links")
        if bipartite_source_count is not None and not 0 < bipartite_source_count < num_nodes:
            raise ValueError("bipartite_source_count must split users and items")
        if num_layers < 1 or num_neighbors < 1 or train_batch_size < 1:
            raise ValueError("TGAT layer, neighbor and batch counts must be positive")
        if (2 * interaction_feature_dim) % num_heads:
            raise ValueError("twice the TGAT feature dimension must divide into heads")

        self.num_nodes = num_nodes
        self.num_users = bipartite_source_count
        self.num_items = (
            None if bipartite_source_count is None else num_nodes - bipartite_source_count
        )
        self.dimension = interaction_feature_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.uniform_neighbors = uniform_neighbors
        self.train_batch_size = train_batch_size
        self.eval_group_batch_size = eval_group_batch_size
        self.negative_ratio = negative_ratio
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = new_edges_only

        # The official Wikipedia preprocessing supplies 172-D edge features
        # and zero node features of the same width.  The final row is padding.
        self.register_buffer(
            "node_features",
            torch.zeros(num_nodes + 1, interaction_feature_dim),
            persistent=False,
        )
        self.register_buffer(
            "edge_features",
            torch.zeros(1, interaction_feature_dim),
            persistent=False,
        )
        self.time_encoder = HarmonicTimeEncoder(interaction_feature_dim)
        self.attention_layers = nn.ModuleList(
            [
                TemporalAttentionLayer(
                    interaction_feature_dim,
                    interaction_feature_dim,
                    interaction_feature_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.affinity = MergeLayer(
            interaction_feature_dim,
            interaction_feature_dim,
            interaction_feature_dim,
            1,
        )

        self._train_stream: EventStream | None = None
        self._train_index: TemporalNeighborIndex | None = None
        self._full_index: TemporalNeighborIndex | None = None

    def prepare_streams(
        self,
        all_snapshots: Sequence[Snapshot],
        train_snapshots: Sequence[Snapshot],
    ) -> None:
        full = stream_events(
            all_snapshots,
            num_users=self.num_users,
            num_nodes=self.num_nodes,
            feature_dim=self.dimension,
        )
        train = stream_events(
            train_snapshots,
            num_users=self.num_users,
            num_nodes=self.num_nodes,
            feature_dim=self.dimension,
        )
        self._train_stream = train
        self._train_index = TemporalNeighborIndex(train, self.num_nodes)
        self._full_index = TemporalNeighborIndex(full, self.num_nodes)
        padding = torch.zeros(
            1, self.dimension, dtype=full.features.dtype, device=full.features.device
        )
        self.edge_features = torch.cat([full.features, padding], dim=0)

    def _temporal_embedding(
        self,
        nodes: Tensor,
        times: Tensor,
        layers: int,
        index: TemporalNeighborIndex,
        rng: np.random.Generator,
    ) -> Tensor:
        source = self.node_features[nodes]
        if layers == 0:
            return source
        neighbor_nodes, neighbor_events, neighbor_times, mask = index.sample(
            nodes,
            times,
            self.num_neighbors,
            uniform=self.uniform_neighbors,
            rng=rng,
            device=nodes.device,
        )
        source_previous = self._temporal_embedding(
            nodes, times, layers - 1, index, rng
        )
        flat_neighbors = neighbor_nodes.reshape(-1)
        flat_times = neighbor_times.reshape(-1)
        neighbor_previous = self._temporal_embedding(
            flat_neighbors, flat_times, layers - 1, index, rng
        ).reshape(nodes.shape[0], self.num_neighbors, self.dimension)
        delta = (times.unsqueeze(1) - neighbor_times).clamp_min(0.0)
        neighbor_time = self.time_encoder(delta)
        source_time = self.time_encoder(times.new_zeros(nodes.shape[0], 1))
        edge_features = self.edge_features[neighbor_events]
        return self.attention_layers[layers - 1](
            source_previous,
            source_time,
            neighbor_previous,
            neighbor_time,
            edge_features,
            mask,
        )

    def _score_pairs(
        self,
        sources: Tensor,
        destinations: Tensor,
        times: Tensor,
        index: TemporalNeighborIndex,
        rng: np.random.Generator,
    ) -> Tensor:
        if sources.numel() == 0:
            return torch.empty(0, dtype=times.dtype, device=times.device)
        source = self._temporal_embedding(
            sources, times, self.num_layers, index, rng
        )
        destination = self._temporal_embedding(
            destinations, times, self.num_layers, index, rng
        )
        return self.affinity(source, destination).squeeze(-1)

    def train_epoch(
        self,
        windows: Sequence[Sequence[Snapshot]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int = 42,
    ) -> dict[str, float]:
        del windows
        if self._train_stream is None or self._train_index is None:
            raise RuntimeError("call prepare_streams before TGAT training")
        stream, index = self._train_stream, self._train_index
        generator, random_device = seeded_torch_generator(
            stream.sources.device, seed
        )
        order = torch.randperm(
            len(stream), generator=generator, device=random_device
        ).to(stream.sources.device)
        rng = np.random.default_rng(seed)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(stream), self.train_batch_size):
            rows = order[start : start + self.train_batch_size]
            if rows.numel() == 0:
                continue
            sources = stream.sources[rows]
            positives = stream.destinations[rows]
            times = stream.timestamps[rows]
            negative_start = 0 if self.num_users is None else self.num_users
            negatives = torch.randint(
                negative_start, self.num_nodes, positives.shape,
                generator=generator, device=random_device,
            ).to(positives.device)
            collision = negatives == positives
            if self.num_users is None:
                collision |= negatives == sources
            while collision.any():
                negatives[collision] = torch.randint(
                    negative_start,
                    self.num_nodes,
                    (int(collision.sum()),),
                    generator=generator,
                    device=random_device,
                ).to(positives.device)
                collision = negatives == positives
                if self.num_users is None:
                    collision |= negatives == sources

            optimizer.zero_grad(set_to_none=True)
            positive_logits = self._score_pairs(sources, positives, times, index, rng)
            negative_logits = self._score_pairs(sources, negatives, times, index, rng)
            loss = F.binary_cross_entropy_with_logits(
                positive_logits, torch.ones_like(positive_logits)
            ) + F.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().item())
            batches += 1
        return {"loss": total_loss / max(1, batches)}

    @torch.no_grad()
    def evaluate_protocol(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int = 42,
    ) -> dict[str, float]:
        del history_windows
        if self._full_index is None:
            raise RuntimeError("call prepare_streams before TGAT evaluation")
        all_scores: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_groups: list[Tensor] = []
        group_offset = 0
        rng = np.random.default_rng(query_seed)
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(window, query_seed + window_index)
            groups = torch.unique(queries.group_ids, sorted=True)
            for start in range(0, groups.numel(), self.eval_group_batch_size):
                selected = groups[start : start + self.eval_group_batch_size]
                row_parts = [
                    torch.nonzero(queries.group_ids == group, as_tuple=False).flatten()
                    for group in selected
                ]
                rows = torch.cat(row_parts)
                logits = self._score_pairs(
                    queries.pairs[rows, 0],
                    queries.pairs[rows, 1],
                    queries.timestamps[rows]
                    if queries.timestamps is not None
                    else queries.labels[rows].new_full(
                        (rows.numel(),), float(window[-1].time)
                    ),
                    self._full_index,
                    rng,
                )
                all_scores.append(torch.sigmoid(logits))
                all_labels.append(queries.labels[rows])
                remapped = []
                for local_group, part in enumerate(row_parts):
                    remapped.append(
                        torch.full_like(part, group_offset + local_group)
                    )
                all_groups.append(torch.cat(remapped))
                group_offset += len(row_parts)
        if not all_scores:
            raise ValueError("TGAT evaluation produced no queries")
        return self.metrics(
            torch.cat(all_labels), torch.cat(all_scores), torch.cat(all_groups)
        )


def prepare_tgat_streams(
    model: TGATLinkBaseline,
    all_snapshots: Sequence[Snapshot],
    train_windows: Sequence[Sequence[Snapshot]],
) -> None:
    model.prepare_streams(all_snapshots, unique_snapshots(train_windows))
