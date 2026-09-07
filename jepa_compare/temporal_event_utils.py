from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .data import Snapshot
from .link_prediction import (
    LinkQueries,
    link_prediction_metrics,
    sample_link_queries,
)


@dataclass(frozen=True)
class EventStream:
    sources: Tensor
    destinations: Tensor
    timestamps: Tensor
    features: Tensor

    def __len__(self) -> int:
        return int(self.sources.shape[0])


def seeded_torch_generator(
    device: torch.device, seed: int
) -> tuple[torch.Generator, torch.device]:
    """Create a deterministic generator on the compute device when supported.

    CUDA random sampling stays on the GPU.  PyTorch does not provide a custom
    MPS generator, so MPS retains the compatible CPU-generator-and-copy path.
    """
    random_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=random_device).manual_seed(seed), random_device


def unique_snapshots(
    windows: Sequence[Sequence[Snapshot]], *, targets_only: bool = False
) -> list[Snapshot]:
    snapshots: dict[int, Snapshot] = {}
    for window in windows:
        selected = [window[-1]] if targets_only else window
        for snapshot in selected:
            snapshots[snapshot.time] = snapshot
    return [snapshots[key] for key in sorted(snapshots)]


def snapshot_events(
    snapshot: Snapshot,
    *,
    num_users: int | None,
    num_nodes: int,
    feature_dim: int,
) -> EventStream:
    if snapshot.query_edge_index is None:
        raise ValueError("continuous-time baselines require query_edge_index")
    source, destination = snapshot.query_edge_index
    if num_users is None:
        valid = (
            (source >= 0)
            & (source < num_nodes)
            & (destination >= 0)
            & (destination < num_nodes)
        )
    else:
        valid = (
            (source < num_users)
            & (destination >= num_users)
            & (destination < num_nodes)
        )
    source = source[valid]
    destination = destination[valid]
    if snapshot.query_timestamps is None:
        timestamps = torch.full(
            (source.shape[0],),
            float(snapshot.time),
            dtype=snapshot.x.dtype,
            device=snapshot.x.device,
        )
    else:
        timestamps = snapshot.query_timestamps[valid]
    if snapshot.query_features is None:
        features = torch.zeros(
            source.shape[0],
            feature_dim,
            dtype=snapshot.x.dtype,
            device=snapshot.x.device,
        )
    else:
        features = snapshot.query_features[valid]
        if features.shape[1] < feature_dim:
            features = torch.nn.functional.pad(
                features, (0, feature_dim - features.shape[1])
            )
        elif features.shape[1] > feature_dim:
            raise ValueError(
                f"interaction feature width is {features.shape[1]}, expected {feature_dim}"
            )
    order = torch.argsort(timestamps, stable=True)
    return EventStream(
        source[order], destination[order], timestamps[order], features[order]
    )


def stream_events(
    snapshots: Sequence[Snapshot],
    *,
    num_users: int | None,
    num_nodes: int,
    feature_dim: int,
) -> EventStream:
    entries = [
        snapshot_events(
            snapshot,
            num_users=num_users,
            num_nodes=num_nodes,
            feature_dim=feature_dim,
        )
        for snapshot in snapshots
    ]
    if not entries:
        raise ValueError("event stream requires at least one snapshot")
    sources = torch.cat([entry.sources for entry in entries])
    destinations = torch.cat([entry.destinations for entry in entries])
    timestamps = torch.cat([entry.timestamps for entry in entries])
    features = torch.cat([entry.features for entry in entries])
    order = torch.argsort(timestamps, stable=True)
    return EventStream(
        sources[order], destinations[order], timestamps[order], features[order]
    )


class TemporalNeighborIndex:
    """CPU temporal adjacency with strictly-before-time neighbor queries."""

    def __init__(self, stream: EventStream, num_nodes: int) -> None:
        entries: list[list[tuple[float, int, int]]] = [[] for _ in range(num_nodes)]
        for event_id, (source, destination, timestamp) in enumerate(
            zip(
                stream.sources.detach().cpu().tolist(),
                stream.destinations.detach().cpu().tolist(),
                stream.timestamps.detach().cpu().tolist(),
            )
        ):
            entries[source].append((timestamp, destination, event_id))
            entries[destination].append((timestamp, source, event_id))
        self.timestamps: list[np.ndarray] = []
        self.neighbors: list[np.ndarray] = []
        self.event_ids: list[np.ndarray] = []
        for node_entries in entries:
            node_entries.sort(key=lambda value: (value[0], value[2]))
            self.timestamps.append(
                np.asarray([value[0] for value in node_entries], dtype=np.float64)
            )
            self.neighbors.append(
                np.asarray([value[1] for value in node_entries], dtype=np.int64)
            )
            self.event_ids.append(
                np.asarray([value[2] for value in node_entries], dtype=np.int64)
            )
        self.num_nodes = num_nodes
        self.padding_node = num_nodes
        self.padding_event = len(stream)

    def sample(
        self,
        nodes: Tensor,
        cut_times: Tensor,
        count: int,
        *,
        uniform: bool,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if count < 1:
            raise ValueError("neighbor count must be positive")
        if nodes.numel() != cut_times.numel():
            raise ValueError(
                "nodes and cut_times must contain the same number of rows"
            )
        if nodes.numel() == 0:
            shape = (0, count)
            neighbor_nodes = torch.empty(shape, dtype=torch.long, device=device)
            neighbor_events = torch.empty(shape, dtype=torch.long, device=device)
            neighbor_times = torch.empty(
                shape, dtype=cut_times.dtype, device=device
            )
            padding_mask = torch.empty(shape, dtype=torch.bool, device=device)
            return neighbor_nodes, neighbor_events, neighbor_times, padding_mask
        node_rows: list[np.ndarray] = []
        event_rows: list[np.ndarray] = []
        time_rows: list[np.ndarray] = []
        for node, cut_time in zip(
            nodes.detach().cpu().tolist(), cut_times.detach().cpu().tolist()
        ):
            if node == self.padding_node:
                end = 0
                history_times = np.empty(0, dtype=np.float64)
                history_nodes = np.empty(0, dtype=np.int64)
                history_events = np.empty(0, dtype=np.int64)
            else:
                history_times = self.timestamps[node]
                end = int(np.searchsorted(history_times, cut_time, side="left"))
                history_nodes = self.neighbors[node]
                history_events = self.event_ids[node]
            if end == 0:
                selected_nodes = np.empty(0, dtype=np.int64)
                selected_events = np.empty(0, dtype=np.int64)
                selected_times = np.empty(0, dtype=np.float64)
            elif uniform:
                selected = rng.integers(0, end, size=count)
                selected = selected[np.argsort(history_times[selected], kind="stable")]
                selected_nodes = history_nodes[selected]
                selected_events = history_events[selected]
                selected_times = history_times[selected]
            else:
                start = max(0, end - count)
                selected_nodes = history_nodes[start:end]
                selected_events = history_events[start:end]
                selected_times = history_times[start:end]
            padding = count - len(selected_nodes)
            node_rows.append(
                np.pad(
                    selected_nodes,
                    (padding, 0),
                    constant_values=self.padding_node,
                )
            )
            event_rows.append(
                np.pad(
                    selected_events,
                    (padding, 0),
                    constant_values=self.padding_event,
                )
            )
            time_rows.append(
                np.pad(selected_times, (padding, 0), constant_values=0.0)
            )
        neighbor_nodes = torch.as_tensor(
            np.stack(node_rows), dtype=torch.long, device=device
        )
        neighbor_events = torch.as_tensor(
            np.stack(event_rows), dtype=torch.long, device=device
        )
        neighbor_times = torch.as_tensor(
            np.stack(time_rows), dtype=cut_times.dtype, device=device
        )
        padding_mask = neighbor_nodes == self.padding_node
        return neighbor_nodes, neighbor_events, neighbor_times, padding_mask


class HarmonicTimeEncoder(nn.Module):
    """TGAT functional time encoding with learned frequencies and phases."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        frequency = 1.0 / 10 ** torch.linspace(0, 9, dimension)
        self.frequency = nn.Parameter(frequency)
        self.phase = nn.Parameter(torch.zeros(dimension))

    def forward(self, delta: Tensor) -> Tensor:
        return torch.cos(delta.unsqueeze(-1) * self.frequency + self.phase)


class MergeLayer(nn.Module):
    def __init__(self, dim1: int, dim2: int, hidden: int, output: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim1 + dim2, hidden)
        self.fc2 = nn.Linear(hidden, output)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return self.fc2(torch.relu(self.fc1(torch.cat([first, second], dim=-1))))


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        time_dim: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        query_dim = node_dim + time_dim
        key_dim = node_dim + edge_dim + time_dim
        if query_dim % heads:
            raise ValueError("node_dim + time_dim must be divisible by heads")
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            kdim=key_dim,
            vdim=key_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.merger = MergeLayer(query_dim, node_dim, node_dim, node_dim)

    def forward(
        self,
        source: Tensor,
        source_time: Tensor,
        neighbors: Tensor,
        neighbor_time: Tensor,
        edge_features: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        query = torch.cat([source.unsqueeze(1), source_time], dim=-1)
        key = torch.cat([neighbors, edge_features, neighbor_time], dim=-1)
        invalid = padding_mask.all(dim=1)
        safe_mask = padding_mask.clone()
        if invalid.any():
            safe_mask[invalid, 0] = False
        output, _ = self.attention(
            query, key, key, key_padding_mask=safe_mask, need_weights=False
        )
        output = output[:, 0]
        output = output.masked_fill(invalid.unsqueeze(-1), 0.0)
        return self.merger(output, source)


class SharedLinkProtocol:
    num_nodes: int
    num_users: int | None
    negative_ratio: float
    max_positive_pairs: int | None
    new_edges_only: bool
    negative_destination_candidates: Tensor | None
    allow_negative_collisions: bool
    eval_positive_batch_size: int | None

    def sample_queries(self, window: Sequence[Snapshot], seed: int) -> LinkQueries:
        return sample_link_queries(
            window[-1],
            window[-2],
            negative_ratio=self.negative_ratio,
            max_positive=self.max_positive_pairs,
            seed=seed,
            new_edges_only=self.new_edges_only,
            undirected=False,
            bipartite_source_count=self.num_users,
            negative_destination_candidates=self.negative_destination_candidates,
            allow_negative_collisions=self.allow_negative_collisions,
        )

    def metrics(
        self, labels: Tensor, scores: Tensor, groups: Tensor
    ) -> dict[str, float]:
        return link_prediction_metrics(
            labels,
            scores,
            groups,
            positive_batch_size=self.eval_positive_batch_size,
        )
