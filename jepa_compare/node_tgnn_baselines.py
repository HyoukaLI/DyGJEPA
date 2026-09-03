from __future__ import annotations

"""Dynamic-GNN baselines for the DBLP node-classification protocol.

The snapshot models reproduce the defining EvolveGCN-H and ROLAND update
equations without depending on the old PyTorch-Geometric releases required by
their upstream repositories.  TGN and TGAT follow their official two-stage
node-classification protocol: link-prediction pretraining followed by a frozen
node classifier.
"""

from dataclasses import dataclass
from collections import Counter
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import DynamicGraph, Snapshot
from .dyglib_official import (
    CAWN,
    DyGFormer,
    GraphMixer,
    MemoryModel,
    MergeLayer as DyGLibMergeLayer,
    NeighborSampler,
    TCL,
    compute_src_dst_node_time_shifts,
)
from .layers import GraphSAGELayer
from .temporal_event_utils import (
    EventStream,
    HarmonicTimeEncoder,
    MergeLayer,
    TemporalAttentionLayer,
    TemporalNeighborIndex,
)


def _normalized_gcn(x: Tensor, edge_index: Tensor) -> Tensor:
    """Apply symmetric GCN normalization with an explicit self-loop."""
    nodes = x.shape[0]
    loop = torch.arange(nodes, device=x.device)
    if edge_index.numel():
        source = torch.cat([edge_index[0], loop])
        destination = torch.cat([edge_index[1], loop])
    else:
        source = destination = loop
    degree = torch.bincount(destination, minlength=nodes).to(x.dtype).clamp_min(1)
    weight = degree[source].rsqrt() * degree[destination].rsqrt()
    output = torch.zeros_like(x)
    output.index_add_(0, destination, x[source] * weight.unsqueeze(1))
    return output


class MatrixGRUGate(nn.Module):
    """Matrix-valued GRU gate from the official EvolveGCN-H implementation."""

    def __init__(self, rows: int, columns: int, activation: nn.Module) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(rows, rows))
        self.recurrent = nn.Parameter(torch.empty(rows, rows))
        self.bias = nn.Parameter(torch.zeros(rows, columns))
        self.activation = activation
        bound = 1.0 / rows**0.5
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.recurrent, -bound, bound)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        return self.activation(self.weight @ x + self.recurrent @ hidden + self.bias)


class EvolveTopK(nn.Module):
    """Learned TopK node summary used as the input to EvolveGCN-H's GRU."""

    def __init__(self, features: int, count: int) -> None:
        super().__init__()
        self.scorer = nn.Parameter(torch.empty(features, 1))
        self.count = count
        nn.init.uniform_(self.scorer, -features**-0.5, features**-0.5)

    def forward(self, node_embeddings: Tensor, active: Tensor) -> Tensor:
        scores = (node_embeddings @ self.scorer).flatten() / self.scorer.norm().clamp_min(1e-12)
        scores = scores.masked_fill(~active, -torch.inf)
        available = min(self.count, int(active.sum().item()))
        if available == 0:
            return node_embeddings.new_zeros(node_embeddings.shape[1], self.count)
        values, indices = torch.topk(scores, available)
        if available < self.count:
            indices = torch.cat([indices, indices[-1:].expand(self.count - available)])
            values = torch.cat([values, values[-1:].expand(self.count - available)])
        selected = node_embeddings[indices] * torch.tanh(values).unsqueeze(1)
        return selected.t()


class MatrixGRUCell(nn.Module):
    def __init__(self, rows: int, columns: int) -> None:
        super().__init__()
        self.update = MatrixGRUGate(rows, columns, nn.Sigmoid())
        self.reset = MatrixGRUGate(rows, columns, nn.Sigmoid())
        self.candidate = MatrixGRUGate(rows, columns, nn.Tanh())
        self.topk = EvolveTopK(rows, columns)

    def forward(self, previous_weight: Tensor, node_embeddings: Tensor, active: Tensor) -> Tensor:
        summary = self.topk(node_embeddings, active)
        update = self.update(summary, previous_weight)
        reset = self.reset(summary, previous_weight)
        candidate = self.candidate(summary, reset * previous_weight)
        return (1.0 - update) * previous_weight + update * candidate


class EvolveGCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.initial_weight = nn.Parameter(torch.empty(input_dim, output_dim))
        nn.init.uniform_(
            self.initial_weight, -output_dim**-0.5, output_dim**-0.5
        )
        self.evolve = MatrixGRUCell(input_dim, output_dim)

    def forward(self, snapshots: Sequence[Snapshot], inputs: Sequence[Tensor]) -> list[Tensor]:
        weight = self.initial_weight
        outputs: list[Tensor] = []
        for snapshot, x in zip(snapshots, inputs):
            weight = self.evolve(weight, x, snapshot.active)
            outputs.append(F.rrelu(_normalized_gcn(x, snapshot.edge_index) @ weight))
        return outputs


class EvolveGCNHNodeClassifier(nn.Module):
    """EvolveGCN-H with weight evolution before every snapshot convolution."""

    def __init__(self, feature_dim: int, hidden_dim: int, classes: int, layers: int = 2) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("EvolveGCN-H requires at least one layer")
        dimensions = [feature_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            EvolveGCNLayer(a, b) for a, b in zip(dimensions, dimensions[1:])
        )
        self.classifier = nn.Linear(hidden_dim, classes)

    def node_embeddings(self, snapshots: Sequence[Snapshot]) -> Tensor:
        sequence = [snapshot.x for snapshot in snapshots]
        for layer in self.layers:
            sequence = layer(snapshots, sequence)
        return sequence[-1]

    def forward(self, snapshots: Sequence[Snapshot]) -> Tensor:
        return self.classifier(self.node_embeddings(snapshots))


class ROLANDLayer(nn.Module):
    """One hierarchical ROLAND state: static GraphSAGE then a GRU update."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.message_passing = GraphSAGELayer(input_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.PReLU(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, previous: Tensor, active: Tensor) -> Tensor:
        candidate = self.message_passing(x, edge_index)
        candidate = self.dropout(self.activation(self.normalization(candidate)))
        updated = self.update(candidate, previous)
        return torch.where(active.unsqueeze(1), updated, previous)


class ROLANDNodeClassifier(nn.Module):
    """ROLAND hierarchical node-state adaptation of a GraphSAGE backbone."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        classes: int,
        layers: int = 2,
        dropout: float = 0.0,
        bptt_steps: int = 4,
    ) -> None:
        super().__init__()
        if layers < 1 or bptt_steps < 1:
            raise ValueError("ROLAND layer and BPTT lengths must be positive")
        dimensions = [feature_dim] + [hidden_dim] * layers
        self.layers = nn.ModuleList(
            ROLANDLayer(a, b, dropout) for a, b in zip(dimensions, dimensions[1:])
        )
        self.classifier = nn.Linear(hidden_dim, classes)
        self.bptt_steps = bptt_steps

    def node_embeddings(self, snapshots: Sequence[Snapshot]) -> Tensor:
        states = [
            snapshots[0].x.new_zeros(snapshots[0].x.shape[0], layer.update.hidden_size)
            for layer in self.layers
        ]
        for time_index, snapshot in enumerate(snapshots):
            x = snapshot.x
            for layer_index, layer in enumerate(self.layers):
                states[layer_index] = layer(
                    x, snapshot.edge_index, states[layer_index], snapshot.active
                )
                x = states[layer_index]
            # ROLAND's scalable incremental training truncates BPTT while
            # carrying the numerical node states into the next segment.
            if (time_index + 1) % self.bptt_steps == 0 and time_index + 1 < len(snapshots):
                states = [state.detach() for state in states]
        return states[-1]

    def forward(self, snapshots: Sequence[Snapshot]) -> Tensor:
        return self.classifier(self.node_embeddings(snapshots))


@dataclass(frozen=True)
class HomogeneousEvents:
    sources: np.ndarray
    destinations: np.ndarray
    timestamps: np.ndarray
    edge_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.sources.shape[0])

    def take(self, rows: slice) -> "HomogeneousEvents":
        return HomogeneousEvents(
            self.sources[rows],
            self.destinations[rows],
            self.timestamps[rows],
            self.edge_ids[rows],
        )


def snapshot_addition_events(snapshots: Sequence[Snapshot]) -> HomogeneousEvents:
    """Convert cumulative undirected snapshots to new-edge interaction events.

    Multiplicity is retained: repeated co-authorship interactions are separate
    temporal events even when they share endpoints.  One orientation of the
    doubled message-passing COO is counted so an undirected edge is not emitted
    twice.
    """
    sources: list[np.ndarray] = []
    destinations: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    previous: Counter[tuple[int, int]] = Counter()
    for snapshot in sorted(snapshots, key=lambda item: item.time):
        edge = snapshot.edge_index.detach().cpu().numpy()
        current: Counter[tuple[int, int]] = Counter(
            (int(a), int(b))
            for a, b in zip(edge[0], edge[1])
            if a < b
        )
        # The preprocessing also duplicates a self-loop when it appends the
        # reverse orientation, hence two COO columns represent one event.
        self_loops = Counter(int(a) for a, b in zip(edge[0], edge[1]) if a == b)
        current.update({(node, node): count // 2 for node, count in self_loops.items()})
        additions = [
            pair
            for pair in sorted(current)
            for _ in range(max(0, current[pair] - previous[pair]))
        ]
        previous = current
        if not additions:
            continue
        pair = np.asarray(additions, dtype=np.int64)
        sources.append(pair[:, 0])
        destinations.append(pair[:, 1])
        timestamps.append(np.full(pair.shape[0], float(snapshot.time), dtype=np.float64))
    if not sources:
        raise ValueError("temporal node baselines require at least one edge event")
    source = np.concatenate(sources)
    destination = np.concatenate(destinations)
    time = np.concatenate(timestamps)
    return HomogeneousEvents(
        source,
        destination,
        time,
        np.arange(1, len(source) + 1, dtype=np.int64),
    )


def _homogeneous_sampler(
    events: HomogeneousEvents,
    num_nodes: int,
    seed: int,
    strategy: str = "recent",
    time_scaling_factor: float = 0.0,
) -> NeighborSampler:
    adjacency: list[list[tuple[int, int, float]]] = [
        [] for _ in range(num_nodes + 1)
    ]
    for source, destination, edge_id, timestamp in zip(
        events.sources, events.destinations, events.edge_ids, events.timestamps
    ):
        source_id, destination_id = int(source) + 1, int(destination) + 1
        entry = (destination_id, int(edge_id), float(timestamp))
        reverse = (source_id, int(edge_id), float(timestamp))
        adjacency[source_id].append(entry)
        adjacency[destination_id].append(reverse)
    return NeighborSampler(
        adjacency,
        sample_neighbor_strategy=strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed,
    )


class TGNNodeSSL(nn.Module):
    """Official DyGLib TGN backbone with homogeneous snapshot-event adapter."""

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        time_dim: int = 100,
        num_layers: int = 1,
        num_heads: int = 2,
        num_neighbors: int = 10,
        dropout: float = 0.1,
        batch_size: int = 100,
        inference_batch_size: int = 512,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes
        self.time_dim = time_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.seed = seed
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.backbone: MemoryModel | None = None
        self.link_predictor: DyGLibMergeLayer | None = None
        self.events: HomogeneousEvents | None = None
        self.sampler: NeighborSampler | None = None

    def prepare(self, graph: DynamicGraph) -> None:
        self.events = snapshot_addition_events(graph.snapshots)
        self.sampler = _homogeneous_sampler(self.events, self.num_nodes, self.seed)
        node_features = np.concatenate(
            [
                np.zeros((1, self.feature_dim), dtype=np.float32),
                graph.snapshots[-1].x.detach().cpu().numpy().astype(np.float32),
            ],
            axis=0,
        )
        edge_features = np.zeros(
            (len(self.events) + 1, self.feature_dim), dtype=np.float32
        )
        shifts = compute_src_dst_node_time_shifts(
            self.events.sources + 1,
            self.events.destinations + 1,
            self.events.timestamps,
        )
        self.backbone = MemoryModel(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=self.sampler,
            time_feat_dim=self.time_dim,
            model_name="TGN",
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            src_node_mean_time_shift=shifts[0],
            src_node_std_time_shift=max(float(shifts[1]), 1e-12),
            dst_node_mean_time_shift_dst=shifts[2],
            dst_node_std_time_shift=max(float(shifts[3]), 1e-12),
            device=str(self._device_anchor.device),
        ).to(self._device_anchor.device)
        self.link_predictor = DyGLibMergeLayer(
            self.feature_dim, self.feature_dim, self.feature_dim, 1
        ).to(self._device_anchor.device)

    def _require_prepared(self) -> tuple[MemoryModel, DyGLibMergeLayer, HomogeneousEvents]:
        if self.backbone is None or self.link_predictor is None or self.events is None:
            raise RuntimeError("call prepare before TGN training")
        return self.backbone, self.link_predictor, self.events

    def _embeddings(
        self,
        source: np.ndarray,
        destination: np.ndarray,
        timestamp: np.ndarray,
        edge_ids: np.ndarray | None,
        positive: bool,
    ) -> tuple[Tensor, Tensor]:
        backbone, _, _ = self._require_prepared()
        return backbone.compute_src_dst_node_temporal_embeddings(
            source,
            destination,
            timestamp,
            edge_ids=edge_ids,
            edges_are_positive=positive,
            num_neighbors=self.num_neighbors,
        )

    def train_epoch(self, optimizer: torch.optim.Optimizer, grad_clip: float, seed: int) -> dict[str, float]:
        backbone, predictor, events = self._require_prepared()
        backbone.memory_bank.__init_memory_bank__()
        generator = np.random.RandomState(seed)
        total_loss, batches = 0.0, 0
        for start in range(0, len(events), self.batch_size):
            batch = events.take(slice(start, start + self.batch_size))
            source = batch.sources + 1
            positive = batch.destinations + 1
            negative = generator.randint(1, self.num_nodes + 1, size=len(batch))
            collision = (negative == positive) | (negative == source)
            while collision.any():
                negative[collision] = generator.randint(1, self.num_nodes + 1, size=int(collision.sum()))
                collision = (negative == positive) | (negative == source)
            optimizer.zero_grad(set_to_none=True)
            negative_source, negative_destination = self._embeddings(
                source, negative, batch.timestamps, None, False
            )
            positive_source, positive_destination = self._embeddings(
                source, positive, batch.timestamps, batch.edge_ids, True
            )
            positive_logits = predictor(positive_source, positive_destination).squeeze(-1)
            negative_logits = predictor(negative_source, negative_destination).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([positive_logits, negative_logits]),
                torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in self.parameters() if parameter.requires_grad],
                grad_clip,
            )
            optimizer.step()
            backbone.memory_bank.detach_memory_bank()
            total_loss += float(loss.detach())
            batches += 1
        return {"loss": total_loss / max(1, batches), "events": float(len(events))}

    @torch.no_grad()
    def node_embeddings(self) -> Tensor:
        backbone, _, events = self._require_prepared()
        backbone.memory_bank.__init_memory_bank__()
        for start in range(0, len(events), self.batch_size):
            batch = events.take(slice(start, start + self.batch_size))
            self._embeddings(
                batch.sources + 1,
                batch.destinations + 1,
                batch.timestamps,
                batch.edge_ids,
                True,
            )
            backbone.memory_bank.detach_memory_bank()
        cutoff = float(events.timestamps.max()) + 1.0
        parts: list[Tensor] = []
        for start in range(0, self.num_nodes, self.inference_batch_size):
            nodes = np.arange(start, min(start + self.inference_batch_size, self.num_nodes)) + 1
            time = np.full(nodes.shape[0], cutoff, dtype=np.float64)
            source, _ = self._embeddings(nodes, nodes, time, None, False)
            parts.append(source)
        return torch.cat(parts)


class DyGLibStatelessNodeSSL(nn.Module):
    """Official stateless DyGLib backbone adapted to homogeneous snapshots.

    CAWN, TCL, GraphMixer and DyGFormer are kept unchanged.  This class only
    creates timestamp-tied events from cumulative DBLP snapshots, trains the
    native one-negative link objective, and exposes a final-time node readout
    for the common frozen classifier.
    """

    SUPPORTED = {"cawn", "tcl", "graphmixer", "dygformer"}

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        num_nodes: int,
        time_dim: int = 100,
        num_layers: int = 2,
        num_heads: int = 2,
        num_neighbors: int = 20,
        dropout: float = 0.1,
        channel_embedding_dim: int = 50,
        position_feat_dim: int | None = None,
        walk_length: int = 1,
        num_walk_heads: int = 8,
        patch_size: int = 1,
        max_input_sequence_length: int = 32,
        time_gap: int = 2000,
        batch_size: int = 200,
        inference_batch_size: int = 256,
        sample_neighbor_strategy: str = "recent",
        time_scaling_factor: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        normalized = model_name.lower()
        if normalized not in self.SUPPORTED:
            raise ValueError(f"unsupported stateless DyGLib node model: {model_name}")
        self.model_name = normalized
        self.feature_dim = int(feature_dim)
        self.num_nodes = int(num_nodes)
        self.time_dim = int(time_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.num_neighbors = int(num_neighbors)
        self.dropout = float(dropout)
        self.channel_embedding_dim = int(channel_embedding_dim)
        self.position_feat_dim = int(position_feat_dim or feature_dim)
        self.walk_length = int(walk_length)
        self.num_walk_heads = int(num_walk_heads)
        self.patch_size = int(patch_size)
        self.max_input_sequence_length = int(max_input_sequence_length)
        self.time_gap = int(time_gap)
        self.batch_size = int(batch_size)
        self.inference_batch_size = int(inference_batch_size)
        self.sample_neighbor_strategy = str(sample_neighbor_strategy)
        self.time_scaling_factor = float(time_scaling_factor)
        self.seed = int(seed)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.backbone: nn.Module | None = None
        self.link_predictor: DyGLibMergeLayer | None = None
        self.events: HomogeneousEvents | None = None
        self._negative_rng: np.random.RandomState | None = None

    def prepare(self, graph: DynamicGraph) -> None:
        self.events = snapshot_addition_events(graph.snapshots)
        sampler = _homogeneous_sampler(
            self.events,
            self.num_nodes,
            self.seed,
            self.sample_neighbor_strategy,
            self.time_scaling_factor,
        )
        node_features = np.concatenate(
            [
                np.zeros((1, self.feature_dim), dtype=np.float32),
                graph.snapshots[-1].x.detach().cpu().numpy().astype(np.float32),
            ],
            axis=0,
        )
        edge_features = np.zeros(
            (len(self.events) + 1, self.feature_dim), dtype=np.float32
        )
        common = dict(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=sampler,
            time_feat_dim=self.time_dim,
            dropout=self.dropout,
            device=str(self._device_anchor.device),
        )
        if self.model_name == "cawn":
            self.backbone = CAWN(
                **common,
                position_feat_dim=self.position_feat_dim,
                walk_length=self.walk_length,
                num_walk_heads=self.num_walk_heads,
            )
        elif self.model_name == "tcl":
            self.backbone = TCL(
                **common,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                num_depths=self.num_neighbors + 1,
            )
        elif self.model_name == "graphmixer":
            self.backbone = GraphMixer(
                **common,
                num_tokens=self.num_neighbors,
                num_layers=self.num_layers,
            )
        else:
            self.backbone = DyGFormer(
                **common,
                channel_embedding_dim=self.channel_embedding_dim,
                patch_size=self.patch_size,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                max_input_sequence_length=self.max_input_sequence_length,
            )
        self.backbone = self.backbone.to(self._device_anchor.device)
        self.link_predictor = DyGLibMergeLayer(
            self.feature_dim, self.feature_dim, self.feature_dim, 1
        ).to(self._device_anchor.device)

    def _require_prepared(
        self,
    ) -> tuple[nn.Module, DyGLibMergeLayer, HomogeneousEvents]:
        if self.backbone is None or self.link_predictor is None or self.events is None:
            raise RuntimeError("call prepare before DyGLib node pretraining")
        return self.backbone, self.link_predictor, self.events

    def _embeddings(
        self,
        source: np.ndarray,
        destination: np.ndarray,
        timestamp: np.ndarray,
    ) -> tuple[Tensor, Tensor]:
        backbone, _, _ = self._require_prepared()
        common = dict(
            src_node_ids=source,
            dst_node_ids=destination,
            node_interact_times=timestamp,
        )
        if self.model_name in {"cawn", "tcl"}:
            return backbone.compute_src_dst_node_temporal_embeddings(  # type: ignore[attr-defined]
                **common, num_neighbors=self.num_neighbors
            )
        if self.model_name == "graphmixer":
            return backbone.compute_src_dst_node_temporal_embeddings(  # type: ignore[attr-defined]
                **common, num_neighbors=self.num_neighbors, time_gap=self.time_gap
            )
        return backbone.compute_src_dst_node_temporal_embeddings(**common)  # type: ignore[attr-defined]

    def train_epoch(
        self, optimizer: torch.optim.Optimizer, grad_clip: float, seed: int
    ) -> dict[str, float]:
        del grad_clip
        _, predictor, events = self._require_prepared()
        if self._negative_rng is None:
            self._negative_rng = np.random.RandomState(seed)
        generator = self._negative_rng
        total_loss, batches = 0.0, 0
        for start in range(0, len(events), self.batch_size):
            batch = events.take(slice(start, start + self.batch_size))
            source = batch.sources + 1
            positive = batch.destinations + 1
            timestamp = batch.timestamps
            negative = generator.randint(1, self.num_nodes + 1, size=len(batch))
            collision = (negative == positive) | (negative == source)
            while collision.any():
                negative[collision] = generator.randint(
                    1, self.num_nodes + 1, size=int(collision.sum())
                )
                collision = (negative == positive) | (negative == source)
            optimizer.zero_grad(set_to_none=True)
            positive_source, positive_destination = self._embeddings(
                source, positive, timestamp
            )
            negative_source, negative_destination = self._embeddings(
                source, negative, timestamp
            )
            positive_logits = predictor(
                positive_source, positive_destination
            ).squeeze(-1)
            negative_logits = predictor(
                negative_source, negative_destination
            ).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([positive_logits, negative_logits]),
                torch.cat(
                    [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]
                ),
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        return {"loss": total_loss / max(1, batches), "events": float(len(events))}

    @torch.no_grad()
    def node_embeddings(self, seed: int = 42) -> Tensor:
        del seed
        _, _, events = self._require_prepared()
        cutoff = float(events.timestamps.max()) + 1.0
        parts: list[Tensor] = []
        for start in range(0, self.num_nodes, self.inference_batch_size):
            nodes = np.arange(
                start, min(start + self.inference_batch_size, self.num_nodes)
            ) + 1
            times = np.full(nodes.shape[0], cutoff, dtype=np.float64)
            # DBLP provides one label per node rather than event-conditioned
            # labels.  A self-pair gives every node the same deterministic
            # final-time query context while preserving the official encoder.
            source, _ = self._embeddings(nodes, nodes, times)
            parts.append(source)
        return torch.cat(parts)


class TGATNodeSSL(nn.Module):
    """TGAT temporal convolution with official link-pretrain/frozen-probe use."""

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        hidden_dim: int = 100,
        num_layers: int = 2,
        num_heads: int = 2,
        num_neighbors: int = 20,
        dropout: float = 0.1,
        uniform_neighbors: bool = False,
        batch_size: int = 200,
        inference_batch_size: int = 256,
    ) -> None:
        super().__init__()
        if (2 * hidden_dim) % num_heads:
            raise ValueError("twice hidden_dim must be divisible by num_heads")
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.uniform_neighbors = uniform_neighbors
        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.raw_projection = nn.Linear(feature_dim, hidden_dim)
        self.register_buffer("raw_node_features", torch.empty(0), persistent=False)
        self.register_buffer("edge_features", torch.empty(0), persistent=False)
        self.time_encoder = HarmonicTimeEncoder(hidden_dim)
        self.attention_layers = nn.ModuleList(
            TemporalAttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.affinity = MergeLayer(hidden_dim, hidden_dim, hidden_dim, 1)
        self.events: HomogeneousEvents | None = None
        self.stream: EventStream | None = None
        self.index: TemporalNeighborIndex | None = None

    def prepare(self, graph: DynamicGraph) -> None:
        events = snapshot_addition_events(graph.snapshots)
        device = graph.snapshots[-1].x.device
        source = torch.as_tensor(events.sources, dtype=torch.long, device=device)
        destination = torch.as_tensor(events.destinations, dtype=torch.long, device=device)
        timestamp = torch.as_tensor(events.timestamps, dtype=torch.float32, device=device)
        features = torch.zeros(len(events), self.hidden_dim, device=device)
        self.events = events
        self.stream = EventStream(source, destination, timestamp, features)
        self.index = TemporalNeighborIndex(self.stream, self.num_nodes)
        padding = torch.zeros(1, graph.feature_dim, device=device)
        self.raw_node_features = torch.cat([graph.snapshots[-1].x, padding])
        self.edge_features = torch.zeros(len(events) + 1, self.hidden_dim, device=device)

    def _temporal_embedding(
        self,
        nodes: Tensor,
        times: Tensor,
        layers: int,
        rng: np.random.Generator,
    ) -> Tensor:
        if self.index is None:
            raise RuntimeError("call prepare before TGAT training")
        source = self.raw_projection(self.raw_node_features[nodes])
        if layers == 0:
            return source
        neighbor_nodes, neighbor_events, neighbor_times, mask = self.index.sample(
            nodes,
            times,
            self.num_neighbors,
            uniform=self.uniform_neighbors,
            rng=rng,
            device=nodes.device,
        )
        source_previous = self._temporal_embedding(nodes, times, layers - 1, rng)
        neighbor_previous = self._temporal_embedding(
            neighbor_nodes.reshape(-1), neighbor_times.reshape(-1), layers - 1, rng
        ).reshape(nodes.shape[0], self.num_neighbors, self.hidden_dim)
        delta = (times.unsqueeze(1) - neighbor_times).clamp_min(0)
        return self.attention_layers[layers - 1](
            source_previous,
            self.time_encoder(times.new_zeros(nodes.shape[0], 1)),
            neighbor_previous,
            self.time_encoder(delta),
            self.edge_features[neighbor_events],
            mask,
        )

    def _score(self, source: Tensor, destination: Tensor, time: Tensor, rng: np.random.Generator) -> Tensor:
        source_embedding = self._temporal_embedding(source, time, self.num_layers, rng)
        destination_embedding = self._temporal_embedding(destination, time, self.num_layers, rng)
        return self.affinity(source_embedding, destination_embedding).squeeze(-1)

    def train_epoch(self, optimizer: torch.optim.Optimizer, grad_clip: float, seed: int) -> dict[str, float]:
        if self.stream is None:
            raise RuntimeError("call prepare before TGAT training")
        stream = self.stream
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(stream), generator=generator).to(stream.sources.device)
        rng = np.random.default_rng(seed)
        total_loss, batches = 0.0, 0
        for start in range(0, len(stream), self.batch_size):
            rows = order[start : start + self.batch_size]
            source = stream.sources[rows]
            positive = stream.destinations[rows]
            time = stream.timestamps[rows]
            negative = torch.randint(
                self.num_nodes,
                positive.shape,
                generator=generator,
                device="cpu",
            ).to(positive.device)
            collision = (negative == positive) | (negative == source)
            while collision.any():
                replacement = torch.randint(
                    self.num_nodes,
                    (int(collision.sum()),),
                    generator=generator,
                    device="cpu",
                ).to(positive.device)
                negative[collision] = replacement
                collision = (negative == positive) | (negative == source)
            optimizer.zero_grad(set_to_none=True)
            positive_logits = self._score(source, positive, time, rng)
            negative_logits = self._score(source, negative, time, rng)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([positive_logits, negative_logits]),
                torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        return {"loss": total_loss / max(1, batches), "events": float(len(stream))}

    @torch.no_grad()
    def node_embeddings(self, seed: int = 42) -> Tensor:
        if self.stream is None:
            raise RuntimeError("call prepare before TGAT inference")
        cutoff = float(self.stream.timestamps.max().item()) + 1.0
        rng = np.random.default_rng(seed)
        parts: list[Tensor] = []
        device = self.stream.sources.device
        for start in range(0, self.num_nodes, self.inference_batch_size):
            nodes = torch.arange(
                start, min(start + self.inference_batch_size, self.num_nodes), device=device
            )
            time = torch.full((nodes.numel(),), cutoff, device=device)
            parts.append(self._temporal_embedding(nodes, time, self.num_layers, rng))
        return torch.cat(parts)
