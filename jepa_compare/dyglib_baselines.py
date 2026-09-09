from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import Snapshot
from .dyglib_official import (
    CAWN,
    DyGFormer,
    GraphMixer,
    MemoryModel,
    MergeLayer,
    NeighborSampler,
    TCL,
    compute_src_dst_node_time_shifts,
)
from .temporal_event_utils import SharedLinkProtocol, snapshot_events, unique_snapshots


@dataclass(frozen=True)
class NumpyEventStream:
    sources: np.ndarray
    destinations: np.ndarray
    timestamps: np.ndarray
    edge_ids: np.ndarray
    features: np.ndarray

    def __len__(self) -> int:
        return int(self.sources.shape[0])

    def take(self, rows: slice | np.ndarray) -> "NumpyEventStream":
        return NumpyEventStream(
            self.sources[rows],
            self.destinations[rows],
            self.timestamps[rows],
            self.edge_ids[rows],
            self.features[rows],
        )


def _snapshot_stream(
    snapshot: Snapshot,
    *,
    num_users: int | None,
    num_nodes: int,
    feature_dim: int,
    first_edge_id: int,
) -> NumpyEventStream:
    raw_feature_dim = (
        int(snapshot.query_features.shape[1])
        if snapshot.query_features is not None
        else feature_dim
    )
    events = snapshot_events(
        snapshot,
        num_users=num_users,
        num_nodes=num_nodes,
        feature_dim=raw_feature_dim,
    )
    count = len(events)
    features = events.features
    if raw_feature_dim < feature_dim:
        # DyGLib pads featureless/low-dimensional datasets to its shared
        # 172-dimensional non-time representation width.
        features = F.pad(features, (0, feature_dim - raw_feature_dim))
    elif raw_feature_dim > feature_dim:
        raise ValueError(
            f"interaction feature width {raw_feature_dim} exceeds configured "
            f"DyGLib width {feature_dim}"
        )
    # DyGLib reserves node/edge id zero for padding.
    return NumpyEventStream(
        events.sources.detach().cpu().numpy().astype(np.int64) + 1,
        events.destinations.detach().cpu().numpy().astype(np.int64) + 1,
        events.timestamps.detach().cpu().numpy().astype(np.float64),
        np.arange(first_edge_id, first_edge_id + count, dtype=np.int64),
        features.detach().cpu().numpy().astype(np.float32),
    )


def _concatenate(streams: Sequence[NumpyEventStream], feature_dim: int) -> NumpyEventStream:
    if not streams:
        return NumpyEventStream(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty((0, feature_dim), dtype=np.float32),
        )
    return NumpyEventStream(
        np.concatenate([stream.sources for stream in streams]),
        np.concatenate([stream.destinations for stream in streams]),
        np.concatenate([stream.timestamps for stream in streams]),
        np.concatenate([stream.edge_ids for stream in streams]),
        np.concatenate([stream.features for stream in streams]),
    )


def _neighbor_sampler(
    stream: NumpyEventStream,
    num_nodes: int,
    strategy: str,
    seed: int,
    time_scaling_factor: float,
) -> NeighborSampler:
    adjacency: list[list[tuple[int, int, float]]] = [
        [] for _ in range(num_nodes + 1)
    ]
    for source, destination, edge_id, timestamp in zip(
        stream.sources, stream.destinations, stream.edge_ids, stream.timestamps
    ):
        adjacency[int(source)].append((int(destination), int(edge_id), float(timestamp)))
        adjacency[int(destination)].append((int(source), int(edge_id), float(timestamp)))
    return NeighborSampler(
        adjacency,
        sample_neighbor_strategy=strategy,
        time_scaling_factor=time_scaling_factor,
        seed=seed,
    )


class DyGLibLinkBaseline(nn.Module, SharedLinkProtocol):
    """Unmodified DyGLib backbone behind this repository's evaluation protocol.

    Model equations and modules are vendored from the author-maintained DyGLib
    repository (MIT).  This adapter only converts snapshot ids to DyGLib's
    padded event representation and supplies the common split/query/metrics.
    """

    SUPPORTED = {"dyrep", "tgn", "cawn", "tcl", "graphmixer", "dygformer"}

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        num_nodes: int,
        bipartite_source_count: int | None = None,
        interaction_feature_dim: int = 172,
        time_feat_dim: int = 100,
        position_feat_dim: int = 172,
        channel_embedding_dim: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        num_neighbors: int = 20,
        dropout: float = 0.1,
        walk_length: int = 1,
        num_walk_heads: int = 8,
        time_gap: int = 2000,
        patch_size: int = 1,
        max_input_sequence_length: int = 32,
        sample_neighbor_strategy: str = "recent",
        time_scaling_factor: float = 0.0,
        train_batch_size: int = 200,
        eval_pair_batch_size: int = 200,
        negative_ratio: float = 20.0,
        max_positive_pairs: int | None = 1024,
        new_edges_only: bool = False,
        undirected: bool = False,
        negative_destination_candidates: Tensor | None = None,
        allow_negative_collisions: bool = False,
        eval_positive_batch_size: int | None = None,
        sampler_seed: int = 1,
    ) -> None:
        super().__init__()
        del feature_dim
        normalized = model_name.lower()
        if normalized not in self.SUPPORTED:
            raise ValueError(f"unsupported DyGLib model: {model_name}")
        if undirected:
            raise ValueError("event-stream comparison requires directed links")
        if interaction_feature_dim < 1 or train_batch_size < 1:
            raise ValueError("feature and batch dimensions must be positive")
        self.model_name = normalized
        self.num_nodes = num_nodes
        self.num_users = bipartite_source_count
        self.dimension = interaction_feature_dim
        self.time_feat_dim = time_feat_dim
        self.position_feat_dim = position_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.walk_length = walk_length
        self.num_walk_heads = num_walk_heads
        self.time_gap = time_gap
        self.patch_size = patch_size
        self.max_input_sequence_length = max_input_sequence_length
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.time_scaling_factor = time_scaling_factor
        self.train_batch_size = train_batch_size
        self.eval_pair_batch_size = eval_pair_batch_size
        self.negative_ratio = negative_ratio
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = new_edges_only
        self.negative_destination_candidates = negative_destination_candidates
        self.allow_negative_collisions = allow_negative_collisions
        self.eval_positive_batch_size = eval_positive_batch_size
        self.sampler_seed = sampler_seed

        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self.backbone: nn.Module | None = None
        self.link_predictor: MergeLayer | None = None
        self._train_stream: NumpyEventStream | None = None
        self._snapshot_streams: dict[int, NumpyEventStream] = {}
        self._train_sampler: NeighborSampler | None = None
        self._full_sampler: NeighborSampler | None = None
        self._train_destinations: np.ndarray | None = None
        self._full_destinations: np.ndarray | None = None
        self._negative_rng: np.random.RandomState | None = None

    @property
    def is_memory_model(self) -> bool:
        return self.model_name in {"dyrep", "tgn"}

    def prepare_streams(
        self,
        all_snapshots: Sequence[Snapshot],
        train_snapshots: Sequence[Snapshot],
    ) -> None:
        streams: list[NumpyEventStream] = []
        next_edge_id = 1
        for snapshot in sorted(all_snapshots, key=lambda item: item.time):
            stream = _snapshot_stream(
                snapshot,
                num_users=self.num_users,
                num_nodes=self.num_nodes,
                feature_dim=self.dimension,
                first_edge_id=next_edge_id,
            )
            self._snapshot_streams[snapshot.time] = stream
            streams.append(stream)
            next_edge_id += len(stream)
        full_stream = _concatenate(streams, self.dimension)
        train_ids = {snapshot.time for snapshot in train_snapshots}
        train_streams = [
            self._snapshot_streams[snapshot.time]
            for snapshot in sorted(all_snapshots, key=lambda item: item.time)
            if snapshot.time in train_ids
        ]
        self._train_stream = _concatenate(train_streams, self.dimension)
        self._train_destinations = np.unique(self._train_stream.destinations)
        self._full_destinations = np.unique(full_stream.destinations)

        self._train_sampler = _neighbor_sampler(
            self._train_stream,
            self.num_nodes,
            self.sample_neighbor_strategy,
            self.sampler_seed,
            self.time_scaling_factor,
        )
        self._full_sampler = _neighbor_sampler(
            full_stream,
            self.num_nodes,
            self.sample_neighbor_strategy,
            self.sampler_seed,
            self.time_scaling_factor,
        )
        node_features = np.zeros(
            (self.num_nodes + 1, self.dimension), dtype=np.float32
        )
        edge_features = np.concatenate(
            [np.zeros((1, self.dimension), dtype=np.float32), full_stream.features],
            axis=0,
        )
        device = str(self._device_anchor.device)
        common = dict(
            node_raw_features=node_features,
            edge_raw_features=edge_features,
            neighbor_sampler=self._train_sampler,
            time_feat_dim=self.time_feat_dim,
            dropout=self.dropout,
            device=device,
        )
        if self.model_name in {"dyrep", "tgn"}:
            shifts = compute_src_dst_node_time_shifts(
                self._train_stream.sources,
                self._train_stream.destinations,
                self._train_stream.timestamps,
            )
            self.backbone = MemoryModel(
                **common,
                model_name="DyRep" if self.model_name == "dyrep" else "TGN",
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                src_node_mean_time_shift=shifts[0],
                src_node_std_time_shift=shifts[1],
                dst_node_mean_time_shift_dst=shifts[2],
                dst_node_std_time_shift=shifts[3],
            )
        elif self.model_name == "cawn":
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
        # The adapter is moved to the requested device before prepare_streams
        # constructs the upstream backbone. Newly attached child modules do not
        # inherit an existing module's device automatically, so move all
        # parameters created by the upstream constructor explicitly. Raw node
        # and edge feature tensors already use ``device`` from ``common``.
        self.backbone = self.backbone.to(self._device_anchor.device)
        self.link_predictor = MergeLayer(
            input_dim1=self.dimension,
            input_dim2=self.dimension,
            hidden_dim=self.dimension,
            output_dim=1,
        ).to(self._device_anchor.device)

    def _require_prepared(self) -> tuple[nn.Module, MergeLayer]:
        if self.backbone is None or self.link_predictor is None:
            raise RuntimeError("call prepare_streams before training or evaluation")
        return self.backbone, self.link_predictor

    def _set_sampler(self, sampler: NeighborSampler) -> None:
        backbone, _ = self._require_prepared()
        backbone.set_neighbor_sampler(sampler)  # type: ignore[attr-defined]

    def _embeddings(
        self,
        sources: np.ndarray,
        destinations: np.ndarray,
        timestamps: np.ndarray,
        *,
        edge_ids: np.ndarray | None = None,
        positive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        backbone, _ = self._require_prepared()
        if self.is_memory_model:
            return backbone.compute_src_dst_node_temporal_embeddings(  # type: ignore[attr-defined]
                src_node_ids=sources,
                dst_node_ids=destinations,
                node_interact_times=timestamps,
                edge_ids=edge_ids,
                edges_are_positive=positive,
                num_neighbors=self.num_neighbors,
            )
        if self.model_name in {"cawn", "tcl"}:
            return backbone.compute_src_dst_node_temporal_embeddings(  # type: ignore[attr-defined]
                src_node_ids=sources,
                dst_node_ids=destinations,
                node_interact_times=timestamps,
                num_neighbors=self.num_neighbors,
            )
        if self.model_name == "graphmixer":
            return backbone.compute_src_dst_node_temporal_embeddings(  # type: ignore[attr-defined]
                src_node_ids=sources,
                dst_node_ids=destinations,
                node_interact_times=timestamps,
                num_neighbors=self.num_neighbors,
                time_gap=self.time_gap,
            )
        return backbone.compute_src_dst_node_temporal_embeddings(  # type: ignore[attr-defined]
            src_node_ids=sources,
            dst_node_ids=destinations,
            node_interact_times=timestamps,
        )

    def _logits(
        self, sources: np.ndarray, destinations: np.ndarray, timestamps: np.ndarray
    ) -> Tensor:
        _, predictor = self._require_prepared()
        source_embeddings, destination_embeddings = self._embeddings(
            sources, destinations, timestamps, positive=False
        )
        return predictor(
            input_1=source_embeddings, input_2=destination_embeddings
        ).squeeze(-1)

    def _advance_memory(self, stream: NumpyEventStream) -> None:
        if not self.is_memory_model or len(stream) == 0:
            return
        for start in range(0, len(stream), self.train_batch_size):
            batch = stream.take(slice(start, start + self.train_batch_size))
            self._embeddings(
                batch.sources,
                batch.destinations,
                batch.timestamps,
                edge_ids=batch.edge_ids,
                positive=True,
            )
            backbone, _ = self._require_prepared()
            backbone.memory_bank.detach_memory_bank()  # type: ignore[attr-defined]

    def train_epoch(
        self,
        windows: Sequence[Sequence[Snapshot]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int = 42,
    ) -> dict[str, float]:
        del windows, grad_clip
        if (
            self._train_stream is None
            or self._train_sampler is None
            or self._train_destinations is None
        ):
            raise RuntimeError("call prepare_streams before training")
        self._set_sampler(self._train_sampler)
        backbone, predictor = self._require_prepared()
        if self.is_memory_model:
            backbone.memory_bank.__init_memory_bank__()  # type: ignore[attr-defined]
        # The upstream unseeded training sampler is initialized once per run
        # and its state advances continuously across epochs.
        if self._negative_rng is None:
            self._negative_rng = np.random.RandomState(seed)
        rng = self._negative_rng
        total_loss = 0.0
        batches = 0
        stream = self._train_stream
        for start in range(0, len(stream), self.train_batch_size):
            batch = stream.take(slice(start, start + self.train_batch_size))
            # DyGLib's training NegativeEdgeSampler draws uniformly from the
            # destinations observed in the training prefix.
            negatives = rng.choice(
                self._train_destinations, size=len(batch), replace=True
            ).astype(np.int64)
            if not self.allow_negative_collisions:
                collision = negatives == batch.destinations
                while collision.any() and self._train_destinations.size > 1:
                    negatives[collision] = rng.choice(
                        self._train_destinations,
                        size=int(collision.sum()),
                        replace=True,
                    )
                    collision = negatives == batch.destinations

            optimizer.zero_grad(set_to_none=True)
            if self.is_memory_model:
                negative_source, negative_destination = self._embeddings(
                    batch.sources, negatives, batch.timestamps, positive=False
                )
                positive_source, positive_destination = self._embeddings(
                    batch.sources,
                    batch.destinations,
                    batch.timestamps,
                    edge_ids=batch.edge_ids,
                    positive=True,
                )
            else:
                positive_source, positive_destination = self._embeddings(
                    batch.sources, batch.destinations, batch.timestamps
                )
                negative_source, negative_destination = self._embeddings(
                    batch.sources, negatives, batch.timestamps
                )
            positive_logits = predictor(
                input_1=positive_source, input_2=positive_destination
            ).squeeze(-1)
            negative_logits = predictor(
                input_1=negative_source, input_2=negative_destination
            ).squeeze(-1)
            logits = torch.cat([positive_logits, negative_logits])
            labels = torch.cat(
                [torch.ones_like(positive_logits), torch.zeros_like(negative_logits)]
            )
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            if self.is_memory_model:
                backbone.memory_bank.detach_memory_bank()  # type: ignore[attr-defined]
            total_loss += float(loss.detach().item())
            batches += 1
        return {"loss": total_loss / max(1, batches)}

    @torch.no_grad()
    def _evaluate_dyglib_random(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int,
    ) -> dict[str, float]:
        """Mirror DyGLib's batched evaluator.

        Negatives are DyGLib random destinations unless a historical/inductive
        ``negative_edge_table`` is attached, in which case both endpoints of
        every negative come from the table (as in ``evaluate_link_prediction.py``).
        """
        if self._full_sampler is None or self._full_destinations is None:
            raise RuntimeError("call prepare_streams before evaluation")
        self._set_sampler(self._full_sampler)
        backbone, predictor = self._require_prepared()
        if self.is_memory_model:
            backbone.memory_bank.__init_memory_bank__()  # type: ignore[attr-defined]
            history = [
                self._snapshot_streams[item.time]
                for item in unique_snapshots(history_windows)
            ]
            self._advance_memory(_concatenate(history, self.dimension))

        target_snapshots = unique_snapshots(windows, targets_only=True)
        targets = [self._snapshot_streams[item.time] for item in target_snapshots]
        stream = _concatenate(targets, self.dimension)
        table_negatives = None
        if self.negative_edge_table is not None:
            table_negatives = self.negative_edge_table.for_snapshots(
                [item.time for item in target_snapshots]
            )
            # DyGLib reserves id zero, so the adapter's stream is shifted by one.
            table_negatives.check_alignment(
                stream.sources - 1,
                stream.destinations - 1,
                context=f"{self.model_name} evaluation",
            )
        rng = np.random.RandomState(query_seed)
        score_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        group_parts: list[Tensor] = []
        group_offset = 0
        for start in range(0, len(stream), self.eval_pair_batch_size):
            batch = stream.take(slice(start, start + self.eval_pair_batch_size))
            if table_negatives is None:
                negative_sources = batch.sources
                negatives = rng.choice(
                    self._full_destinations, size=len(batch), replace=True
                ).astype(np.int64)
                if not self.allow_negative_collisions:
                    collision = negatives == batch.destinations
                    while collision.any() and self._full_destinations.size > 1:
                        negatives[collision] = rng.choice(
                            self._full_destinations,
                            size=int(collision.sum()),
                            replace=True,
                        )
                        collision = negatives == batch.destinations
            else:
                rows = slice(start, start + len(batch))
                negative_sources = table_negatives.sources[rows] + 1
                negatives = table_negatives.destinations[rows] + 1
            if self.is_memory_model:
                negative_source, negative_destination = self._embeddings(
                    negative_sources, negatives, batch.timestamps, positive=False
                )
                positive_source, positive_destination = self._embeddings(
                    batch.sources,
                    batch.destinations,
                    batch.timestamps,
                    edge_ids=batch.edge_ids,
                    positive=True,
                )
            else:
                positive_source, positive_destination = self._embeddings(
                    batch.sources, batch.destinations, batch.timestamps
                )
                negative_source, negative_destination = self._embeddings(
                    negative_sources, negatives, batch.timestamps
                )
            positive_scores = torch.sigmoid(
                predictor(
                    input_1=positive_source, input_2=positive_destination
                ).squeeze(-1)
            )
            negative_scores = torch.sigmoid(
                predictor(
                    input_1=negative_source, input_2=negative_destination
                ).squeeze(-1)
            )
            score_parts.extend([positive_scores, negative_scores])
            label_parts.extend(
                [torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]
            )
            group = torch.arange(
                group_offset,
                group_offset + len(batch),
                dtype=torch.long,
                device=positive_scores.device,
            )
            group_parts.extend([group, group])
            group_offset += len(batch)
            if self.is_memory_model:
                backbone.memory_bank.detach_memory_bank()  # type: ignore[attr-defined]
        if not score_parts:
            raise ValueError(f"{self.model_name} evaluation produced no events")
        return self.metrics(
            torch.cat(label_parts), torch.cat(score_parts), torch.cat(group_parts)
        )

    @torch.no_grad()
    def _evaluate_stateless(
        self, windows: Sequence[Sequence[Snapshot]], query_seed: int
    ) -> dict[str, float]:
        if self._full_sampler is None:
            raise RuntimeError("call prepare_streams before evaluation")
        self._set_sampler(self._full_sampler)
        score_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        group_parts: list[Tensor] = []
        group_offset = 0
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(window, query_seed + window_index)
            for start in range(0, queries.labels.numel(), self.eval_pair_batch_size):
                rows = slice(start, start + self.eval_pair_batch_size)
                pairs = queries.pairs[rows].detach().cpu().numpy().astype(np.int64) + 1
                if queries.timestamps is None:
                    times = np.full(len(pairs), float(window[-1].time), dtype=np.float64)
                else:
                    times = queries.timestamps[rows].detach().cpu().numpy().astype(np.float64)
                score_parts.append(torch.sigmoid(self._logits(pairs[:, 0], pairs[:, 1], times)))
                label_parts.append(queries.labels[rows])
                group_parts.append(queries.group_ids[rows] + group_offset)
            group_offset += int(queries.group_ids.max().item()) + 1
        if not score_parts:
            raise ValueError(f"{self.model_name} evaluation produced no queries")
        return self.metrics(
            torch.cat(label_parts), torch.cat(score_parts), torch.cat(group_parts)
        )

    @torch.no_grad()
    def _evaluate_memory_model(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int,
    ) -> dict[str, float]:
        if self._full_sampler is None:
            raise RuntimeError("call prepare_streams before evaluation")
        self._set_sampler(self._full_sampler)
        backbone, _ = self._require_prepared()
        backbone.memory_bank.__init_memory_bank__()  # type: ignore[attr-defined]
        history = [self._snapshot_streams[item.time] for item in unique_snapshots(history_windows)]
        self._advance_memory(_concatenate(history, self.dimension))

        score_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        group_parts: list[Tensor] = []
        group_offset = 0
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(window, query_seed + window_index)
            target = self._snapshot_streams[window[-1].time]
            group_ids = torch.unique(queries.group_ids, sorted=True)
            groups_by_time: dict[float, list[int]] = {}
            for group in group_ids.detach().cpu().tolist():
                row = int(torch.nonzero(queries.group_ids == group, as_tuple=False)[0])
                timestamp = (
                    float(queries.timestamps[row].item())
                    if queries.timestamps is not None
                    else float(window[-1].time)
                )
                groups_by_time.setdefault(timestamp, []).append(group)
            event_cursor = 0
            for timestamp in sorted(groups_by_time):
                before = int(np.searchsorted(target.timestamps, timestamp, side="left"))
                self._advance_memory(target.take(slice(event_cursor, before)))
                event_cursor = before
                selected_groups = groups_by_time[timestamp]
                row_parts = [
                    torch.nonzero(queries.group_ids == group, as_tuple=False).flatten()
                    for group in selected_groups
                ]
                rows = torch.cat(row_parts)
                for start in range(0, rows.numel(), self.eval_pair_batch_size):
                    chunk = rows[start : start + self.eval_pair_batch_size]
                    pairs = queries.pairs[chunk].detach().cpu().numpy().astype(np.int64) + 1
                    times = np.full(len(pairs), timestamp, dtype=np.float64)
                    score_parts.append(
                        torch.sigmoid(self._logits(pairs[:, 0], pairs[:, 1], times))
                    )
                    label_parts.append(queries.labels[chunk])
                    remapped = torch.empty_like(chunk)
                    for local, part in enumerate(row_parts):
                        mask = torch.isin(chunk, part)
                        remapped[mask] = group_offset + local
                    group_parts.append(remapped)
                group_offset += len(row_parts)
                through = int(np.searchsorted(target.timestamps, timestamp, side="right"))
                self._advance_memory(target.take(slice(event_cursor, through)))
                event_cursor = through
            self._advance_memory(target.take(slice(event_cursor, len(target))))
        if not score_parts:
            raise ValueError(f"{self.model_name} evaluation produced no queries")
        return self.metrics(
            torch.cat(label_parts), torch.cat(score_parts), torch.cat(group_parts)
        )

    @torch.no_grad()
    def evaluate_protocol(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int = 42,
    ) -> dict[str, float]:
        if (
            self.negative_ratio == 1.0
            and self.max_positive_pairs is None
            and self.eval_positive_batch_size is not None
        ):
            return self._evaluate_dyglib_random(
                windows, history_windows, query_seed
            )
        if self.is_memory_model:
            return self._evaluate_memory_model(windows, history_windows, query_seed)
        return self._evaluate_stateless(windows, query_seed)


class EdgeBankLinkBaseline(nn.Module, SharedLinkProtocol):
    """EdgeBank baseline under the shared event protocol."""

    def __init__(
        self,
        num_nodes: int,
        bipartite_source_count: int | None = None,
        negative_ratio: float = 20.0,
        max_positive_pairs: int | None = 1024,
        new_edges_only: bool = False,
        undirected: bool = False,
        negative_destination_candidates: Tensor | None = None,
        allow_negative_collisions: bool = False,
        eval_positive_batch_size: int | None = None,
        memory_mode: str = "time_window_memory",
        time_window_proportion: float = 0.15,
    ) -> None:
        super().__init__()
        if undirected:
            raise ValueError("event-stream comparison requires directed links")
        self.num_nodes = num_nodes
        self.num_users = bipartite_source_count
        self.negative_ratio = negative_ratio
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = new_edges_only
        self.negative_destination_candidates = negative_destination_candidates
        self.allow_negative_collisions = allow_negative_collisions
        self.eval_positive_batch_size = eval_positive_batch_size
        if memory_mode not in {"unlimited_memory", "time_window_memory"}:
            raise ValueError("unsupported EdgeBank memory mode")
        if not 0.0 < time_window_proportion <= 1.0:
            raise ValueError("time_window_proportion must be in (0, 1]")
        self.memory_mode = memory_mode
        self.time_window_proportion = time_window_proportion

    @staticmethod
    def _pairs(snapshot: Snapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if snapshot.query_edge_index is None:
            raise ValueError("EdgeBank requires event query edges")
        sources, destinations = snapshot.query_edge_index.detach().cpu().numpy()
        if snapshot.query_timestamps is None:
            times = np.full(len(sources), float(snapshot.time), dtype=np.float64)
        else:
            times = snapshot.query_timestamps.detach().cpu().numpy().astype(np.float64)
        order = np.argsort(times, kind="stable")
        return sources[order], destinations[order], times[order]

    @torch.no_grad()
    def evaluate_protocol(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int = 42,
    ) -> dict[str, float]:
        if (
            self.negative_ratio == 1.0
            and self.max_positive_pairs is None
            and self.eval_positive_batch_size is not None
        ):
            history_snapshots = unique_snapshots(history_windows)
            target_snapshots = unique_snapshots(windows, targets_only=True)
            history_entries = [self._pairs(snapshot) for snapshot in history_snapshots]
            target_entries = [self._pairs(snapshot) for snapshot in target_snapshots]
            history_sources = np.concatenate([entry[0] for entry in history_entries])
            history_destinations = np.concatenate([entry[1] for entry in history_entries])
            history_times = np.concatenate([entry[2] for entry in history_entries])
            target_sources = np.concatenate([entry[0] for entry in target_entries])
            target_destinations = np.concatenate([entry[1] for entry in target_entries])
            target_times = np.concatenate([entry[2] for entry in target_entries])
            pool = self.negative_destination_candidates
            if pool is None:
                pool = torch.unique(
                    torch.as_tensor(
                        np.concatenate([history_destinations, target_destinations])
                    ),
                    sorted=True,
                )
            pool_array = pool.detach().cpu().numpy().astype(np.int64)
            table_negatives = None
            if self.negative_edge_table is not None:
                table_negatives = self.negative_edge_table.for_snapshots(
                    [snapshot.time for snapshot in target_snapshots]
                )
                table_negatives.check_alignment(
                    target_sources, target_destinations, context="EdgeBank evaluation"
                )
            rng = np.random.RandomState(query_seed)
            scores: list[Tensor] = []
            labels: list[Tensor] = []
            groups: list[Tensor] = []
            group_offset = 0
            batch_size = self.eval_positive_batch_size
            for start in range(0, len(target_sources), batch_size):
                stop = min(start + batch_size, len(target_sources))
                if self.memory_mode == "time_window_memory":
                    threshold = np.quantile(
                        history_times, 1.0 - self.time_window_proportion
                    )
                    keep = history_times >= threshold
                else:
                    keep = np.ones(len(history_times), dtype=bool)
                seen = set(
                    zip(
                        history_sources[keep].tolist(),
                        history_destinations[keep].tolist(),
                    )
                )
                source = target_sources[start:stop]
                positive = target_destinations[start:stop]
                if table_negatives is None:
                    negative_source = source
                    negative = rng.choice(pool_array, size=len(source), replace=True)
                else:
                    negative_source = table_negatives.sources[start:stop]
                    negative = table_negatives.destinations[start:stop]
                positive_scores = torch.tensor(
                    [float((int(u), int(v)) in seen) for u, v in zip(source, positive)]
                )
                negative_scores = torch.tensor(
                    [
                        float((int(u), int(v)) in seen)
                        for u, v in zip(negative_source, negative)
                    ]
                )
                scores.extend([positive_scores, negative_scores])
                labels.extend(
                    [torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]
                )
                group = torch.arange(group_offset, group_offset + len(source))
                groups.extend([group, group])
                group_offset += len(source)
                history_sources = np.concatenate([history_sources, source])
                history_destinations = np.concatenate(
                    [history_destinations, positive]
                )
                history_times = np.concatenate(
                    [history_times, target_times[start:stop]]
                )
            return self.metrics(
                torch.cat(labels), torch.cat(scores), torch.cat(groups)
            )

        seen: set[tuple[int, int]] = set()
        for snapshot in unique_snapshots(history_windows):
            sources, destinations, _ = self._pairs(snapshot)
            seen.update(zip(sources.tolist(), destinations.tolist()))
        score_parts: list[Tensor] = []
        label_parts: list[Tensor] = []
        group_parts: list[Tensor] = []
        group_offset = 0
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(window, query_seed + window_index)
            sources, destinations, event_times = self._pairs(window[-1])
            event_cursor = 0
            query_times = (
                queries.timestamps
                if queries.timestamps is not None
                else queries.labels.new_full(queries.labels.shape, float(window[-1].time))
            )
            scores = torch.empty_like(queries.labels)
            for timestamp in sorted(set(query_times.detach().cpu().tolist())):
                before = int(np.searchsorted(event_times, timestamp, side="left"))
                seen.update(
                    zip(
                        sources[event_cursor:before].tolist(),
                        destinations[event_cursor:before].tolist(),
                    )
                )
                event_cursor = before
                rows = torch.nonzero(query_times == timestamp, as_tuple=False).flatten()
                for row in rows.detach().cpu().tolist():
                    pair = tuple(queries.pairs[row].detach().cpu().tolist())
                    scores[row] = float(pair in seen)
                through = int(np.searchsorted(event_times, timestamp, side="right"))
                seen.update(
                    zip(
                        sources[event_cursor:through].tolist(),
                        destinations[event_cursor:through].tolist(),
                    )
                )
                event_cursor = through
            seen.update(zip(sources[event_cursor:].tolist(), destinations[event_cursor:].tolist()))
            score_parts.append(scores)
            label_parts.append(queries.labels)
            group_parts.append(queries.group_ids + group_offset)
            group_offset += int(queries.group_ids.max().item()) + 1
        return self.metrics(
            torch.cat(label_parts), torch.cat(score_parts), torch.cat(group_parts)
        )
