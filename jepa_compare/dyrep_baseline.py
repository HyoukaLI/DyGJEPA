from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import Snapshot
from .temporal_event_utils import (
    EventStream,
    SharedLinkProtocol,
    TemporalNeighborIndex,
    snapshot_events,
    stream_events,
    unique_snapshots,
)


@dataclass
class _DyRepState:
    embeddings: list[Tensor]
    last_time: Tensor
    strengths: dict[tuple[int, int], Tensor]


class DyRepLinkBaseline(nn.Module, SharedLinkProtocol):
    """DyRep point-process baseline adapted to a single event type.

    DyRep's original recurrent embedding update, structural aggregation,
    softplus intensity and sampled survival objective are retained.  Wikipedia
    The datasets expose one interaction stream, so the model uses one intensity
    channel and builds its evolving neighborhood from observed events.
    """

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        bipartite_source_count: int | None,
        hidden_dim: int = 32,
        interaction_feature_dim: int = 172,
        neighbor_count: int = 20,
        survival_samples: int = 5,
        train_batch_size: int = 200,
        survival_weight: float = 1.0,
        negative_ratio: float = 20.0,
        max_positive_pairs: int | None = 1024,
        new_edges_only: bool = False,
        undirected: bool = False,
    ) -> None:
        super().__init__()
        del feature_dim, interaction_feature_dim
        if undirected:
            raise ValueError("DyRep event evaluation requires directed links")
        if bipartite_source_count is not None and not 0 < bipartite_source_count < num_nodes:
            raise ValueError("bipartite_source_count must split users and items")
        if min(hidden_dim, neighbor_count, survival_samples, train_batch_size) < 1:
            raise ValueError("DyRep dimensions and sample counts must be positive")

        self.num_nodes = num_nodes
        self.num_users = bipartite_source_count
        self.hidden_dim = hidden_dim
        self.neighbor_count = neighbor_count
        self.survival_samples = survival_samples
        self.train_batch_size = train_batch_size
        self.survival_weight = survival_weight
        self.negative_ratio = negative_ratio
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = new_edges_only

        initial = F.normalize(torch.rand(num_nodes, hidden_dim), dim=-1)
        self.initial_embeddings = nn.Parameter(initial)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.structure_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.recurrent_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.time_projection = nn.Linear(4, hidden_dim, bias=False)
        self.intensity_projection = nn.Linear(2 * hidden_dim, 1, bias=False)
        self.raw_psi = nn.Parameter(torch.tensor(0.0))

        self._train_stream: EventStream | None = None
        self._train_index: TemporalNeighborIndex | None = None
        self._full_index: TemporalNeighborIndex | None = None

    def prepare_streams(
        self,
        all_snapshots: Sequence[Snapshot],
        train_snapshots: Sequence[Snapshot],
    ) -> None:
        # DyRep does not consume Wikipedia's edge covariates, but stream_events
        # validates and preserves exactly the same event ordering as TGAT/JODIE.
        feature_dim = int(all_snapshots[0].query_features.shape[1]) if (
            all_snapshots[0].query_features is not None
        ) else 1
        full = stream_events(
            all_snapshots,
            num_users=self.num_users,
            num_nodes=self.num_nodes,
            feature_dim=feature_dim,
        )
        train = stream_events(
            train_snapshots,
            num_users=self.num_users,
            num_nodes=self.num_nodes,
            feature_dim=feature_dim,
        )
        self._train_stream = train
        self._train_index = TemporalNeighborIndex(train, self.num_nodes)
        self._full_index = TemporalNeighborIndex(full, self.num_nodes)

    def _initial_state(self, dtype: torch.dtype, device: torch.device) -> _DyRepState:
        embeddings = F.normalize(self.initial_embeddings, dim=-1)
        return _DyRepState(
            embeddings=[embeddings[node] for node in range(self.num_nodes)],
            last_time=torch.zeros(self.num_nodes, dtype=dtype, device=device),
            strengths={},
        )

    @staticmethod
    def _time_features(delta: Tensor) -> Tensor:
        delta = delta.clamp_min(0.0)
        day = delta / 86_400.0 / 50.0
        hour = torch.remainder(delta, 86_400.0) / 3_600.0 / 7.0
        minute = torch.remainder(delta, 3_600.0) / 60.0 / 15.0
        second = torch.remainder(delta, 60.0) / 15.0
        return torch.stack([day, hour, minute, second])

    def _psi(self) -> Tensor:
        return F.softplus(self.raw_psi) + 1e-6

    def _intensity(self, source: Tensor, destination: Tensor) -> Tensor:
        score = self.intensity_projection(
            torch.cat([source, destination], dim=-1)
        ).squeeze(-1)
        psi = self._psi()
        return psi * F.softplus(score / psi)

    def _aggregate(
        self,
        state: _DyRepState,
        node: int,
        timestamp: Tensor,
        index: TemporalNeighborIndex,
        rng: np.random.Generator,
    ) -> Tensor:
        node_tensor = torch.tensor([node], dtype=torch.long, device=timestamp.device)
        neighbor_nodes, _, _, mask = index.sample(
            node_tensor,
            timestamp.reshape(1),
            self.neighbor_count,
            uniform=False,
            rng=rng,
            device=timestamp.device,
        )
        valid_nodes = neighbor_nodes[0, ~mask[0]]
        if valid_nodes.numel() == 0:
            return timestamp.new_zeros(self.hidden_dim)
        neighbor_embeddings = torch.stack(
            [state.embeddings[int(neighbor)] for neighbor in valid_nodes]
        )
        strengths = torch.stack(
            [
                state.strengths.get(
                    (node, int(neighbor)), timestamp.new_tensor(1.0)
                )
                for neighbor in valid_nodes
            ]
        )
        weights = torch.softmax(strengths, dim=0).unsqueeze(-1)
        messages = weights * self.neighbor_projection(neighbor_embeddings)
        return messages.max(dim=0).values

    def _update_event(
        self,
        state: _DyRepState,
        source: Tensor,
        destination: Tensor,
        timestamp: Tensor,
        index: TemporalNeighborIndex,
        rng: np.random.Generator,
    ) -> None:
        source_id, destination_id = int(source), int(destination)
        source_previous = state.embeddings[source_id]
        destination_previous = state.embeddings[destination_id]
        source_context = self._aggregate(
            state, source_id, timestamp, index, rng
        )
        destination_context = self._aggregate(
            state, destination_id, timestamp, index, rng
        )
        source_delta = self._time_features(timestamp - state.last_time[source_id])
        destination_delta = self._time_features(
            timestamp - state.last_time[destination_id]
        )
        # DyRep updates each endpoint using the structural context around the
        # other endpoint of the observed event.
        source_updated = torch.sigmoid(
            self.structure_projection(destination_context)
            + self.recurrent_projection(source_previous)
            + self.time_projection(source_delta)
        )
        destination_updated = torch.sigmoid(
            self.structure_projection(source_context)
            + self.recurrent_projection(destination_previous)
            + self.time_projection(destination_delta)
        )
        intensity = self._intensity(source_previous, destination_previous)
        state.embeddings[source_id] = source_updated
        state.embeddings[destination_id] = destination_updated
        state.last_time[source_id] = timestamp.detach()
        state.last_time[destination_id] = timestamp.detach()
        state.strengths[(source_id, destination_id)] = intensity
        state.strengths[(destination_id, source_id)] = intensity

    @staticmethod
    def _detach_state(state: _DyRepState) -> None:
        state.embeddings = [embedding.detach() for embedding in state.embeddings]
        state.strengths = {
            pair: strength.detach() for pair, strength in state.strengths.items()
        }

    def train_epoch(
        self,
        windows: Sequence[Sequence[Snapshot]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int = 42,
    ) -> dict[str, float]:
        del windows
        if self._train_stream is None or self._train_index is None:
            raise RuntimeError("call prepare_streams before DyRep training")
        stream, index = self._train_stream, self._train_index
        state = self._initial_state(stream.timestamps.dtype, stream.timestamps.device)
        generator = torch.Generator().manual_seed(seed)
        rng = np.random.default_rng(seed)
        event_total = 0.0
        survival_total = 0.0
        chunks = 0

        for start in range(0, len(stream), self.train_batch_size):
            end = min(start + self.train_batch_size, len(stream))
            optimizer.zero_grad(set_to_none=True)
            event_losses: list[Tensor] = []
            survival_losses: list[Tensor] = []
            for row in range(start, end):
                source = stream.sources[row]
                destination = stream.destinations[row]
                timestamp = stream.timestamps[row]
                source_embedding = state.embeddings[int(source)]
                destination_embedding = state.embeddings[int(destination)]
                observed = self._intensity(source_embedding, destination_embedding)
                event_losses.append(-torch.log(observed.clamp_min(1e-9)))

                destination_start = 0 if self.num_users is None else self.num_users
                source_end = self.num_nodes if self.num_users is None else self.num_users
                negative_items = torch.randint(
                    destination_start,
                    self.num_nodes,
                    (self.survival_samples,),
                    generator=generator,
                    device="cpu",
                ).to(source.device)
                negative_users = torch.randint(
                    0,
                    source_end,
                    (self.survival_samples,),
                    generator=generator,
                    device="cpu",
                ).to(source.device)
                source_repeat = source_embedding.expand(self.survival_samples, -1)
                destination_repeat = destination_embedding.expand(
                    self.survival_samples, -1
                )
                item_embeddings = torch.stack(
                    [state.embeddings[int(item)] for item in negative_items]
                )
                user_embeddings = torch.stack(
                    [state.embeddings[int(user)] for user in negative_users]
                )
                survival_losses.append(
                    0.5
                    * (
                        self._intensity(source_repeat, item_embeddings).mean()
                        + self._intensity(user_embeddings, destination_repeat).mean()
                    )
                )
                self._update_event(
                    state, source, destination, timestamp, index, rng
                )

            event_loss = torch.stack(event_losses).mean()
            survival_loss = torch.stack(survival_losses).mean()
            loss = event_loss + self.survival_weight * survival_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
            self._detach_state(state)
            event_total += float(event_loss.detach().item())
            survival_total += float(survival_loss.detach().item())
            chunks += 1
        event_mean = event_total / max(1, chunks)
        survival_mean = survival_total / max(1, chunks)
        return {
            "event_nll": event_mean,
            "survival_loss": survival_mean,
            "loss": event_mean + self.survival_weight * survival_mean,
        }

    def _replay(
        self,
        state: _DyRepState,
        snapshots: Sequence[Snapshot],
        index: TemporalNeighborIndex,
        rng: np.random.Generator,
    ) -> None:
        if not snapshots:
            return
        feature_dim = int(snapshots[0].query_features.shape[1]) if (
            snapshots[0].query_features is not None
        ) else 1
        events = stream_events(
            snapshots,
            num_users=self.num_users,
            num_nodes=self.num_nodes,
            feature_dim=feature_dim,
        )
        for source, destination, timestamp in zip(
            events.sources, events.destinations, events.timestamps
        ):
            self._update_event(
                state, source, destination, timestamp, index, rng
            )

    @staticmethod
    def _event_key(source: int, destination: int, timestamp: float) -> tuple[int, int, float]:
        return source, destination, timestamp

    @torch.no_grad()
    def evaluate_protocol(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int = 42,
    ) -> dict[str, float]:
        if self._full_index is None:
            raise RuntimeError("call prepare_streams before DyRep evaluation")
        device = self.initial_embeddings.device
        dtype = self.initial_embeddings.dtype
        state = self._initial_state(dtype, device)
        rng = np.random.default_rng(query_seed)
        self._replay(
            state, unique_snapshots(history_windows), self._full_index, rng
        )

        probabilities: list[Tensor] = []
        labels: list[Tensor] = []
        groups: list[Tensor] = []
        group_offset = 0
        for window_index, window in enumerate(windows):
            target = window[-1]
            queries = self.sample_queries(window, query_seed + window_index)
            rows_by_group: dict[int, list[int]] = {}
            positive_groups: dict[tuple[int, int, float], list[int]] = {}
            for row, group in enumerate(queries.group_ids.detach().cpu().tolist()):
                rows_by_group.setdefault(group, []).append(row)
                if queries.labels[row] > 0:
                    timestamp = float(queries.timestamps[row].item())
                    key = self._event_key(
                        int(queries.pairs[row, 0]),
                        int(queries.pairs[row, 1]),
                        timestamp,
                    )
                    positive_groups.setdefault(key, []).append(group)

            feature_dim = int(target.query_features.shape[1]) if (
                target.query_features is not None
            ) else 1
            events = snapshot_events(
                target,
                num_users=self.num_users,
                num_nodes=self.num_nodes,
                feature_dim=feature_dim,
            )
            for source, destination, timestamp in zip(
                events.sources, events.destinations, events.timestamps
            ):
                key = self._event_key(
                    int(source), int(destination), float(timestamp.item())
                )
                queued = positive_groups.get(key)
                if queued:
                    group = queued.pop(0)
                    rows = torch.tensor(
                        rows_by_group[group], dtype=torch.long, device=device
                    )
                    candidates = queries.pairs[rows, 1]
                    source_embedding = state.embeddings[int(source)].expand(
                        rows.numel(), -1
                    )
                    candidate_embeddings = torch.stack(
                        [state.embeddings[int(candidate)] for candidate in candidates]
                    )
                    intensity = self._intensity(
                        source_embedding, candidate_embeddings
                    )
                    probabilities.append(1.0 - torch.exp(-intensity.clamp_max(30.0)))
                    labels.append(queries.labels[rows])
                    groups.append(
                        torch.full(
                            (rows.numel(),),
                            group_offset,
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    group_offset += 1
                self._update_event(
                    state, source, destination, timestamp, self._full_index, rng
                )
            unmatched = sum(len(queue) for queue in positive_groups.values())
            if unmatched:
                raise RuntimeError(
                    f"{unmatched} sampled positive events were not found in the DyRep stream"
                )
        if not probabilities:
            raise ValueError("DyRep evaluation produced no queries")
        return self.metrics(
            torch.cat(labels), torch.cat(probabilities), torch.cat(groups)
        )


def prepare_dyrep_streams(
    model: DyRepLinkBaseline,
    all_snapshots: Sequence[Snapshot],
    train_windows: Sequence[Sequence[Snapshot]],
) -> None:
    model.prepare_streams(all_snapshots, unique_snapshots(train_windows))
