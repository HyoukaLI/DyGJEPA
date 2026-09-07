from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import Snapshot
from .link_prediction import (
    LinkQueries,
    link_prediction_metrics,
    sample_link_queries,
)


@dataclass
class _JODIEState:
    users: list[Tensor]
    items: list[Tensor]
    last_user_time: Tensor
    last_item_time: Tensor
    last_item_for_user: Tensor


class JODIELinkBaseline(nn.Module):
    """Original JODIE internals under the suite's shared evaluation protocol.

    JODIE keeps its coupled RNNs, elapsed-time projection, one-hot static
    identities, linear future-item predictor, item-embedding MSE, smoothness
    loss, time-consistent t-batches, and continuous event-stream state.  The
    comparison suite still controls the temporal split, sampled candidates,
    metrics, and validation checkpoint rule for both methods.
    """

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int,
        bipartite_source_count: int,
        hidden_dim: int = 128,
        interaction_feature_dim: int = 172,
        negative_ratio: float = 1.0,
        max_positive_pairs: int | None = 512,
        new_edges_only: bool = False,
        undirected: bool = False,
        negative_destination_candidates: Tensor | None = None,
        allow_negative_collisions: bool = False,
        eval_positive_batch_size: int | None = None,
        tbatch_count: int = 500,
        state_change: bool = True,
    ) -> None:
        super().__init__()
        del feature_dim
        if undirected:
            raise ValueError("JODIE requires directed user-to-item interactions")
        if not 0 < bipartite_source_count < num_nodes:
            raise ValueError("bipartite_source_count must split users and items")
        if interaction_feature_dim < 1:
            raise ValueError("interaction_feature_dim must be positive")
        if tbatch_count < 1:
            raise ValueError("tbatch_count must be positive")

        self.num_nodes = num_nodes
        self.num_users = bipartite_source_count
        self.num_items = num_nodes - bipartite_source_count
        self.num_state_items = self.num_items + 1
        self.hidden_dim = hidden_dim
        self.interaction_feature_dim = interaction_feature_dim
        self.negative_ratio = negative_ratio
        self.max_positive_pairs = max_positive_pairs
        self.new_edges_only = new_edges_only
        self.undirected = undirected
        self.negative_destination_candidates = negative_destination_candidates
        self.allow_negative_collisions = allow_negative_collisions
        self.eval_positive_batch_size = eval_positive_batch_size
        self.tbatch_count = tbatch_count
        self.state_change = state_change

        self.initial_user_embedding = nn.Parameter(torch.empty(hidden_dim))
        self.initial_item_embedding = nn.Parameter(torch.empty(hidden_dim))
        with torch.no_grad():
            self.initial_user_embedding.copy_(F.normalize(torch.rand(hidden_dim), dim=0))
            self.initial_item_embedding.copy_(F.normalize(torch.rand(hidden_dim), dim=0))

        recurrent_input_dim = hidden_dim + 1 + interaction_feature_dim
        self.user_rnn = nn.RNNCell(recurrent_input_dim, hidden_dim)
        self.item_rnn = nn.RNNCell(recurrent_input_dim, hidden_dim)
        self.time_projection = nn.Linear(1, hidden_dim)
        nn.init.normal_(self.time_projection.weight, std=1.0)
        nn.init.normal_(self.time_projection.bias, std=1.0)

        predictor_input_dim = hidden_dim * 2 + self.num_users + self.num_state_items
        predictor_output_dim = hidden_dim + self.num_state_items
        self.prediction_layer = nn.Linear(predictor_input_dim, predictor_output_dim)
        self.state_layer1 = nn.Linear(hidden_dim, 50)
        self.state_layer2 = nn.Linear(50, 2)

        self.register_buffer("time_origin", torch.tensor(0.0))
        self.register_buffer("user_delta_mean", torch.tensor(0.0))
        self.register_buffer("user_delta_std", torch.tensor(1.0))
        self.register_buffer("item_delta_mean", torch.tensor(0.0))
        self.register_buffer("item_delta_std", torch.tensor(1.0))
        self.register_buffer("state_positive_weight", torch.tensor(1.0))
        self.register_buffer("tbatch_span", torch.tensor(1.0))
        self.register_buffer("time_statistics_fitted", torch.tensor(False))

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

    @staticmethod
    def _unique_snapshots(
        windows: Sequence[Sequence[Snapshot]], *, targets_only: bool = False
    ) -> list[Snapshot]:
        snapshots: dict[int, Snapshot] = {}
        for window in windows:
            selected = [window[-1]] if targets_only else window
            for snapshot in selected:
                snapshots[snapshot.time] = snapshot
        return [snapshots[key] for key in sorted(snapshots)]

    def _snapshot_events(
        self, snapshot: Snapshot
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if snapshot.query_edge_index is None:
            raise ValueError("JODIE requires duplicate-preserving query events")
        source, destination = snapshot.query_edge_index
        valid = (
            (source < self.num_users)
            & (destination >= self.num_users)
            & (destination < self.num_nodes)
        )
        users = source[valid]
        items = destination[valid] - self.num_users
        if snapshot.query_timestamps is None:
            timestamps = torch.full(
                (users.shape[0],),
                float(snapshot.time),
                dtype=snapshot.x.dtype,
                device=snapshot.x.device,
            )
        else:
            timestamps = snapshot.query_timestamps[valid]
        if snapshot.query_features is None:
            features = torch.zeros(
                users.shape[0],
                self.interaction_feature_dim,
                dtype=snapshot.x.dtype,
                device=snapshot.x.device,
            )
        else:
            features = snapshot.query_features[valid]
            if features.shape[1] != self.interaction_feature_dim:
                raise ValueError(
                    "JODIE interaction feature width is "
                    f"{features.shape[1]}, expected {self.interaction_feature_dim}; "
                    "regenerate Wikipedia with scripts/prepare_wikipedia.py"
                )
        if snapshot.query_labels is None:
            labels = torch.zeros(users.shape[0], dtype=torch.long, device=users.device)
        else:
            labels = snapshot.query_labels[valid]
        order = torch.argsort(timestamps, stable=True)
        return (
            users[order],
            items[order],
            timestamps[order],
            features[order],
            labels[order],
        )

    def _stream_events(
        self, snapshots: Sequence[Snapshot]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        entries = [self._snapshot_events(snapshot) for snapshot in snapshots]
        users = torch.cat([entry[0] for entry in entries])
        items = torch.cat([entry[1] for entry in entries])
        timestamps = torch.cat([entry[2] for entry in entries])
        features = torch.cat([entry[3] for entry in entries])
        labels = torch.cat([entry[4] for entry in entries])
        order = torch.argsort(timestamps, stable=True)
        return users[order], items[order], timestamps[order], features[order], labels[order]

    def _initial_state(self, device: torch.device, dtype: torch.dtype) -> _JODIEState:
        user = F.normalize(self.initial_user_embedding, dim=0)
        item = F.normalize(self.initial_item_embedding, dim=0)
        return _JODIEState(
            users=[user] * self.num_users,
            items=[item] * self.num_state_items,
            last_user_time=torch.zeros(self.num_users, dtype=dtype, device=device),
            last_item_time=torch.zeros(self.num_state_items, dtype=dtype, device=device),
            last_item_for_user=torch.full(
                (self.num_users,), self.num_items, dtype=torch.long, device=device
            ),
        )

    def _fit_stream_statistics(
        self,
        users: Tensor,
        items: Tensor,
        timestamps: Tensor,
        labels: Tensor,
        class_weight_labels: Tensor | None = None,
    ) -> None:
        origin = timestamps[0]
        relative = timestamps - origin
        last_user: dict[int, float] = {}
        last_item: dict[int, float] = {}
        user_deltas: list[float] = []
        item_deltas: list[float] = []
        for user, item, timestamp in zip(
            users.detach().cpu().tolist(),
            items.detach().cpu().tolist(),
            relative.detach().cpu().tolist(),
        ):
            user_deltas.append(timestamp - last_user.get(user, 0.0) + 1.0)
            item_deltas.append(timestamp - last_item.get(item, 0.0) + 1.0)
            last_user[user] = timestamp
            last_item[item] = timestamp
        user_tensor = torch.tensor(user_deltas, dtype=timestamps.dtype, device=timestamps.device)
        item_tensor = torch.tensor(item_deltas, dtype=timestamps.dtype, device=timestamps.device)
        self.time_origin.copy_(origin.detach())
        self.user_delta_mean.copy_(user_tensor.mean())
        self.user_delta_std.copy_(user_tensor.std(unbiased=False).clamp_min(1e-6))
        self.item_delta_mean.copy_(item_tensor.mean())
        self.item_delta_std.copy_(item_tensor.std(unbiased=False).clamp_min(1e-6))
        weight_labels = labels if class_weight_labels is None else class_weight_labels
        self.state_positive_weight.copy_(
            torch.tensor(
                weight_labels.numel() / (float(weight_labels.sum().item()) + 1.0),
                dtype=timestamps.dtype,
                device=timestamps.device,
            )
        )
        self.tbatch_span.copy_(
            (timestamps[-1] - timestamps[0]).abs().clamp_min(1e-6)
            / self.tbatch_count
        )
        self.time_statistics_fitted.fill_(True)

    def fit_stream_statistics(
        self,
        snapshots: Sequence[Snapshot],
        label_snapshots: Sequence[Snapshot] | None = None,
    ) -> None:
        """Fit JODIE time scaling while keeping label weights train-only."""
        users, items, timestamps, _, labels = self._stream_events(snapshots)
        class_weight_labels = None
        if label_snapshots is not None:
            _, _, _, _, class_weight_labels = self._stream_events(label_snapshots)
        self._fit_stream_statistics(
            users, items, timestamps, labels, class_weight_labels=class_weight_labels
        )

    def _relative_time(self, timestamp: Tensor) -> Tensor:
        return timestamp - self.time_origin.to(timestamp.dtype)

    def _scaled_user_delta(self, timestamp: Tensor, previous: Tensor) -> Tensor:
        delta = self._relative_time(timestamp) - previous
        return (delta + 1.0 - self.user_delta_mean) / self.user_delta_std

    def _scaled_item_delta(self, timestamp: Tensor, previous: Tensor) -> Tensor:
        delta = self._relative_time(timestamp) - previous
        return (delta + 1.0 - self.item_delta_mean) / self.item_delta_std

    @staticmethod
    def _time_consistent_batches(users: Tensor, items: Tensor) -> list[list[int]]:
        last_user: dict[int, int] = {}
        last_item: dict[int, int] = {}
        batches: list[list[int]] = []
        for event_index, (user, item) in enumerate(
            zip(users.detach().cpu().tolist(), items.detach().cpu().tolist())
        ):
            batch_index = max(last_user.get(user, -1), last_item.get(item, -1)) + 1
            if batch_index == len(batches):
                batches.append([])
            batches[batch_index].append(event_index)
            last_user[user] = batch_index
            last_item[item] = batch_index
        return batches

    def _static_users(self, users: Tensor, dtype: torch.dtype) -> Tensor:
        return F.one_hot(users, self.num_users).to(dtype)

    def _static_items(self, items: Tensor, dtype: torch.dtype) -> Tensor:
        return F.one_hot(items, self.num_state_items).to(dtype)

    def _train_tbatch(
        self,
        state: _JODIEState,
        users: Tensor,
        items: Tensor,
        timestamps: Tensor,
        features: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        previous_users = torch.stack([state.users[int(user)] for user in users])
        previous_items = torch.stack([state.items[int(item)] for item in items])
        previous_item_ids = state.last_item_for_user[users]
        previous_user_items = torch.stack(
            [state.items[int(item)] for item in previous_item_ids]
        )
        user_delta = self._scaled_user_delta(
            timestamps, state.last_user_time[users]
        ).unsqueeze(-1)
        item_delta = self._scaled_item_delta(
            timestamps, state.last_item_time[items]
        ).unsqueeze(-1)

        projected_users = previous_users * (1.0 + self.time_projection(user_delta))
        predictor_input = torch.cat(
            [
                projected_users,
                previous_user_items,
                self._static_items(previous_item_ids, features.dtype),
                self._static_users(users, features.dtype),
            ],
            dim=-1,
        )
        prediction = self.prediction_layer(predictor_input)
        target = torch.cat(
            [previous_items, self._static_items(items, features.dtype)], dim=-1
        ).detach()
        prediction_loss = F.mse_loss(prediction, target)

        updated_users = F.normalize(
            self.user_rnn(
                torch.cat([previous_items, user_delta, features], dim=-1),
                previous_users,
            ),
            dim=-1,
        )
        updated_items = F.normalize(
            self.item_rnn(
                torch.cat([previous_users, item_delta, features], dim=-1),
                previous_items,
            ),
            dim=-1,
        )
        smoothness_loss = F.mse_loss(
            updated_users, previous_users.detach()
        ) + F.mse_loss(updated_items, previous_items.detach())
        if self.state_change:
            state_logits = self.state_layer2(F.relu(self.state_layer1(updated_users)))
            state_weight = torch.stack(
                [torch.ones_like(self.state_positive_weight), self.state_positive_weight]
            ).to(device=features.device, dtype=features.dtype)
            state_loss = F.cross_entropy(state_logits, labels, weight=state_weight)
        else:
            state_loss = features.new_zeros(())

        for position, (user, item) in enumerate(zip(users, items)):
            user_id, item_id = int(user), int(item)
            state.users[user_id] = updated_users[position]
            state.items[item_id] = updated_items[position]
            relative_time = self._relative_time(timestamps[position]).detach()
            state.last_user_time[user_id] = relative_time
            state.last_item_time[item_id] = relative_time
            state.last_item_for_user[user_id] = item_id
        return prediction_loss, smoothness_loss, state_loss

    @staticmethod
    def _detach_state(state: _JODIEState) -> None:
        state.users = [embedding.detach() for embedding in state.users]
        state.items = [embedding.detach() for embedding in state.items]

    def train_epoch(
        self,
        windows: Sequence[Sequence[Snapshot]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
    ) -> dict[str, float]:
        snapshots = self._unique_snapshots(windows)
        users, items, timestamps, features, labels = self._stream_events(snapshots)
        if not bool(self.time_statistics_fitted.item()):
            self._fit_stream_statistics(users, items, timestamps, labels)
        state = self._initial_state(timestamps.device, timestamps.dtype)
        chunk_span = float(self.tbatch_span.item())
        chunk_start = 0
        chunk_time = float(timestamps[0].item())
        prediction_total = 0.0
        smoothness_total = 0.0
        state_total = 0.0
        batch_total = 0

        for end in range(1, timestamps.shape[0] + 1):
            boundary = end == timestamps.shape[0]
            if not boundary:
                boundary = float(timestamps[end].item()) - chunk_time > chunk_span
            if not boundary:
                continue
            chunk_users = users[chunk_start:end]
            chunk_items = items[chunk_start:end]
            chunk_times = timestamps[chunk_start:end]
            chunk_features = features[chunk_start:end]
            chunk_labels = labels[chunk_start:end]
            optimizer.zero_grad(set_to_none=True)
            losses: list[Tensor] = []
            for indices in self._time_consistent_batches(chunk_users, chunk_items):
                index = torch.tensor(indices, dtype=torch.long, device=users.device)
                prediction_loss, smoothness_loss, state_loss = self._train_tbatch(
                    state,
                    chunk_users[index],
                    chunk_items[index],
                    chunk_times[index],
                    chunk_features[index],
                    chunk_labels[index],
                )
                losses.append(prediction_loss + smoothness_loss + state_loss)
                prediction_total += float(prediction_loss.detach().item())
                smoothness_total += float(smoothness_loss.detach().item())
                state_total += float(state_loss.detach().item())
                batch_total += 1
            torch.stack(losses).sum().backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
            self._detach_state(state)
            chunk_start = end
            if end < timestamps.shape[0]:
                chunk_time = float(timestamps[end].item())

        denominator = max(1, batch_total)
        prediction_mean = prediction_total / denominator
        smoothness_mean = smoothness_total / denominator
        state_mean = state_total / denominator
        return {
            "prediction_loss": prediction_mean,
            "smoothness_loss": smoothness_mean,
            "state_loss": state_mean,
            "loss": prediction_mean + smoothness_mean + state_mean,
        }

    def _update_one(
        self,
        state: _JODIEState,
        user: Tensor,
        item: Tensor,
        timestamp: Tensor,
        feature: Tensor,
    ) -> None:
        user_id, item_id = int(user), int(item)
        previous_user = state.users[user_id].unsqueeze(0)
        previous_item = state.items[item_id].unsqueeze(0)
        user_delta = self._scaled_user_delta(
            timestamp, state.last_user_time[user_id]
        ).reshape(1, 1)
        item_delta = self._scaled_item_delta(
            timestamp, state.last_item_time[item_id]
        ).reshape(1, 1)
        updated_user = F.normalize(
            self.user_rnn(
                torch.cat([previous_item, user_delta, feature.unsqueeze(0)], dim=-1),
                previous_user,
            ),
            dim=-1,
        )[0]
        updated_item = F.normalize(
            self.item_rnn(
                torch.cat([previous_user, item_delta, feature.unsqueeze(0)], dim=-1),
                previous_item,
            ),
            dim=-1,
        )[0]
        state.users[user_id] = updated_user
        state.items[item_id] = updated_item
        relative_time = self._relative_time(timestamp)
        state.last_user_time[user_id] = relative_time
        state.last_item_time[item_id] = relative_time
        state.last_item_for_user[user_id] = item_id

    def _score_candidates(
        self,
        state: _JODIEState,
        user: Tensor,
        candidate_items: Tensor,
        timestamp: Tensor,
    ) -> Tensor:
        user_id = int(user)
        previous_item_id = state.last_item_for_user[user_id]
        user_embedding = state.users[user_id].unsqueeze(0)
        user_delta = self._scaled_user_delta(
            timestamp, state.last_user_time[user_id]
        ).reshape(1, 1)
        projected_user = user_embedding * (1.0 + self.time_projection(user_delta))
        previous_item = state.items[int(previous_item_id)].unsqueeze(0)
        predictor_input = torch.cat(
            [
                projected_user,
                previous_item,
                self._static_items(previous_item_id.reshape(1), user_embedding.dtype),
                self._static_users(user.reshape(1), user_embedding.dtype),
            ],
            dim=-1,
        )
        predicted_item = self.prediction_layer(predictor_input)
        candidate_dynamic = torch.stack([state.items[int(item)] for item in candidate_items])
        candidate = torch.cat(
            [candidate_dynamic, self._static_items(candidate_items, candidate_dynamic.dtype)],
            dim=-1,
        )
        distance = (predicted_item - candidate).square().sum(dim=-1).sqrt()
        return torch.sigmoid(-distance)

    def _replay(self, state: _JODIEState, snapshots: Sequence[Snapshot]) -> None:
        if not snapshots:
            return
        users, items, timestamps, features, _ = self._stream_events(snapshots)
        for user, item, timestamp, feature in zip(users, items, timestamps, features):
            self._update_one(state, user, item, timestamp, feature)

    @staticmethod
    def _event_key(user: int, item_global: int, timestamp: float) -> tuple[int, int, float]:
        return user, item_global, timestamp

    @torch.no_grad()
    def evaluate_protocol(
        self,
        windows: Sequence[Sequence[Snapshot]],
        history_windows: Sequence[Sequence[Snapshot]],
        query_seed: int = 42,
    ) -> dict[str, float]:
        if not bool(self.time_statistics_fitted.item()):
            raise RuntimeError("JODIE time statistics must be fitted before evaluation")
        history = self._unique_snapshots(history_windows)
        device = self.initial_user_embedding.device
        dtype = self.initial_user_embedding.dtype
        state = self._initial_state(device, dtype)
        self._replay(state, history)

        probabilities: list[Tensor] = []
        labels: list[Tensor] = []
        groups: list[Tensor] = []
        group_offset = 0
        for window_index, window in enumerate(windows):
            target_snapshot = window[-1]
            queries = self.sample_queries(window, query_seed + window_index)
            rows_by_group: dict[int, list[int]] = {}
            positive_groups: dict[tuple[int, int, float], list[int]] = {}
            for row, group in enumerate(queries.group_ids.detach().cpu().tolist()):
                rows_by_group.setdefault(group, []).append(row)
                if queries.labels[row] > 0:
                    timestamp = (
                        float(queries.timestamps[row].item())
                        if queries.timestamps is not None
                        else float(target_snapshot.time)
                    )
                    key = self._event_key(
                        int(queries.pairs[row, 0]),
                        int(queries.pairs[row, 1]),
                        timestamp,
                    )
                    positive_groups.setdefault(key, []).append(group)

            users, items, timestamps, features, _ = self._snapshot_events(target_snapshot)
            for user, item, timestamp, feature in zip(users, items, timestamps, features):
                global_item = int(item) + self.num_users
                key = self._event_key(int(user), global_item, float(timestamp.item()))
                queued = positive_groups.get(key)
                if queued:
                    group = queued.pop(0)
                    rows = torch.tensor(rows_by_group[group], dtype=torch.long, device=device)
                    candidate_items = queries.pairs[rows, 1] - self.num_users
                    probabilities.append(
                        self._score_candidates(state, user, candidate_items, timestamp)
                    )
                    labels.append(queries.labels[rows])
                    groups.append(
                        torch.full(
                            (rows.numel(),), group_offset, dtype=torch.long, device=device
                        )
                    )
                    group_offset += 1
                self._update_one(state, user, item, timestamp, feature)
            unmatched = sum(len(queue) for queue in positive_groups.values())
            if unmatched:
                raise RuntimeError(
                    f"{unmatched} sampled positive events were not found in the JODIE stream"
                )

        if not probabilities:
            raise ValueError("JODIE evaluation produced no sampled query groups")
        probability = torch.cat(probabilities)
        target = torch.cat(labels)
        group_ids = torch.cat(groups)
        return link_prediction_metrics(
            target,
            probability,
            group_ids,
            positive_batch_size=self.eval_positive_batch_size,
        )
