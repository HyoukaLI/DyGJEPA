from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import Snapshot
from .encoding import random_walk_positional_encoding, sinusoidal_time_encoding
from .layers import GraphSAGE
from .link_prediction import (
    NODE_EVENT_DIM,
    PAIR_STAT_DIM,
    LinkQueries,
    link_prediction_metrics,
    neighbor_mean_embeddings,
    relation_context_nodes,
    sample_link_queries,
    temporal_pair_increments,
    temporal_node_increments,
)
from .signature import signature_dimension, truncated_signature


RELATION_HISTORY_STAT_DIM = 8


@dataclass
class RCPSWindowOutput:
    node_u_prediction: Tensor
    node_v_prediction: Tensor
    relation_prediction: Tensor
    node_u_target: Tensor
    node_v_target: Tensor
    relation_target: Tensor
    logit: Tensor
    intensity: Tensor
    probability: Tensor
    context: Tensor


@dataclass
class RCPSPreparedWindow:
    context_snapshots: list[Snapshot]
    target_snapshot: Snapshot
    context_embeddings: Tensor
    target_embedding: Tensor
    node_context: Tensor | None = None
    mean_relation_gate: Tensor | None = None

    def clear_node_cache(self) -> None:
        self.node_context = None
        self.mean_relation_gate = None


@dataclass
class RCPSNodeOutput:
    prediction: Tensor
    target: Tensor
    node_ids: Tensor
    mean_relation_gate: Tensor
    representation: Tensor


def _normalized_distance(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return (prediction - target).square().sum(dim=-1).mean()


def _contrastive_prediction_loss(
    prediction: Tensor, target: Tensor, temperature: float
) -> Tensor:
    if prediction.shape[0] < 2:
        return prediction.sum() * 0.0
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


_QUERY_CAP_UNSET = object()


def _masked_distance(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not bool(mask.any()):
        return prediction.sum() * 0.0
    return _normalized_distance(prediction[mask], target[mask])


def _variance_covariance_loss(embedding: Tensor, target_std: float) -> tuple[Tensor, Tensor]:
    if embedding.shape[0] < 2:
        zero = embedding.sum() * 0.0
        return zero, zero
    centered = embedding - embedding.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(target_std - std).mean()
    covariance = centered.T @ centered / max(1, embedding.shape[0] - 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    covariance_loss = off_diagonal.square().sum() / embedding.shape[1]
    return variance, covariance_loss


def _groupwise_ranking_loss(logits: Tensor, labels: Tensor, group_ids: Tensor) -> Tensor:
    """Listwise cross-entropy for one-positive candidate groups."""
    order = torch.argsort(group_ids, stable=True)
    sorted_groups = group_ids[order]
    _, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
    if counts.numel() == 0:
        return logits.sum() * 0.0
    if bool(torch.all(counts == counts[0])):
        width = int(counts[0].item())
        group_logits = logits[order].reshape(-1, width)
        group_labels = labels[order].reshape(-1, width)
        if not bool(torch.all(group_labels.sum(dim=1) == 1)):
            raise ValueError("each ranking group must contain one positive")
        return F.cross_entropy(group_logits, group_labels.argmax(dim=1))
    losses = []
    start = 0
    for count in counts.tolist():
        rows = order[start : start + count]
        positives = torch.nonzero(labels[rows] > 0.5, as_tuple=False).flatten()
        if positives.numel() != 1:
            raise ValueError("each ranking group must contain one positive")
        losses.append(F.cross_entropy(logits[rows].unsqueeze(0), positives))
        start += count
    return torch.stack(losses).mean()


def _query_group_batches(queries: LinkQueries, pair_batch_size: int | None) -> list[Tensor]:
    """Keep all candidates for a query group in the same optimizer batch."""
    order = torch.argsort(queries.group_ids, stable=True)
    sorted_groups = queries.group_ids[order]
    _, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
    if pair_batch_size is None:
        return [order]
    if pair_batch_size < int(counts.max().item()):
        raise ValueError("pair_batch_size is smaller than one candidate group")
    ends = counts.cumsum(0)
    batches = []
    group_start = 0
    while group_start < counts.numel():
        row_start = 0 if group_start == 0 else int(ends[group_start - 1].item())
        group_end = group_start
        while group_end < counts.numel():
            row_end = int(ends[group_end].item())
            if group_end > group_start and row_end - row_start > pair_batch_size:
                break
            group_end += 1
        row_end = int(ends[group_end - 1].item())
        batches.append(order[row_start:row_end])
        group_start = group_end
    return batches


class RCPSJEPA(nn.Module):
    """Relation-Centric Path-Signature JEPA on the SG-JEPA snapshot interface.

    Every snapshot contributes a path increment. This is the directly comparable
    discrete-time version of the event-stream method described in the design
    document; exact event timestamps can later replace snapshot times without
    changing the JEPA or survival heads.
    """

    def __init__(
        self,
        feature_dim: int,
        num_nodes: int | None = None,
        hidden_dim: int = 64,
        rwpe_dim: int = 8,
        rwpe_walks: int = 16,
        time_dim: int = 16,
        gnn_layers: int = 2,
        window_size: int = 4,
        predictor_hidden_dim: int = 128,
        event_dim: int = 8,
        signature_depth: int = 2,
        subgraph_budget: int = 32,
        path_decay: float = 0.8,
        bridge_weight: float = 1.0,
        ema_momentum: float = 0.99,
        negative_ratio: float = 1.0,
        train_negative_ratio: float | None = None,
        max_positive_pairs: int | None = 512,
        train_max_positive_pairs: int | None = None,
        new_edges_only: bool = True,
        undirected: bool = True,
        bipartite_source_count: int | None = None,
        negative_destination_candidates: Tensor | None = None,
        allow_negative_collisions: bool = False,
        eval_positive_batch_size: int | None = None,
        node_loss_weight: float = 1.0,
        node_contrastive_loss_weight: float = 0.5,
        contrastive_temperature: float = 0.2,
        relation_loss_weight: float = 1.0,
        link_loss_weight: float = 1.0,
        rank_loss_weight: float = 0.0,
        variance_loss_weight: float = 0.05,
        covariance_loss_weight: float = 0.005,
        variance_target: float = 0.5,
        rwpe_seed: int = 42,
        cache_rwpe: bool = True,
        use_causal_history: bool = False,
        history_semantic_dim: int = 0,
        id_embedding_dim: int = 0,
        id_embedding_dropout: float = 0.0,
        initial_id_score_scale: float = 1.0,
        ema_steps_per_epoch: int | None = None,
        ema_update_per_step: bool = False,
        shuffle_windows: bool = False,
        initial_link_logit_bias: float = -3.0,
    ) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if not 0 <= ema_momentum < 1:
            raise ValueError("ema_momentum must be in [0, 1)")
        if event_dim < 1:
            raise ValueError("event_dim must be positive")
        if negative_ratio <= 0 or (
            train_negative_ratio is not None and train_negative_ratio <= 0
        ):
            raise ValueError("negative ratios must be positive")
        if ema_steps_per_epoch is not None and ema_steps_per_epoch < 1:
            raise ValueError("ema_steps_per_epoch must be positive")
        if history_semantic_dim < 0 or id_embedding_dim < 0:
            raise ValueError("history/id embedding dimensions must be non-negative")
        if not 0.0 <= id_embedding_dropout < 1.0:
            raise ValueError("id_embedding_dropout must be in [0, 1)")
        if id_embedding_dim and num_nodes is None:
            raise ValueError("num_nodes is required when id_embedding_dim is positive")
        self.hidden_dim = hidden_dim
        self.rwpe_dim = rwpe_dim
        self.rwpe_walks = rwpe_walks
        self.time_dim = time_dim
        self.window_size = window_size
        self.subgraph_budget = subgraph_budget
        self.path_decay = path_decay
        self.bridge_weight = bridge_weight
        self.ema_momentum = ema_momentum
        self.negative_ratio = negative_ratio
        self.train_negative_ratio = (
            negative_ratio
            if train_negative_ratio is None
            else float(train_negative_ratio)
        )
        self.max_positive_pairs = max_positive_pairs
        # None keeps every target-window positive during training; evaluation
        # still uses ``max_positive_pairs`` so the reported candidate protocol
        # stays comparable with the other Wikipedia methods.
        self.train_max_positive_pairs = train_max_positive_pairs
        self.new_edges_only = new_edges_only
        self.undirected = undirected
        self.bipartite_source_count = bipartite_source_count
        self.negative_destination_candidates = negative_destination_candidates
        self.allow_negative_collisions = allow_negative_collisions
        self.eval_positive_batch_size = eval_positive_batch_size
        self.signature_depth = signature_depth
        self.node_loss_weight = node_loss_weight
        self.node_contrastive_loss_weight = node_contrastive_loss_weight
        self.contrastive_temperature = contrastive_temperature
        self.relation_loss_weight = relation_loss_weight
        self.link_loss_weight = link_loss_weight
        self.rank_loss_weight = rank_loss_weight
        self.variance_loss_weight = variance_loss_weight
        self.covariance_loss_weight = covariance_loss_weight
        self.variance_target = variance_target
        self.rwpe_seed = rwpe_seed
        self.cache_rwpe = cache_rwpe
        self.use_causal_history = use_causal_history
        self.history_semantic_dim = history_semantic_dim
        self.id_embedding_dropout = float(id_embedding_dropout)
        self.history_feature_dim = (
            RELATION_HISTORY_STAT_DIM + 3 * history_semantic_dim
        )
        self.ema_steps_per_epoch = ema_steps_per_epoch
        self.ema_update_per_step = bool(ema_update_per_step)
        self.shuffle_windows = shuffle_windows
        self._rwpe_cache: dict[tuple[int, int, str], Tensor] = {}
        self._history_time_to_row: dict[int, int] = {}
        self._history_pair_keys: Tensor | None = None
        self._history_pair_counts: Tensor | None = None
        self._history_previous_bin_counts: Tensor | None = None
        self._history_pair_last_times: Tensor | None = None
        self._history_pair_active_bins: Tensor | None = None
        self._history_node_counts: Tensor | None = None
        self._history_pair_semantics: Tensor | None = None
        self._history_node_semantics: Tensor | None = None
        # Compact lexicographic indexes for event-exact causal counts. They let
        # the discrete snapshot encoder retain its interface while the link
        # head sees interactions strictly before each query timestamp,
        # including earlier events from the same snapshot.
        self._exact_event_times: Tensor | None = None
        self._exact_pair_codes: Tensor | None = None
        self._exact_node_codes: Tensor | None = None
        self._exact_event_stride: int | None = None

        encoder_dim = feature_dim + rwpe_dim + time_dim
        self.feature_skip = nn.Linear(feature_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.feature_skip.weight)
        self.feature_skip.requires_grad_(False)
        # Start with a usable GNN mix; sigmoid(-2) kept GraphSAGE nearly off and
        # made the frozen probe collapse to a slightly noisier DeepWalk copy.
        self.snapshot_graph_logit = nn.Parameter(torch.tensor(0.0))
        self.online_encoder = GraphSAGE(encoder_dim, hidden_dim, gnn_layers)
        self.target_encoder = deepcopy(self.online_encoder)
        self.target_encoder.requires_grad_(False)

        self.node_gru = nn.GRU(hidden_dim, hidden_dim)
        self.node_history_norm = nn.LayerNorm(hidden_dim)
        self.node_history_logit = nn.Parameter(torch.tensor(-1.0))
        self.node_homophily_logit = nn.Parameter(torch.tensor(0.0))
        self.node_relation_gru = nn.GRU(hidden_dim, hidden_dim)
        self.graph_gru = nn.GRU(hidden_dim, hidden_dim)
        self.event_projector = nn.Sequential(
            nn.Linear(PAIR_STAT_DIM + 1, event_dim),
            nn.Tanh(),
        )
        signature_dim = signature_dimension(event_dim, signature_depth)
        self.signature_projector = nn.Sequential(
            nn.Linear(signature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.node_event_projector = nn.Sequential(
            nn.Linear(NODE_EVENT_DIM + 1, event_dim),
            nn.Tanh(),
        )
        self.node_signature_projector = nn.Sequential(
            nn.Linear(signature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.node_context_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, predictor_hidden_dim),
            nn.LayerNorm(predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )
        self.node_context_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.node_context_norm = nn.LayerNorm(hidden_dim)

        # The relation JEPA target is composed only from future endpoint latents.
        # Explicit target-snapshot pair statistics would expose future structure
        # (including the target edge itself) to the auxiliary prediction target.
        # Wikipedia is a directed bipartite user->page graph.  Keep endpoint
        # roles instead of forcing the relation representation to be symmetric.
        pair_state_dim = hidden_dim * 4
        self.online_relation_encoder = nn.Sequential(
            nn.Linear(pair_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_relation_encoder = deepcopy(self.online_relation_encoder)
        self.target_relation_encoder.requires_grad_(False)

        self.history_projector = nn.Sequential(
            nn.Linear(self.history_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        context_dim = hidden_dim * 8
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, predictor_hidden_dim),
            nn.LayerNorm(predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )
        self.future_predictor = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.node_predictor[-1].weight)
        nn.init.zeros_(self.node_predictor[-1].bias)
        self.node_prediction_norm = nn.LayerNorm(hidden_dim)
        self.node_dynamics_logit = nn.Parameter(torch.tensor(0.0))
        self.relation_predictor = nn.Sequential(
            nn.Linear(hidden_dim, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, hidden_dim),
        )
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, predictor_hidden_dim),
            nn.GELU(),
            nn.Linear(predictor_hidden_dim, 1),
        )
        # Start from a causal recurrence prior; the neural head learns a
        # residual for unseen and historically inactive pairs.
        nn.init.zeros_(self.intensity_head[-1].weight)
        nn.init.constant_(self.intensity_head[-1].bias, initial_link_logit_bias)
        # Directed, feature-wise recurrence prior over the eight causal history
        # statistics. Unlike a single scale, this can distinguish frequency,
        # recency, burstiness and endpoint popularity without changing the JEPA
        # representation pathway.
        self.history_prior_weights = nn.Parameter(
            torch.tensor([0.80, 0.45, 1.25, 0.20, 0.0, 0.12, 0.45, 0.45])
        )
        self.source_id_embedding = (
            nn.Embedding(int(num_nodes), id_embedding_dim)
            if id_embedding_dim > 0
            else None
        )
        self.destination_id_embedding = (
            nn.Embedding(int(num_nodes), id_embedding_dim)
            if id_embedding_dim > 0
            else None
        )
        if self.source_id_embedding is not None:
            assert self.destination_id_embedding is not None
            nn.init.normal_(self.source_id_embedding.weight, std=0.02)
            nn.init.normal_(self.destination_id_embedding.weight, std=0.02)
        # A nonzero scale lets the directed transductive embeddings receive a
        # learning signal from the first optimizer step.
        self.id_score_scale = nn.Parameter(
            torch.tensor(float(initial_id_score_scale))
        )
        # Downstream residual on top of frozen multi-hop homophily. Zero-init keeps
        # the starting point at hop5 (~paper SG level) while allowing temporal
        # JEPA features to add a supervised or SSL-driven correction.
        self.homophily_residual = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.homophily_residual.weight)
        nn.init.zeros_(self.homophily_residual.bias)
        self.homophily_residual_logit = nn.Parameter(torch.tensor(-2.0))
        # DBLP grid search favors hop2/hop4/hop5 ≈ 0.2/0.1/0.7 over pure hop5.
        # Logits are log-weights so softmax recovers the prior blend.
        self.hop_blend_logits = nn.Parameter(
            torch.tensor([-1.6094379, -2.3025851, -0.3566749])
        )

    def _snapshot_input(self, snapshot: Snapshot) -> Tensor:
        num_nodes = snapshot.x.shape[0]
        key = (snapshot.time, snapshot.edge_index.shape[1], str(snapshot.x.device))
        rwpe = self._rwpe_cache.get(key) if self.cache_rwpe else None
        if rwpe is None:
            rwpe = random_walk_positional_encoding(
                snapshot.edge_index,
                num_nodes,
                self.rwpe_dim,
                self.rwpe_walks,
                seed=self.rwpe_seed + snapshot.time,
            )
            if self.cache_rwpe:
                self._rwpe_cache[key] = rwpe.detach()
        time = sinusoidal_time_encoding(snapshot.time, self.time_dim, snapshot.x.device)
        return torch.cat([snapshot.x, rwpe, time.expand(num_nodes, -1)], dim=-1)

    def encode_snapshot(self, snapshot: Snapshot, target: bool = False) -> Tensor:
        encoder = self.target_encoder if target else self.online_encoder
        graph_innovation = encoder(self._snapshot_input(snapshot), snapshot.edge_index)
        content = self.feature_skip(snapshot.x)
        graph_scale = torch.sigmoid(self.snapshot_graph_logit)
        return F.layer_norm(content + graph_scale * graph_innovation, (self.hidden_dim,))

    @staticmethod
    def _directed_pair_state(node_u: Tensor, node_v: Tensor) -> Tensor:
        return torch.cat([node_u, node_v, node_u * node_v, node_u - node_v], dim=-1)

    @torch.no_grad()
    def prepare_causal_history(self, snapshots: Sequence[Snapshot]) -> None:
        """Index duplicate-preserving events strictly before every snapshot.

        The tensors are intentionally non-persistent: they are deterministic
        dataset state, not learned parameters or checkpoint contents.
        """
        if not self.use_causal_history:
            return
        if not snapshots:
            raise ValueError("causal history requires at least one snapshot")
        device = snapshots[0].x.device
        num_nodes = snapshots[0].x.shape[0]
        all_keys = []
        for snapshot in snapshots:
            edges = snapshot.query_edge_index
            if edges is not None and edges.numel():
                all_keys.append(edges[0].long() * num_nodes + edges[1].long())
        if not all_keys:
            raise ValueError("causal relation history requires query_edge_index")
        pair_keys = torch.unique(torch.cat(all_keys), sorted=True)
        pair_count = torch.zeros(pair_keys.numel(), device=device)
        previous_bin_count = torch.zeros_like(pair_count)
        pair_last_time = torch.full_like(pair_count, -1.0)
        pair_active_bins = torch.zeros_like(pair_count)
        node_count = torch.zeros(num_nodes, device=device)
        pair_semantic_sum = torch.zeros(
            pair_keys.numel(), self.history_semantic_dim, device=device
        )
        node_semantic_sum = torch.zeros(
            num_nodes, self.history_semantic_dim, device=device
        )
        count_rows, previous_rows, last_rows, active_rows, node_rows = [], [], [], [], []
        pair_semantic_rows, node_semantic_rows = [], []
        raw_feature_dim = next(
            (
                int(snapshot.query_features.shape[1])
                for snapshot in snapshots
                if snapshot.query_features is not None
            ),
            0,
        )
        semantic_projection = None
        if self.history_semantic_dim and raw_feature_dim:
            source_axis = torch.arange(
                1, raw_feature_dim + 1, device=device, dtype=torch.float32
            ).unsqueeze(1)
            target_axis = torch.arange(
                1, self.history_semantic_dim + 1, device=device, dtype=torch.float32
            ).unsqueeze(0)
            semantic_projection = (
                torch.cos(source_axis * target_axis)
                * (2.0 / max(1, raw_feature_dim)) ** 0.5
            )
        self._history_time_to_row = {}
        for row, snapshot in enumerate(snapshots):
            self._history_time_to_row[int(snapshot.time)] = row
            count_rows.append(pair_count.clone())
            previous_rows.append(previous_bin_count.clone())
            last_rows.append(pair_last_time.clone())
            active_rows.append(pair_active_bins.clone())
            node_rows.append(node_count.clone())
            if self.history_semantic_dim:
                pair_semantic_rows.append(
                    (pair_semantic_sum / pair_count.clamp_min(1.0).unsqueeze(1))
                    .to(torch.float16)
                )
                node_semantic_rows.append(
                    (node_semantic_sum / node_count.clamp_min(1.0).unsqueeze(1))
                    .to(torch.float16)
                )

            edges = snapshot.query_edge_index
            previous_bin_count = torch.zeros_like(pair_count)
            if edges is None or not edges.numel():
                continue
            keys = edges[0].long() * num_nodes + edges[1].long()
            positions = torch.searchsorted(pair_keys, keys)
            ones = torch.ones_like(positions, dtype=pair_count.dtype)
            previous_bin_count.scatter_add_(0, positions, ones)
            pair_count.add_(previous_bin_count)
            pair_active_bins.add_((previous_bin_count > 0).to(pair_count.dtype))
            node_count.scatter_add_(0, edges[0].long(), ones)
            node_count.scatter_add_(0, edges[1].long(), ones)
            if self.history_semantic_dim:
                if snapshot.query_features is None or semantic_projection is None:
                    projected_features = torch.zeros(
                        edges.shape[1], self.history_semantic_dim, device=device
                    )
                else:
                    raw_features = snapshot.query_features.to(
                        device=device, dtype=torch.float32
                    )
                    normalized_features = F.layer_norm(
                        raw_features, (raw_features.shape[1],)
                    )
                    projected_features = normalized_features @ semantic_projection
                pair_semantic_sum.index_add_(0, positions, projected_features)
                node_semantic_sum.index_add_(
                    0, edges[0].long(), projected_features
                )
                node_semantic_sum.index_add_(
                    0, edges[1].long(), projected_features
                )
            # RCPS remains a discrete-time model: repeated events are retained,
            # but recency is measured in snapshot indices rather than exact
            # within-snapshot timestamps.
            times = torch.full_like(pair_count[positions], float(snapshot.time))
            latest = torch.full_like(pair_count, -1.0)
            latest.scatter_reduce_(0, positions, times, reduce="amax", include_self=True)
            observed = latest >= 0
            pair_last_time[observed] = latest[observed]
        self._history_pair_keys = pair_keys
        self._history_pair_counts = torch.stack(count_rows)
        self._history_previous_bin_counts = torch.stack(previous_rows)
        self._history_pair_last_times = torch.stack(last_rows)
        self._history_pair_active_bins = torch.stack(active_rows)
        self._history_node_counts = torch.stack(node_rows)
        self._history_pair_semantics = (
            torch.stack(pair_semantic_rows) if pair_semantic_rows else None
        )
        self._history_node_semantics = (
            torch.stack(node_semantic_rows) if node_semantic_rows else None
        )

        event_sources: list[Tensor] = []
        event_destinations: list[Tensor] = []
        event_times: list[Tensor] = []
        for snapshot in snapshots:
            edges = snapshot.query_edge_index
            if edges is None or not edges.numel():
                continue
            event_sources.append(edges[0].long())
            event_destinations.append(edges[1].long())
            if snapshot.query_timestamps is None:
                event_times.append(
                    torch.full(
                        (edges.shape[1],),
                        float(snapshot.time),
                        dtype=snapshot.x.dtype,
                        device=device,
                    )
                )
            else:
                event_times.append(
                    snapshot.query_timestamps.to(device=device, dtype=snapshot.x.dtype)
                )
        sources = torch.cat(event_sources)
        destinations = torch.cat(event_destinations)
        times = torch.cat(event_times)
        chronological = torch.argsort(times, stable=True)
        times = times[chronological]
        sources = sources[chronological]
        destinations = destinations[chronological]
        event_count = int(times.numel())
        stride = event_count + 1
        event_rank = torch.arange(event_count, dtype=torch.long, device=device)
        pair_keys_exact = sources * num_nodes + destinations
        self._exact_event_times = times
        self._exact_pair_codes = torch.sort(
            pair_keys_exact * stride + event_rank
        ).values
        node_ids_exact = torch.cat([sources, destinations])
        node_ranks_exact = torch.cat([event_rank, event_rank])
        self._exact_node_codes = torch.sort(
            node_ids_exact * stride + node_ranks_exact
        ).values
        self._exact_event_stride = stride

    def _causal_history_features(
        self,
        target: Snapshot,
        pairs: Tensor,
        timestamps: Tensor | None,
    ) -> Tensor:
        if not self.use_causal_history:
            return torch.zeros(
                pairs.shape[0], self.history_feature_dim,
                dtype=target.x.dtype, device=pairs.device,
            )
        tensors = (
            self._history_pair_keys,
            self._history_pair_counts,
            self._history_previous_bin_counts,
            self._history_pair_last_times,
            self._history_pair_active_bins,
            self._history_node_counts,
        )
        if any(value is None for value in tensors):
            raise RuntimeError("call prepare_causal_history before RCPS training")
        row = self._history_time_to_row[int(target.time)]
        pair_keys = self._history_pair_keys
        assert pair_keys is not None
        keys = pairs[:, 0].long() * target.x.shape[0] + pairs[:, 1].long()
        positions = torch.searchsorted(pair_keys, keys)
        valid = positions < pair_keys.numel()
        safe = positions.clamp_max(pair_keys.numel() - 1)
        valid = valid & (pair_keys[safe] == keys)

        def pair_value(matrix: Tensor | None) -> Tensor:
            assert matrix is not None
            values = matrix[row, safe]
            return torch.where(valid, values, torch.zeros_like(values))

        count = pair_value(self._history_pair_counts)
        previous = pair_value(self._history_previous_bin_counts)
        last_time = pair_value(self._history_pair_last_times)
        active_bins = pair_value(self._history_pair_active_bins)
        node_counts = self._history_node_counts
        assert node_counts is not None
        source_count = node_counts[row, pairs[:, 0]]
        destination_count = node_counts[row, pairs[:, 1]]
        query_time = torch.full_like(count, float(target.time))
        if timestamps is not None and self._exact_event_times is not None:
            pair_codes = self._exact_pair_codes
            node_codes = self._exact_node_codes
            stride = self._exact_event_stride
            if pair_codes is None or node_codes is None or stride is None:
                raise RuntimeError("event-exact causal history index is incomplete")
            event_times = self._exact_event_times
            query_timestamps = timestamps.to(
                device=pairs.device, dtype=event_times.dtype
            )
            query_rank = torch.searchsorted(
                event_times, query_timestamps, right=False
            ).long()
            query_time = query_rank.to(count.dtype)

            pair_base = keys * stride
            pair_start = torch.searchsorted(pair_codes, pair_base, right=False)
            pair_end = torch.searchsorted(
                pair_codes, pair_base + query_rank, right=False
            )
            count = (pair_end - pair_start).to(count.dtype)
            has_pair_history = pair_end > pair_start
            previous_code_position = (pair_end - 1).clamp_min(0)
            previous_event_rank = (
                pair_codes[previous_code_position] % stride
            ).long()
            exact_last_time = previous_event_rank.to(count.dtype)
            last_time = torch.where(
                has_pair_history, exact_last_time, torch.full_like(query_time, -1.0)
            )

            def exact_node_count(node_ids: Tensor) -> Tensor:
                node_base = node_ids.long() * stride
                node_start = torch.searchsorted(node_codes, node_base, right=False)
                node_end = torch.searchsorted(
                    node_codes, node_base + query_rank, right=False
                )
                return (node_end - node_start).to(count.dtype)

            source_count = exact_node_count(pairs[:, 0])
            destination_count = exact_node_count(pairs[:, 1])
        inverse_recency = torch.where(
            count > 0,
            1.0 / (1.0 + torch.log1p((query_time - last_time).clamp_min(0.0))),
            torch.zeros_like(count),
        )
        statistics = torch.stack(
            [
                torch.log1p(count),
                torch.log1p(previous),
                inverse_recency,
                active_bins / max(1, row),
                torch.log1p(source_count),
                torch.log1p(destination_count),
                count / source_count.clamp_min(1.0),
                count / destination_count.clamp_min(1.0),
            ],
            dim=-1,
        )
        if not self.history_semantic_dim:
            return statistics
        pair_semantics = self._history_pair_semantics
        node_semantics = self._history_node_semantics
        if pair_semantics is None or node_semantics is None:
            raise RuntimeError("causal semantic history was not prepared")
        pair_profile = pair_semantics[row, safe].to(target.x.dtype)
        pair_profile = torch.where(
            valid.unsqueeze(1), pair_profile, torch.zeros_like(pair_profile)
        )
        source_profile = node_semantics[row, pairs[:, 0]].to(target.x.dtype)
        destination_profile = node_semantics[row, pairs[:, 1]].to(target.x.dtype)
        return torch.cat(
            [statistics, pair_profile, source_profile, destination_profile], dim=-1
        )

    def _predict_node_future(self, node_context: Tensor, horizon_encoding: Tensor) -> Tensor:
        innovation = self.node_predictor(
            torch.cat([node_context, horizon_encoding], dim=-1)
        )
        scale = torch.sigmoid(self.node_dynamics_logit)
        return self.node_prediction_norm(node_context + scale * innovation)

    def _content_views(self, snapshot: Snapshot) -> dict[str, Tensor]:
        """Frozen content skip plus multi-hop homophily views."""
        content = self.feature_skip(snapshot.x)
        hop1 = neighbor_mean_embeddings(snapshot, content, self.undirected)
        hop2 = neighbor_mean_embeddings(snapshot, hop1, self.undirected)
        hop4 = neighbor_mean_embeddings(snapshot, hop2, self.undirected)
        hop4 = neighbor_mean_embeddings(snapshot, hop4, self.undirected)
        hop5 = neighbor_mean_embeddings(snapshot, hop4, self.undirected)
        hop6 = neighbor_mean_embeddings(snapshot, hop5, self.undirected)
        hop8 = neighbor_mean_embeddings(snapshot, hop6, self.undirected)
        hop8 = neighbor_mean_embeddings(snapshot, hop8, self.undirected)
        return {
            "content": content,
            "hop1": hop1,
            "hop2": hop2,
            "hop4": hop4,
            "hop5": hop5,
            "hop6": hop6,
            "hop8": hop8,
        }

    def _homophily_state(self, snapshot: Snapshot) -> Tensor:
        """Default 2-hop mean of the frozen content skip."""
        return self._content_views(snapshot)["hop2"]

    def _predict_homophily_future(
        self,
        node_context: Tensor,
        horizon_encoding: Tensor,
        backbone: Tensor,
    ) -> Tensor:
        innovation = self.node_predictor(
            torch.cat([node_context, horizon_encoding], dim=-1)
        )
        return backbone + innovation

    def _pool_relation_context(
        self,
        embeddings: Tensor,
        context_nodes: Tensor,
        context_mask: Tensor,
    ) -> Tensor:
        pooled = []
        weight = context_mask.to(embeddings.dtype).unsqueeze(-1)
        denominator = weight.sum(dim=1).clamp_min(1.0)
        for snapshot_embedding in embeddings.unbind(dim=0):
            selected = snapshot_embedding[context_nodes]
            pooled.append((selected * weight).sum(dim=1) / denominator)
        sequence = torch.stack(pooled, dim=0)
        _, hidden = self.graph_gru(sequence)
        return hidden[-1]

    def prepare_window(self, window: Sequence[Snapshot]) -> RCPSPreparedWindow:
        if len(window) != self.window_size:
            raise ValueError(f"expected {self.window_size} snapshots")
        context_snapshots = list(window[:-1])
        target_snapshot = window[-1]
        context_embeddings = torch.stack(
            [self.encode_snapshot(snapshot, target=False) for snapshot in context_snapshots]
        )
        with torch.no_grad():
            target_embedding = self.encode_snapshot(target_snapshot, target=True)
        return RCPSPreparedWindow(
            context_snapshots=context_snapshots,
            target_snapshot=target_snapshot,
            context_embeddings=context_embeddings,
            target_embedding=target_embedding,
        )

    def _forward_prepared(
        self,
        prepared: RCPSPreparedWindow,
        pairs: Tensor,
        timestamps: Tensor | None = None,
    ) -> RCPSWindowOutput:
        context_snapshots = prepared.context_snapshots
        target_snapshot = prepared.target_snapshot
        context_embeddings = prepared.context_embeddings
        target_embedding = prepared.target_embedding

        context_nodes, context_mask = relation_context_nodes(
            context_snapshots,
            pairs,
            self.subgraph_budget,
            self.path_decay,
            self.bridge_weight,
            self.undirected,
        )
        global_node_context, _ = self._node_context(prepared)
        node_u_context = global_node_context[pairs[:, 0]]
        node_v_context = global_node_context[pairs[:, 1]]
        graph_context = self._pool_relation_context(context_embeddings, context_nodes, context_mask)

        raw_increments = temporal_pair_increments(
            context_snapshots, pairs, context_nodes, self.undirected
        )
        projected_increments = self.event_projector(raw_increments)
        signature = truncated_signature(projected_increments, self.signature_depth)
        signature_context = self.signature_projector(signature)

        endpoint_state = self._directed_pair_state(node_u_context, node_v_context)
        relation_context = self.online_relation_encoder(endpoint_state)
        history_features = self._causal_history_features(
            target_snapshot, pairs, timestamps
        )
        history_context = self.history_projector(history_features)
        context = self.context_encoder(
            torch.cat(
                [
                    endpoint_state,
                    graph_context,
                    signature_context,
                    relation_context,
                    history_context,
                ],
                dim=-1,
            )
        )

        horizon = max(1, target_snapshot.time - context_snapshots[-1].time)
        horizon_encoding = sinusoidal_time_encoding(horizon, self.time_dim, pairs.device)
        future = self.future_predictor(
            torch.cat([context, horizon_encoding.expand(pairs.shape[0], -1)], dim=-1)
        )
        expanded_horizon = horizon_encoding.expand(pairs.shape[0], -1)
        node_u_prediction = self._predict_node_future(node_u_context, expanded_horizon)
        node_v_prediction = self._predict_node_future(node_v_context, expanded_horizon)
        relation_prediction = self.relation_predictor(future)

        node_u_target = target_embedding[pairs[:, 0]].detach()
        node_v_target = target_embedding[pairs[:, 1]].detach()
        target_endpoint_state = self._directed_pair_state(node_u_target, node_v_target)
        with torch.no_grad():
            relation_target = self.target_relation_encoder(target_endpoint_state)

        intensity_input = torch.cat(
            [relation_context, history_context, context], dim=-1
        )
        history_prior = (
            history_features[:, :8]
            * self.history_prior_weights.to(history_features.dtype)
        ).sum(dim=-1)
        logit = self.intensity_head(intensity_input).squeeze(-1)
        logit = logit + history_prior
        if self.source_id_embedding is not None:
            assert self.destination_id_embedding is not None
            source_id = F.normalize(
                self.source_id_embedding(pairs[:, 0]), dim=-1
            )
            destination_id = F.normalize(
                self.destination_id_embedding(pairs[:, 1]), dim=-1
            )
            source_id = F.dropout(
                source_id, p=self.id_embedding_dropout, training=self.training
            )
            destination_id = F.dropout(
                destination_id, p=self.id_embedding_dropout, training=self.training
            )
            id_score = (source_id * destination_id).sum(dim=-1)
            logit = logit + self.id_score_scale * id_score
        intensity = F.softplus(logit)
        probability = -torch.expm1(-intensity * float(horizon))
        return RCPSWindowOutput(
            node_u_prediction=node_u_prediction,
            node_v_prediction=node_v_prediction,
            relation_prediction=relation_prediction,
            node_u_target=node_u_target,
            node_v_target=node_v_target,
            relation_target=relation_target,
            logit=logit,
            intensity=intensity,
            probability=probability,
            context=context,
        )

    def forward_pairs(
        self,
        window: Sequence[Snapshot],
        pairs: Tensor,
        timestamps: Tensor | None = None,
    ) -> RCPSWindowOutput:
        return self._forward_prepared(
            self.prepare_window(window), pairs, timestamps=timestamps
        )

    def sample_queries(
        self,
        window: Sequence[Snapshot],
        seed: int,
        max_positive: int | None | object = _QUERY_CAP_UNSET,
        negative_ratio: float | None = None,
    ) -> LinkQueries:
        if max_positive is _QUERY_CAP_UNSET:
            max_positive = self.max_positive_pairs
        return sample_link_queries(
            window[-1],
            window[-2],
            negative_ratio=(
                self.negative_ratio
                if negative_ratio is None
                else float(negative_ratio)
            ),
            max_positive=None if max_positive is None else int(max_positive),
            seed=seed,
            new_edges_only=self.new_edges_only,
            undirected=self.undirected,
            bipartite_source_count=self.bipartite_source_count,
            negative_destination_candidates=self.negative_destination_candidates,
            allow_negative_collisions=self.allow_negative_collisions,
        )

    def _node_predictions(
        self, prepared: RCPSPreparedWindow
    ) -> RCPSNodeOutput:
        node_context, mean_gate = self._node_context(prepared)
        horizon = max(
            1,
            prepared.target_snapshot.time - prepared.context_snapshots[-1].time,
        )
        horizon_encoding = sinusoidal_time_encoding(
            horizon, self.time_dim, node_context.device
        ).expand(node_context.shape[0], -1)
        backbone = self._homophily_state(prepared.context_snapshots[-1])
        node_prediction = self._predict_homophily_future(
            node_context, horizon_encoding, backbone
        )
        # The sole future target is the EMA node latent.  We intentionally avoid
        # explicit future-neighborhood, added-neighbor, and structure targets.
        latent_target = prepared.target_embedding.detach()
        node_ids = prepared.target_snapshot.active.nonzero(as_tuple=False).flatten()
        if node_ids.numel() == 0:
            raise ValueError("target snapshot has no active nodes")
        return RCPSNodeOutput(
            prediction=node_prediction[node_ids],
            target=latent_target[node_ids],
            node_ids=node_ids,
            mean_relation_gate=mean_gate,
            representation=backbone[node_ids],
        )

    def _node_context(
        self, prepared: RCPSPreparedWindow
    ) -> tuple[Tensor, Tensor]:
        """Fuse individual history with topology-conditioned relation dynamics."""
        if prepared.node_context is not None:
            if prepared.mean_relation_gate is None:
                raise RuntimeError("cached node context is missing its gate statistic")
            gate = prepared.mean_relation_gate
            return prepared.node_context, gate

        _, individual_hidden = self.node_gru(prepared.context_embeddings)
        history_scale = torch.sigmoid(self.node_history_logit)
        homophily_scale = torch.sigmoid(self.node_homophily_logit)
        views = self._content_views(prepared.context_snapshots[-1])
        hop1, hop2 = views["hop1"], views["hop2"]
        individual_context = self.node_history_norm(
            prepared.context_embeddings[-1]
            + history_scale * individual_hidden[-1]
            + homophily_scale * 0.5 * (hop1 + hop2)
        )

        neighbor_sequence = torch.stack(
            [
                neighbor_mean_embeddings(snapshot, embedding, self.undirected)
                for snapshot, embedding in zip(
                    prepared.context_snapshots, prepared.context_embeddings.unbind(dim=0)
                )
            ],
            dim=0,
        )
        _, relation_hidden = self.node_relation_gru(neighbor_sequence)
        relation_context = relation_hidden[-1]

        raw_increments = temporal_node_increments(
            prepared.context_snapshots, self.undirected
        )
        projected_increments = self.node_event_projector(raw_increments)
        node_signature = truncated_signature(projected_increments, self.signature_depth)
        signature_context = self.node_signature_projector(node_signature)

        fusion_input = torch.cat(
            [individual_context, relation_context, signature_context], dim=-1
        )
        relation_update = self.node_context_encoder(fusion_input)
        relation_gate = torch.sigmoid(self.node_context_gate(fusion_input))
        node_context = self.node_context_norm(
            individual_context + relation_gate * relation_update
        )

        prepared.node_context = node_context
        prepared.mean_relation_gate = relation_gate.detach().mean()
        return node_context, relation_gate.detach().mean()

    def node_loss_windows(
        self,
        windows: Sequence[Sequence[Snapshot]],
        node_batch_size: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        losses = []
        node_losses = []
        contrastive_losses = []
        variance_losses = []
        covariance_losses = []
        relation_gates = []
        for window in windows:
            output = self._node_predictions(self.prepare_window(window))
            order = torch.randperm(output.prediction.shape[0], device=output.prediction.device)
            size = output.prediction.shape[0] if node_batch_size is None else node_batch_size
            for start in range(0, output.prediction.shape[0], size):
                index = order[start : start + size]
                prediction = output.prediction[index]
                target = output.target[index]
                node_loss = _normalized_distance(prediction, target)
                contrastive_loss = _contrastive_prediction_loss(
                    prediction, target, self.contrastive_temperature
                )
                variance_loss, covariance_loss = _variance_covariance_loss(
                    prediction, self.variance_target
                )
                loss = (
                    self.node_loss_weight * node_loss
                    + self.node_contrastive_loss_weight * contrastive_loss
                    + self.variance_loss_weight * variance_loss
                    + self.covariance_loss_weight * covariance_loss
                )
                losses.append(loss)
                node_losses.append(node_loss.detach())
                contrastive_losses.append(contrastive_loss.detach())
                variance_losses.append(variance_loss.detach())
                covariance_losses.append(covariance_loss.detach())
            relation_gates.append(output.mean_relation_gate)
        if not losses:
            raise ValueError("no temporal windows were provided")
        loss = torch.stack(losses).mean()
        return loss, {
            "loss": float(loss.detach().item()),
            "node_loss": float(torch.stack(node_losses).mean().item()),
            "contrastive_loss": float(torch.stack(contrastive_losses).mean().item()),
            "variance_loss": float(torch.stack(variance_losses).mean().item()),
            "covariance_loss": float(torch.stack(covariance_losses).mean().item()),
            "relation_gate": float(torch.stack(relation_gates).mean().item()),
            "graph_scale": float(torch.sigmoid(self.snapshot_graph_logit).detach().item()),
            "history_scale": float(torch.sigmoid(self.node_history_logit).detach().item()),
            "homophily_scale": float(torch.sigmoid(self.node_homophily_logit).detach().item()),
            "dynamics_scale": float(torch.sigmoid(self.node_dynamics_logit).detach().item()),
        }

    def blend_content_hops(self, views: dict[str, Tensor]) -> Tensor:
        weights = F.softmax(self.hop_blend_logits, dim=0)
        return (
            weights[0] * views["hop2"]
            + weights[1] * views["hop4"]
            + weights[2] * views["hop5"]
        )

    def _encoder_multihop(self, snapshot: Snapshot, hops: int) -> Tensor:
        state = self.encode_snapshot(snapshot, target=False)
        for _ in range(hops):
            state = neighbor_mean_embeddings(snapshot, state, self.undirected)
        return state

    def encode_temporal_state(self, snapshots: Sequence[Snapshot]) -> Tensor:
        """Encode a contiguous snapshot sequence ending at the readout time.

        Unlike the JEPA context branch, the final snapshot is included so the
        downstream representation can use the complete cumulative graph together
        with short-term relational dynamics.
        """
        if len(snapshots) < 1:
            raise ValueError("at least one snapshot is required")
        embeddings = torch.stack(
            [self.encode_snapshot(snapshot, target=False) for snapshot in snapshots]
        )
        _, individual_hidden = self.node_gru(embeddings)
        history_scale = torch.sigmoid(self.node_history_logit)
        homophily_scale = torch.sigmoid(self.node_homophily_logit)
        views = self._content_views(snapshots[-1])
        individual_context = self.node_history_norm(
            embeddings[-1]
            + history_scale * individual_hidden[-1]
            + homophily_scale * 0.5 * (views["hop1"] + views["hop2"])
        )
        neighbor_sequence = torch.stack(
            [
                neighbor_mean_embeddings(snapshot, embedding, self.undirected)
                for snapshot, embedding in zip(snapshots, embeddings.unbind(dim=0))
            ],
            dim=0,
        )
        _, relation_hidden = self.node_relation_gru(neighbor_sequence)
        relation_context = relation_hidden[-1]
        raw_increments = temporal_node_increments(snapshots, self.undirected)
        projected_increments = self.node_event_projector(raw_increments)
        node_signature = truncated_signature(projected_increments, self.signature_depth)
        signature_context = self.node_signature_projector(node_signature)
        fusion_input = torch.cat(
            [individual_context, relation_context, signature_context], dim=-1
        )
        relation_update = self.node_context_encoder(fusion_input)
        relation_gate = torch.sigmoid(self.node_context_gate(fusion_input))
        return self.node_context_norm(individual_context + relation_gate * relation_update)

    def fuse_homophily(self, hop: Tensor, temporal: Tensor) -> Tensor:
        scale = torch.sigmoid(self.homophily_residual_logit)
        return hop + scale * self.homophily_residual(temporal)

    def _raw_multihop(self, snapshot: Snapshot, hops: int = 5) -> Tensor:
        state = snapshot.x
        for _ in range(hops):
            state = neighbor_mean_embeddings(snapshot, state, self.undirected)
        return state

    def infer_node_views(
        self, window: Sequence[Snapshot], *, grad: bool = False
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Return complementary readouts from the final snapshot in ``window``.

        Self-supervised training still predicts the target from earlier context.
        Downstream node classification on DBLP-style benchmarks may use the
        final snapshot, matching SpikeNet / SG-JEPA table protocol practice.

        Set ``grad=True`` during supervised fine-tuning so temporal and encoder
        pathways can adapt the fused readout.
        """
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            prepared = self.prepare_window(window)
            output = self._node_predictions(prepared)
            last = prepared.target_snapshot
            views = self._content_views(last)
            encoder = self.encode_snapshot(last, target=False)
            temporal = self.encode_temporal_state(window)
            blend = self.blend_content_hops(views)
            fused = self.fuse_homophily(blend, temporal)
            node_ids = output.node_ids
            return (
                {
                    "content": views["content"][node_ids],
                    "hop1": views["hop1"][node_ids],
                    "hop2": views["hop2"][node_ids],
                    "hop4": views["hop4"][node_ids],
                    "hop5": views["hop5"][node_ids],
                    "hop6": views["hop6"][node_ids],
                    "hop8": views["hop8"][node_ids],
                    "blend": blend[node_ids],
                    "raw_hop5": self._raw_multihop(last, 5)[node_ids],
                    "encoder": encoder[node_ids],
                    "enc_hop4": self._encoder_multihop(last, 4)[node_ids],
                    "enc_hop5": self._encoder_multihop(last, 5)[node_ids],
                    "temporal": temporal[node_ids],
                    "fused": fused[node_ids],
                    "prediction": output.prediction,
                },
                node_ids,
            )

    def infer_nodes(self, window: Sequence[Snapshot]) -> tuple[Tensor, Tensor]:
        views, node_ids = self.infer_node_views(window)
        return views["fused"], node_ids

    def _pair_batch_loss(
        self,
        prepared: RCPSPreparedWindow,
        pairs: Tensor,
        labels: Tensor,
        group_ids: Tensor,
        timestamps: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        output = self._forward_prepared(prepared, pairs, timestamps=timestamps)
        positive = labels > 0.5
        node_loss = 0.5 * (
            _masked_distance(output.node_u_prediction, output.node_u_target, positive)
            + _masked_distance(output.node_v_prediction, output.node_v_target, positive)
        )
        relation_loss = _masked_distance(
            output.relation_prediction, output.relation_target, positive
        )
        per_sample = F.binary_cross_entropy(
            output.probability.clamp(1e-6, 1 - 1e-6), labels, reduction="none"
        )
        positive_count = positive.sum().clamp_min(1)
        batch_negative_ratio = (~positive).sum().to(per_sample.dtype) / positive_count
        sample_weight = torch.where(
            positive,
            batch_negative_ratio,
            per_sample.new_tensor(1.0),
        )
        link_loss = (per_sample * sample_weight).mean()
        rank_loss = _groupwise_ranking_loss(output.logit, labels, group_ids)
        if int(positive.sum().item()) >= 2:
            variance_loss, covariance_loss = _variance_covariance_loss(
                output.relation_prediction[positive], self.variance_target
            )
        else:
            variance_loss = output.probability.sum() * 0.0
            covariance_loss = variance_loss
        loss = (
            self.node_loss_weight * node_loss
            + self.relation_loss_weight * relation_loss
            + self.link_loss_weight * link_loss
            + self.rank_loss_weight * rank_loss
            + self.variance_loss_weight * variance_loss
            + self.covariance_loss_weight * covariance_loss
        )
        values = {
            "node_loss": node_loss,
            "relation_loss": relation_loss,
            "link_loss": link_loss,
            "rank_loss": rank_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            "mean_probability": output.probability.mean(),
        }
        return loss, values

    def train_epoch(
        self,
        windows: Sequence[Sequence[Snapshot]],
        optimizer: torch.optim.Optimizer,
        grad_clip: float,
        seed: int = 42,
        pair_batch_size: int | None = None,
    ) -> dict[str, float]:
        """Run one data pass with an optimizer step after every pair batch."""
        if not windows:
            raise ValueError("no temporal windows were provided")
        self.train()
        metric_sums = {
            "node_loss": 0.0,
            "relation_loss": 0.0,
            "link_loss": 0.0,
            "rank_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
            "mean_probability": 0.0,
            "loss": 0.0,
        }
        steps = 0
        window_order = list(range(len(windows)))
        if self.shuffle_windows:
            generator = torch.Generator().manual_seed(seed)
            window_order = torch.randperm(len(windows), generator=generator).tolist()
        for window_index in window_order:
            window = windows[window_index]
            queries = self.sample_queries(
                window,
                seed + window_index,
                max_positive=self.train_max_positive_pairs,
                negative_ratio=self.train_negative_ratio,
            )
            for rows in _query_group_batches(queries, pair_batch_size):
                optimizer.zero_grad(set_to_none=True)
                prepared = self.prepare_window(window)
                timestamps = (
                    None if queries.timestamps is None else queries.timestamps[rows]
                )
                loss, values = self._pair_batch_loss(
                    prepared,
                    queries.pairs[rows],
                    queries.labels[rows],
                    queries.group_ids[rows],
                    timestamps,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"RCPS-JEPA produced a non-finite loss at train batch {steps}"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip))
                optimizer.step()
                if self.ema_update_per_step:
                    momentum = self.ema_momentum
                    self.update_target_encoder(momentum=momentum)
                elif self.ema_steps_per_epoch is not None:
                    momentum = self.ema_momentum ** (1.0 / self.ema_steps_per_epoch)
                    self.update_target_encoder(momentum=momentum)
                steps += 1
                metric_sums["loss"] += float(loss.detach().item())
                for name, value in values.items():
                    metric_sums[name] += float(value.detach().item())
                prepared.clear_node_cache()
                del prepared, loss, values
        if steps == 0:
            raise ValueError("no temporal pair batches were produced")
        metrics = {name: value / steps for name, value in metric_sums.items()}
        metrics["steps"] = float(steps)
        metrics["history_count_weight"] = float(
            self.history_prior_weights[0].detach().item()
        )
        metrics["history_recency_weight"] = float(
            self.history_prior_weights[2].detach().item()
        )
        metrics["id_score_scale"] = float(self.id_score_scale.detach().item())
        metrics["train_negative_ratio"] = float(self.train_negative_ratio)
        return metrics

    def loss_windows(
        self,
        windows: Sequence[Sequence[Snapshot]],
        pair_batch_size: int | None = None,
        query_seed: int = 42,
        *,
        backward: bool = False,
    ) -> tuple[Tensor, dict[str, float]]:
        if not windows:
            raise ValueError("no temporal windows were provided")

        # Count batches without retaining model activations.  This preserves the
        # original objective (an equally weighted mean over pair batches) while
        # allowing the backward path below to release one window at a time.
        total_batches = 0
        if backward:
            for window_index, window in enumerate(windows):
                queries = self.sample_queries(
                    window,
                    query_seed + window_index,
                    negative_ratio=self.train_negative_ratio,
                )
                total_batches += len(_query_group_batches(queries, pair_batch_size))

        metric_sums = {
            "node_loss": 0.0,
            "relation_loss": 0.0,
            "link_loss": 0.0,
            "rank_loss": 0.0,
            "variance_loss": 0.0,
            "covariance_loss": 0.0,
            "mean_probability": 0.0,
        }
        detached_losses: list[float] = []
        batch_entries: list[tuple[Tensor, dict[str, Tensor]]] = []
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(
                window,
                query_seed + window_index,
                negative_ratio=self.train_negative_ratio,
            )
            prepared = self.prepare_window(window)
            query_batches = _query_group_batches(queries, pair_batch_size)
            for batch_index, rows in enumerate(query_batches):
                timestamps = (
                    None if queries.timestamps is None else queries.timestamps[rows]
                )
                loss, values = self._pair_batch_loss(
                    prepared,
                    queries.pairs[rows],
                    queries.labels[rows],
                    queries.group_ids[rows],
                    timestamps,
                )
                if backward:
                    # Pair batches in the same window share the snapshot and
                    # node-context graph. Retain it only until that window's
                    # final pair batch; previous windows are then releasable.
                    is_last_pair_batch = batch_index + 1 == len(query_batches)
                    (loss / total_batches).backward(
                        retain_graph=not is_last_pair_batch
                    )
                    detached_losses.append(float(loss.detach().item()))
                    for name, value in values.items():
                        metric_sums[name] += float(value.detach().item())
                    del loss, values
                else:
                    batch_entries.append((loss, values))

        if backward:
            total = sum(detached_losses) / total_batches
            return torch.as_tensor(total, device=next(self.parameters()).device), {
                name: value / total_batches for name, value in metric_sums.items()
            } | {"loss": total}

        if not batch_entries:
            raise ValueError("no temporal pair batches were produced")
        batches = len(batch_entries)
        total = torch.stack([loss for loss, _ in batch_entries]).mean()
        for _, values in batch_entries:
            for name, value in values.items():
                metric_sums[name] += float(value.detach().item())
        metrics = {name: value / batches for name, value in metric_sums.items()}
        metrics["loss"] = float(total.detach().item())
        return total, metrics

    @torch.no_grad()
    def update_target_encoder(self, momentum: float | None = None) -> None:
        momentum = self.ema_momentum if momentum is None else float(momentum)
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)
        for online, target in zip(
            self.online_relation_encoder.parameters(), self.target_relation_encoder.parameters()
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def evaluate_windows(
        self,
        windows: Sequence[Sequence[Snapshot]],
        pair_batch_size: int | None = None,
        query_seed: int = 42,
    ) -> dict[str, float]:
        probabilities, labels, groups = [], [], []
        group_offset = 0
        for window_index, window in enumerate(windows):
            queries = self.sample_queries(window, query_seed + window_index)
            prepared = self.prepare_window(window)
            size = queries.pairs.shape[0] if pair_batch_size is None else pair_batch_size
            for start in range(0, queries.pairs.shape[0], size):
                pairs = queries.pairs[start : start + size]
                timestamps = (
                    None
                    if queries.timestamps is None
                    else queries.timestamps[start : start + size]
                )
                output = self._forward_prepared(
                    prepared, pairs, timestamps=timestamps
                )
                probabilities.append(output.probability)
                labels.append(queries.labels[start : start + size])
                groups.append(queries.group_ids[start : start + size] + group_offset)
            group_offset += int(queries.group_ids.max().item()) + 1
        probability = torch.cat(probabilities)
        target = torch.cat(labels)
        group_ids = torch.cat(groups)
        return link_prediction_metrics(
            target,
            probability,
            group_ids,
            positive_batch_size=self.eval_positive_batch_size,
        )
