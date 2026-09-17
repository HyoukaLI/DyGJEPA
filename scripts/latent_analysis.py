"""Latent-space analysis of DyGJEPA: does the JEPA branch predict the future?

Trains DyGJEPA on one dataset with the shared link protocol (same recipe as
``compare_link_prediction``), once with the full objective and once with the
four JEPA losses switched off (``no_jepa``: node / relation / variance /
covariance weights = 0, everything else identical), then evaluates both on the
test windows with K candidates per source (1 true destination + K-1 random
destinations, the usual DyGLib corruption) and produces:

* ``agreement_heatmap.{png,pdf}``  -- for one source u at one test bin, the KxK
  cosine matrix between the context-predicted relation latents q_hat(u, v_i)
  (columns) and the stop-gradient future latents q_bar(u, v_j) computed by the
  target encoder on the target snapshot (rows); one panel per variant.  A sharp
  diagonal means the predictor anticipates each candidate's future relation
  state; the true destination is marked with a star.
* ``node_future.{png,pdf}``        -- for the same u: cosine of the predicted
  future node latent h_hat_u (and, as a control, the current state h_{t-1,u})
  with the future latents h_bar_v of the same candidates.
* ``latent_pca.{png,pdf}``         -- 2-D PCA of q_hat over the test set,
  coloured by whether the pair really forms; per variant.
* ``singular_values.{png,pdf}``    -- normalised singular-value spectrum of
  q_hat per variant with the effective rank (collapse check).
* ``metrics.json`` / ``metrics.md`` -- aggregate numbers over ALL test groups so
  the case study is not cherry-picked: top-1 retrieval of the correct future
  latent among the K candidates, diagonal-vs-off-diagonal margin, AUC/AP of the
  JEPA prediction error -d(q_hat, q_bar) used alone as a link score, effective
  rank, and the model's own AP/AUC for reference.

Comparison with a contrastive dynamic-graph method (``--baseline cldg``, the
default; ``maskdgnn`` / ``dvgmae`` use the same hooks, ``none`` disables it).
CLDG is trained with the repository recipe (``_train_snapshot_ssl_one``: SSL
pretraining on the training prefix, frozen encoder, shared link probe).  Its
InfoNCE objective pulls the *same node* together across temporal views, i.e.
it learns temporal invariance, whereas the JEPA target space has to keep what
changes so that the predictor has something to predict.  Node-level products:

* ``change_tracking.{png,pdf}`` -- left: relative latent displacement of a node
  between the last context bin and the target bin, 1-cos(z_{t-1,u}, z_{t,u})
  divided by the method's median inter-node distance at bin t, binned by the
  node's actual neighbourhood turnover 1-Jaccard(N_{t-1}(u), N_t(u)); one line
  per method (JEPA target encoder, w/o JEPA losses, CLDG encoder) with the
  Spearman correlation.  Right: JEPA only -- cosine of the predicted future
  node latent h_hat_u with the actual future h_bar_u versus the persistence
  control cos(h_bar_{t-1,u}, h_bar_{t,u}), per turnover bin.
* ``trajectory.{png,pdf}``      -- one high-turnover node followed through the
  test bins in each method's own PCA (fit on all involved nodes at the last
  test bin, coordinates in units of that cloud's std); the JEPA panel adds the
  predicted next state h_hat_t as hollow markers.
* ``metrics.md`` gains the Spearman correlations, displacement per turnover
  bin, JEPA prediction gain, the baseline's test AP/AUC under the 1:1 protocol,
  and both models' AP on the SAME K-candidate queries split into positives that
  repeat an earlier edge and positives that are new.

Everything comes from ``RCPSJEPA._forward_prepared`` / ``_node_predictions`` /
``encode_snapshot`` and the baselines' ``encode_context`` / ``probe``; no model
code is touched.  Trained weights are cached in the output directory
(``state_<variant>.pt``, ``state_<baseline>.pt``) so the figures can be
regenerated with ``--reuse`` without retraining.

Example (Mac, from the repo root):

    .venv/bin/python -m scripts.latent_analysis --dataset uci --seed 42 \
        --out results/latent/uci

    .venv/bin/python -m scripts.latent_analysis --dataset canparl --seed 42 \
        --epochs 30 --out results/latent/canparl        # shorter training
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from jepa_compare.compare_link_prediction import (
    _build_graph,
    _dataset_configs,
    _negative_destination_pool,
    _train_one,
    _train_snapshot_ssl_one,
    load_config,
)
from jepa_compare.link_prediction import (
    binary_average_precision,
    binary_roc_auc,
    temporal_window_split,
)
from jepa_compare.rcps_jepa import RCPSJEPA
from jepa_compare.snapshot_ssl_baselines import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
)
from jepa_compare.train_sg_jepa import choose_device, device_description


VARIANTS = ("full", "no_jepa")
NO_JEPA_OVERRIDES = {
    "node_loss_weight": 0.0,
    "relation_loss_weight": 0.0,
    "variance_loss_weight": 0.0,
    "covariance_loss_weight": 0.0,
}
VARIANT_TITLES = {
    "full": "DyGJEPA",
    "no_jepa": "DyGJEPA w/o JEPA losses",
    "cldg": "CLDG (contrastive)",
    "maskdgnn": "MaskDGNN (masked reconstruction)",
    "dvgmae": "DVGMAE (masked reconstruction)",
}
BASELINE_CLASSES = {
    "cldg": CLDGLinkBaseline,
    "maskdgnn": MaskDGNNLinkBaseline,
    "dvgmae": DVGMAELinkBaseline,
}
# Fixed categorical assignment (never cycled): method -> colour / marker.
METHOD_STYLE = {
    "full": ("#1f77b4", "o"),
    "no_jepa": ("#ff7f0e", "s"),
    "cldg": ("#2ca02c", "^"),
    "maskdgnn": ("#9467bd", "D"),
    "dvgmae": ("#8c564b", "v"),
}
# Neighbourhood-turnover bins: "no change" and "complete change" get their own
# bin because both are frequent in event-binned graphs.
TURNOVER_EDGES = (0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9, 1.0 + 1e-9)
TURNOVER_LABELS = ("0", "(0, .25]", "(.25, .5]", "(.5, .75]", "(.75, 1)", "1")


# --------------------------------------------------------------------------- #
# configuration / training
# --------------------------------------------------------------------------- #
def dataset_config(config_path: Path, dataset: str) -> dict:
    config = load_config(config_path)
    expanded = dict(_dataset_configs(config))
    if dataset not in expanded:
        raise SystemExit(f"dataset {dataset!r} not in {sorted(expanded)}")
    cfg = expanded[dataset]
    cfg.setdefault("wandb", {})["enabled"] = False
    return cfg


def build_protocol(cfg: dict, seed: int, device: torch.device):
    graph = _build_graph(cfg, seed).to(device)
    common = dict(cfg["common_model"])
    split_cfg = cfg.get("split", {})
    split = temporal_window_split(
        graph.snapshots,
        int(common["window_size"]),
        float(split_cfg.get("train_ratio", 0.6)),
        float(split_cfg.get("validation_ratio", 0.2)),
    )
    link_cfg = dict(cfg.get("link", {}))
    link_cfg.pop("negative_strategy", None)  # random negatives, as in training
    link_cfg["negative_destination_candidates"] = _negative_destination_pool(graph)
    link_cfg["bipartite_source_count"] = graph.num_source_nodes
    rcps_args = {
        "num_nodes": graph.num_nodes,
        **common,
        **link_cfg,
        **dict(cfg.get("rcps_jepa", {})),
    }
    rcps_training = {**dict(cfg.get("training", {})), **dict(cfg.get("rcps_training", {}))}
    return graph, split, link_cfg, rcps_args, rcps_training


def train_baseline(
    name: str,
    graph,
    split,
    cfg: dict,
    link_cfg: dict,
    seed: int,
    device: torch.device,
    out_dir: Path,
    reuse: bool,
    pretrain_epochs: int | None = None,
):
    """Snapshot-SSL baseline exactly as the comparison driver trains it."""
    state_path = out_dir / f"state_{name}.pt"
    torch.manual_seed(seed)
    model = BASELINE_CLASSES[name](
        feature_dim=graph.feature_dim, **dict(cfg.get(name, {})), **link_cfg
    ).to(device)
    if reuse and state_path.exists():
        payload = torch.load(state_path, map_location=device)
        model.freeze_encoder()
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[{name}] reused {state_path}", flush=True)
        return model, payload["metrics"]
    training = {
        **dict(cfg.get("snapshot_ssl_training", {})),
        **dict(cfg.get(f"{name}_training", {})),
    }
    if pretrain_epochs is not None:
        training["pretrain_epochs"] = int(pretrain_epochs)
    print(f"[{name}] SSL pretraining + frozen probe on {device_description(device)} ...", flush=True)
    validation, test = _train_snapshot_ssl_one(name, model, split, training, seed)
    metrics = {"validation": validation, "test": test}
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "variant": name}, state_path)
    print(f"[{name}] test AP {test['ap']:.4f} AUC {test['auc']:.4f}", flush=True)
    model.eval()
    return model, metrics


def variant_args(rcps_args: dict, variant: str) -> dict:
    args = dict(rcps_args)
    if variant == "no_jepa":
        args.update(NO_JEPA_OVERRIDES)
    elif variant != "full":
        raise ValueError(f"unknown variant {variant}")
    return args


def train_variant(
    variant: str,
    graph,
    split,
    rcps_args: dict,
    rcps_training: dict,
    seed: int,
    device: torch.device,
    out_dir: Path,
    reuse: bool,
) -> tuple[RCPSJEPA, dict]:
    state_path = out_dir / f"state_{variant}.pt"
    torch.manual_seed(seed)
    model = RCPSJEPA(feature_dim=graph.feature_dim, **variant_args(rcps_args, variant)).to(device)
    model.prepare_causal_history(graph.snapshots)
    if reuse and state_path.exists():
        payload = torch.load(state_path, map_location=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        print(f"[{variant}] reused {state_path}", flush=True)
        return model, payload["metrics"]
    print(f"[{variant}] training on {device_description(device)} ...", flush=True)
    validation, test = _train_one("rcps_jepa", model, split, rcps_training, seed)
    metrics = {"validation": validation, "test": test}
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "variant": variant}, state_path)
    print(f"[{variant}] test AP {test['ap']:.4f} AUC {test['auc']:.4f} (best epoch {test.get('best_epoch')})", flush=True)
    model.eval()
    return model, metrics


# --------------------------------------------------------------------------- #
# latent collection
# --------------------------------------------------------------------------- #
class LatentRecords:
    """Per-query latents for the test windows, grouped by (window, group)."""

    def __init__(self) -> None:
        self.pairs: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.group_keys: list[np.ndarray] = []  # global group id
        self.window_index: list[np.ndarray] = []
        self.target_time: list[np.ndarray] = []
        self.q_hat: list[torch.Tensor] = []
        self.q_bar: list[torch.Tensor] = []
        self.h_hat_u: list[torch.Tensor] = []
        self.h_bar_v: list[torch.Tensor] = []
        self.h_now_u: list[torch.Tensor] = []
        self.probability: list[np.ndarray] = []

    def finish(self) -> None:
        self.pairs = np.concatenate(self.pairs)
        self.labels = np.concatenate(self.labels)
        self.group_keys = np.concatenate(self.group_keys)
        self.window_index = np.concatenate(self.window_index)
        self.target_time = np.concatenate(self.target_time)
        self.probability = np.concatenate(self.probability)
        self.q_hat = torch.cat(self.q_hat)
        self.q_bar = torch.cat(self.q_bar)
        self.h_hat_u = torch.cat(self.h_hat_u)
        self.h_bar_v = torch.cat(self.h_bar_v)
        self.h_now_u = torch.cat(self.h_now_u)


@torch.no_grad()
def collect_latents(
    model: RCPSJEPA,
    windows: Sequence[Sequence],
    *,
    candidates: int,
    query_seed: int,
    max_groups: int,
    chunk: int,
) -> LatentRecords:
    model.eval()
    records = LatentRecords()
    group_offset = 0
    for window_index, window in enumerate(windows):
        if group_offset >= max_groups:
            break
        queries = model.sample_queries(window, query_seed + window_index, negative_ratio=candidates - 1)
        prepared = model.prepare_window(window)
        current = prepared.context_embeddings[-1]  # h_{t-1} of every node
        pairs = queries.pairs
        timestamps = queries.timestamps
        for start in range(0, pairs.shape[0], chunk):
            rows = slice(start, start + chunk)
            output = model._forward_prepared(
                prepared, pairs[rows], None if timestamps is None else timestamps[rows]
            )
            records.q_hat.append(output.relation_prediction.detach().float().cpu())
            records.q_bar.append(output.relation_target.detach().float().cpu())
            records.h_hat_u.append(output.node_u_prediction.detach().float().cpu())
            records.h_bar_v.append(output.node_v_target.detach().float().cpu())
            records.h_now_u.append(current[pairs[rows, 0]].detach().float().cpu())
            records.probability.append(output.probability.detach().float().cpu().numpy())
        records.pairs.append(pairs.detach().cpu().numpy())
        records.labels.append(queries.labels.detach().cpu().numpy())
        records.group_keys.append(queries.group_ids.detach().cpu().numpy() + group_offset)
        records.window_index.append(np.full(pairs.shape[0], window_index, dtype=np.int64))
        records.target_time.append(np.full(pairs.shape[0], int(window[-1].time), dtype=np.int64))
        group_offset += int(queries.group_ids.max().item()) + 1
    records.finish()
    return records


@torch.no_grad()
def baseline_probabilities(model, windows: Sequence[Sequence], records: LatentRecords, device: torch.device, chunk: int) -> np.ndarray:
    """Score the SAME K-candidate queries with the baseline's frozen encoder + probe."""
    model.eval()
    out: list[np.ndarray] = []
    for window_index, window in enumerate(windows):
        rows = np.flatnonzero(records.window_index == window_index)
        if rows.size == 0:
            continue
        embeddings = model.encode_context(window[:-1])
        pairs = torch.as_tensor(records.pairs[rows], dtype=torch.long, device=device)
        for start in range(0, pairs.shape[0], chunk):
            logits = model.probe(embeddings, pairs[start : start + chunk])
            out.append(logits.sigmoid().float().cpu().numpy())
    probability = np.concatenate(out)
    if probability.shape[0] != records.labels.shape[0]:
        raise RuntimeError("baseline probabilities do not align with the JEPA queries")
    return probability


def repeat_edge_flags(snapshots: Sequence, records: LatentRecords, windows: Sequence[Sequence], num_nodes: int) -> np.ndarray:
    """True for query rows whose (u, v) already occurred in any earlier snapshot."""
    ordered = sorted(snapshots, key=lambda s: int(s.time))
    needed = {int(window[-1].time) for window in windows}
    seen_until: dict[int, set] = {}
    seen: set = set()
    for snapshot in ordered:
        if int(snapshot.time) in needed:
            seen_until[int(snapshot.time)] = set(seen)  # edges strictly before this bin
        ei = snapshot.edge_index.detach().cpu().numpy().astype(np.int64)
        if ei.shape[1]:
            lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
            seen.update((lo * num_nodes + hi).tolist())
    flags = np.zeros(records.labels.shape[0], dtype=bool)
    for window_index, window in enumerate(windows):
        rows = np.flatnonzero(records.window_index == window_index)
        if rows.size == 0:
            continue
        before = seen_until[int(window[-1].time)]
        pairs = records.pairs[rows].astype(np.int64)
        keys = np.minimum(pairs[:, 0], pairs[:, 1]) * num_nodes + np.maximum(pairs[:, 0], pairs[:, 1])
        flags[rows] = np.fromiter((k in before for k in keys.tolist()), dtype=bool, count=keys.shape[0])
    return flags


def grouped_subset_metrics(records: LatentRecords, groups: dict[int, np.ndarray], probability: np.ndarray, positive_flag: np.ndarray) -> dict:
    """AP/AUC on the K-candidate groups whose positive satisfies / violates the flag."""
    labels = torch.as_tensor(records.labels, dtype=torch.float32)
    prob = torch.as_tensor(probability, dtype=torch.float32)
    flagged_rows, other_rows = [], []
    for rows in groups.values():
        positive = rows[records.labels[rows] > 0.5]
        if positive.size == 0:
            continue
        (flagged_rows if positive_flag[positive[0]] else other_rows).append(rows)
    result = {}
    for name, chunks in (("repeat", flagged_rows), ("new", other_rows)):
        if not chunks:
            result[name] = {"groups": 0, "ap": float("nan"), "auc": float("nan")}
            continue
        idx = torch.as_tensor(np.concatenate(chunks), dtype=torch.long)
        result[name] = {
            "groups": len(chunks),
            "ap": binary_average_precision(labels[idx], prob[idx]),
            "auc": binary_roc_auc(labels[idx], prob[idx]),
        }
    result["all"] = {
        "groups": len(groups),
        "ap": binary_average_precision(labels, prob),
        "auc": binary_roc_auc(labels, prob),
    }
    return result


# --------------------------------------------------------------------------- #
# node-level dynamics (JEPA target space vs. baseline encoder space)
# --------------------------------------------------------------------------- #
class NodeDynamics:
    """Per (test window, node) statistics of one encoder plus the arrays needed
    for the trajectory figure."""

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.times: list[int] = []
        self.first_time: int | None = None
        self.first_state: np.ndarray | None = None      # state at the bin before the first test target
        self.states: list[np.ndarray] = []              # [N, d] actual state at each test target bin
        self.predictions: list[np.ndarray] | None = None  # [N, d] predicted state (JEPA only)
        self.turnover: list[np.ndarray] = []
        self.involved: list[np.ndarray] = []
        self.deg_prev: list[np.ndarray] = []
        self.deg_next: list[np.ndarray] = []
        self.displacement: list[np.ndarray] = []
        self.relative: list[np.ndarray] = []
        self.spread: list[float] = []
        self.cos_pred: list[np.ndarray] = []
        self.cos_persist: list[np.ndarray] = []

    def stack(self, name: str) -> np.ndarray:
        return np.stack(getattr(self, name))


def neighbourhood_turnover(prev, nxt, num_nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """1 - Jaccard(N_{t-1}(u), N_t(u)) from the bidirected snapshot edges."""

    def keys(snapshot) -> np.ndarray:
        ei = snapshot.edge_index.detach().cpu().numpy().astype(np.int64)
        if ei.shape[1] == 0:
            return np.empty(0, dtype=np.int64)
        both = np.concatenate([ei, ei[[1, 0]]], axis=1)  # make sure it is symmetric
        return np.unique(both[0] * num_nodes + both[1])

    kp, kn = keys(prev), keys(nxt)
    common = np.intersect1d(kp, kn, assume_unique=True)
    deg_p = np.bincount(kp // num_nodes, minlength=num_nodes)
    deg_n = np.bincount(kn // num_nodes, minlength=num_nodes)
    deg_c = np.bincount(common // num_nodes, minlength=num_nodes)
    union = deg_p + deg_n - deg_c
    involved = union > 0
    turnover = np.zeros(num_nodes, dtype=np.float64)
    turnover[involved] = 1.0 - deg_c[involved] / union[involved]
    return turnover, involved, deg_p, deg_n


def _cos_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    return (F.normalize(a.float(), dim=-1) * F.normalize(b.float(), dim=-1)).sum(-1).cpu().numpy()


def _median_pair_distance(z: torch.Tensor, nodes: np.ndarray, rng: np.random.Generator, pairs: int = 4000) -> float:
    if nodes.size < 2:
        return float("nan")
    a = rng.choice(nodes, size=pairs)
    b = rng.choice(nodes, size=pairs)
    keep = a != b
    if not keep.any():
        return float("nan")
    a_t = torch.as_tensor(a[keep], dtype=torch.long, device=z.device)
    b_t = torch.as_tensor(b[keep], dtype=torch.long, device=z.device)
    return float(np.median(1.0 - _cos_rows(z[a_t], z[b_t])))


@torch.no_grad()
def collect_node_dynamics(encode_prev, encode_next, windows: Sequence[Sequence], num_nodes: int, seed: int) -> NodeDynamics:
    """``encode_prev(window)`` -> state at the last context bin; ``encode_next(window)``
    -> (state at the target bin, predicted state or None); both [N, d]."""
    dyn = NodeDynamics(num_nodes)
    rng = np.random.default_rng(seed)
    for window_index, window in enumerate(windows):
        z_prev = encode_prev(window)
        z_next, z_hat = encode_next(window)
        turnover, involved, deg_p, deg_n = neighbourhood_turnover(window[-2], window[-1], num_nodes)
        nodes = np.flatnonzero(involved)
        spread = _median_pair_distance(z_next, nodes, rng)
        displacement = 1.0 - _cos_rows(z_prev, z_next)
        dyn.times.append(int(window[-1].time))
        if window_index == 0:
            dyn.first_time = int(window[-2].time)
            dyn.first_state = z_prev.float().cpu().numpy()
        dyn.states.append(z_next.float().cpu().numpy())
        dyn.turnover.append(turnover)
        dyn.involved.append(involved)
        dyn.deg_prev.append(deg_p)
        dyn.deg_next.append(deg_n)
        dyn.displacement.append(displacement)
        dyn.relative.append(displacement / spread if np.isfinite(spread) and spread > 0 else np.full(num_nodes, np.nan))
        dyn.spread.append(spread)
        if z_hat is not None:
            if dyn.predictions is None:
                dyn.predictions = []
            dyn.predictions.append(z_hat.float().cpu().numpy())
            dyn.cos_pred.append(_cos_rows(z_hat, z_next))
            dyn.cos_persist.append(_cos_rows(z_prev, z_next))
    return dyn


def jepa_node_encoders(model: RCPSJEPA):
    def encode_prev(window):
        return model.encode_snapshot(window[-2], target=True)

    def encode_next(window):
        prepared = model.prepare_window(window)
        node_out = model._node_predictions(prepared)
        predicted = torch.zeros_like(prepared.target_embedding)
        predicted[node_out.node_ids] = node_out.prediction.to(predicted.dtype)
        return prepared.target_embedding, predicted

    return encode_prev, encode_next


def baseline_node_encoders(model):
    def encode_prev(window):
        return model.encode_context([window[-2]])

    def encode_next(window):
        return model.encode_context([window[-1]]), None

    return encode_prev, encode_next


def _average_ranks(x: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average = ends - (counts - 1) / 2.0
    return average[inverse]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = _average_ranks(x[keep]), _average_ranks(y[keep])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = math.sqrt(float((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else float("nan")


def turnover_bins(turnover: np.ndarray) -> np.ndarray:
    # right=True: edges[i-1] < x <= edges[i]; 0 -> "0", 0.25 -> "(0, .25]", 1 -> "1"
    return np.clip(np.digitize(turnover, TURNOVER_EDGES[1:-1], right=True), 0, len(TURNOVER_LABELS) - 1)


def _bin_stats(values: np.ndarray, bins: np.ndarray) -> list[dict]:
    rows = []
    for b in range(len(TURNOVER_LABELS)):
        v = values[(bins == b) & np.isfinite(values)]
        rows.append({
            "bin": TURNOVER_LABELS[b],
            "n": int(v.size),
            "mean": float(v.mean()) if v.size else float("nan"),
            "ci95": float(1.96 * v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else float("nan"),
        })
    return rows


def node_dynamics_metrics(dyn: NodeDynamics) -> dict:
    involved = dyn.stack("involved")
    turnover = dyn.stack("turnover")[involved]
    displacement = dyn.stack("displacement")[involved]
    relative = dyn.stack("relative")[involved]
    bins = turnover_bins(turnover)
    result = {
        "rows": int(turnover.size),
        "windows": len(dyn.times),
        "median_inter_node_distance_per_window": [float(s) for s in dyn.spread],
        "spearman_turnover_vs_displacement": spearman(turnover, displacement),
        "spearman_turnover_vs_relative_displacement": spearman(turnover, relative),
        "mean_relative_displacement": float(np.nanmean(relative)),
        "relative_displacement_by_turnover": _bin_stats(relative, bins),
        "displacement_by_turnover": _bin_stats(displacement, bins),
    }
    top = result["relative_displacement_by_turnover"][-1]["mean"]
    bottom = result["relative_displacement_by_turnover"][0]["mean"]
    result["relative_displacement_ratio_full_vs_no_change"] = float(top / bottom) if bottom and np.isfinite(bottom) else float("nan")
    if dyn.predictions is not None:
        cos_pred = dyn.stack("cos_pred")[involved]
        cos_persist = dyn.stack("cos_persist")[involved]
        result.update({
            "cos_prediction_vs_future": float(cos_pred.mean()),
            "cos_persistence_vs_future": float(cos_persist.mean()),
            "prediction_gain": float((cos_pred - cos_persist).mean()),
            "prediction_gain_positive_fraction": float((cos_pred > cos_persist).mean()),
            "cos_prediction_by_turnover": _bin_stats(cos_pred, bins),
            "cos_persistence_by_turnover": _bin_stats(cos_persist, bins),
            "prediction_gain_by_turnover": _bin_stats(cos_pred - cos_persist, bins),
        })
    return result


def choose_trajectory_node(dyn: NodeDynamics, min_degree: int = 2) -> int:
    """A node present in (as many as possible of) the test bins with high
    neighbourhood turnover: the case where 'nothing changes' is false."""
    involved = dyn.stack("involved")
    turnover = dyn.stack("turnover")
    degree = np.minimum(dyn.stack("deg_prev"), dyn.stack("deg_next"))
    presence = involved.sum(0)
    for required in range(len(dyn.times), 0, -1):
        candidates = np.flatnonzero(presence >= required)
        if candidates.size == 0:
            continue
        strong = candidates[(degree[:, candidates].min(0) >= min_degree)]
        pool = strong if strong.size else candidates
        score = np.where(involved[:, pool], turnover[:, pool], np.nan)
        mean_turnover = np.nanmean(score, axis=0)
        mean_degree = degree[:, pool].mean(0)
        order = np.lexsort((mean_degree, mean_turnover))[::-1]
        return int(pool[order[0]])
    raise RuntimeError("no node is involved in any test window")


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.normalize(a, dim=-1) @ F.normalize(b, dim=-1).T


def normalized_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1) - F.normalize(b, dim=-1)).square().sum(-1)


def _np(x) -> np.ndarray:
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def effective_rank(x, sample: int = 20000, seed: int = 0) -> tuple[float, np.ndarray]:
    """Entropy effective rank (Roy & Vetterli) of the centred latents and the
    normalised singular-value spectrum; numpy so the figures need no torch."""
    x = _np(x).astype(np.float64)
    if x.shape[0] > sample:
        take = np.random.default_rng(seed).choice(x.shape[0], size=sample, replace=False)
        x = x[take]
    centered = x - x.mean(0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    p = singular / max(singular.sum(), 1e-12)
    entropy = -(p * np.log(p + 1e-12)).sum()
    return float(np.exp(entropy)), singular / max(singular[0], 1e-12)


def group_rows(records: LatentRecords) -> dict[int, np.ndarray]:
    order = np.argsort(records.group_keys, kind="stable")
    keys = records.group_keys[order]
    boundaries = np.flatnonzero(np.diff(keys)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(keys)]])
    return {int(keys[s]): order[s:e] for s, e in zip(starts, ends)}


def group_matrix(records: LatentRecords, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Rows ordered with the positive first; returns (A, candidate dsts, positive index=0)."""
    rows = rows[np.argsort(-records.labels[rows], kind="stable")]
    index = torch.as_tensor(rows, dtype=torch.long)
    matrix = cosine(records.q_hat[index], records.q_bar[index]).numpy()
    destinations = records.pairs[rows, 1]
    return matrix, destinations, 0


def aggregate_metrics(records: LatentRecords, groups: dict[int, np.ndarray]) -> dict:
    labels = torch.as_tensor(records.labels, dtype=torch.float32)
    latent_score = -normalized_distance(records.q_hat, records.q_bar)
    node_score = (F.normalize(records.h_hat_u, dim=-1) * F.normalize(records.h_bar_v, dim=-1)).sum(-1)
    now_score = (F.normalize(records.h_now_u, dim=-1) * F.normalize(records.h_bar_v, dim=-1)).sum(-1)
    probability = torch.as_tensor(records.probability)

    top1_all: list[float] = []
    top1_pos: list[float] = []
    margins: list[float] = []
    pos_margins: list[float] = []
    for rows in groups.values():
        if rows.shape[0] < 2:
            continue
        matrix, _, pos = group_matrix(records, rows)
        k = matrix.shape[0]
        diag = np.diag(matrix)
        off = (matrix.sum() - diag.sum()) / max(1, k * k - k)
        margins.append(float(diag.mean() - off))
        top1_all.append(float((matrix.argmax(1) == np.arange(k)).mean()))
        top1_pos.append(float(matrix[pos].argmax() == pos))
        others = np.delete(matrix[pos], pos)
        pos_margins.append(float(matrix[pos, pos] - others.max()))
    erank_hat, _ = effective_rank(records.q_hat)
    erank_bar, _ = effective_rank(records.q_bar)
    positive = labels.bool()
    return {
        "groups": len(groups),
        "candidates_per_group": int(round(len(records.labels) / max(1, len(groups)))),
        "retrieval_top1_all_rows": float(np.mean(top1_all)),
        "retrieval_top1_positive_row": float(np.mean(top1_pos)),
        "agreement_margin_diag_minus_offdiag": float(np.mean(margins)),
        "positive_margin_vs_best_other": float(np.mean(pos_margins)),
        "positive_margin_positive_fraction": float(np.mean(np.asarray(pos_margins) > 0)),
        "latent_only_auc": binary_roc_auc(labels, latent_score),
        "latent_only_ap": binary_average_precision(labels, latent_score),
        "model_auc": binary_roc_auc(labels, probability),
        "model_ap": binary_average_precision(labels, probability),
        "node_future_cos_auc": binary_roc_auc(labels, node_score),
        "node_current_cos_auc": binary_roc_auc(labels, now_score),
        "node_future_cos_positive_mean": float(node_score[positive].mean()),
        "node_future_cos_negative_mean": float(node_score[~positive].mean()),
        "node_current_cos_positive_mean": float(now_score[positive].mean()),
        "node_current_cos_negative_mean": float(now_score[~positive].mean()),
        "effective_rank_q_hat": erank_hat,
        "effective_rank_q_bar": erank_bar,
        "q_dim": int(records.q_hat.shape[1]),
    }


def choose_case(records: LatentRecords, groups: dict[int, np.ndarray], rng: np.random.Generator) -> int:
    """A *typical* correctly ranked group: median positive margin among groups
    whose true destination the full model ranks first (not the best-looking one)."""
    candidates: list[tuple[float, int]] = []
    for key, rows in groups.items():
        if rows.shape[0] < 3:
            continue
        ordered = rows[np.argsort(-records.labels[rows], kind="stable")]
        probs = records.probability[ordered]
        if probs.argmax() != 0:
            continue
        matrix, _, pos = group_matrix(records, rows)
        others = np.delete(matrix[pos], pos)
        candidates.append((float(matrix[pos, pos] - others.max()), key))
    if not candidates:  # fall back to any group
        return int(rng.choice(list(groups)))
    candidates.sort()
    return candidates[len(candidates) // 2][1]


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise SystemExit("matplotlib is required for the figures: pip install matplotlib") from error
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42})
    return plt


def _tick_labels(destinations: np.ndarray, positive: int) -> list[str]:
    return [f"{'★ ' if i == positive else ''}{int(d)}" for i, d in enumerate(destinations)]


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _shared_range(arrays: Sequence[np.ndarray]) -> tuple[float, float]:
    """One colour range for every panel so the variants are comparable."""
    low = min(float(a.min()) for a in arrays)
    high = max(float(a.max()) for a in arrays)
    low = max(-1.0, math.floor(low * 10) / 10)
    high = min(1.0, math.ceil(high * 10) / 10)
    if high - low < 0.2:
        low, high = max(-1.0, high - 0.2), high
    return low, high


def plot_agreement(matrices: dict[str, tuple[np.ndarray, np.ndarray, int]], meta: dict, out: Path) -> None:
    plt = _matplotlib()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2), squeeze=False)
    vmin, vmax = _shared_range([m for m, _, _ in matrices.values()])
    threshold = vmin + 0.6 * (vmax - vmin)
    for ax, (variant, (matrix, destinations, pos)) in zip(axes[0], matrices.items()):
        k = matrix.shape[0]
        image = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax)
        for i in range(k):
            for j in range(k):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if value > threshold else "#222222")
        ticks = _tick_labels(destinations, pos)
        ax.set_xticks(range(k), ticks, rotation=90)
        ax.set_yticks(range(k), ticks)
        ax.set_xlabel(r"predicted latent $\hat{q}(u, v_i)$ from context")
        if ax is axes[0][0]:
            ax.set_ylabel(r"future latent $\bar{q}(u, v_j)$ from target encoder")
        ax.set_title(VARIANT_TITLES.get(variant, variant))
        # outline the true destination's row/column
        ax.add_patch(plt.Rectangle((-0.5, pos - 0.5), k, 1, fill=False, lw=1.2, ec="#d62728"))
        ax.add_patch(plt.Rectangle((pos - 0.5, -0.5), 1, k, fill=False, lw=1.2, ec="#d62728"))
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="cosine similarity")
    fig.suptitle(
        f"{meta['dataset']}: source u={meta['source']} at test bin {meta['target_time']} "
        f"(★ = true destination; {meta['candidates']} candidates)", fontsize=8
    )
    _save(fig, out / "agreement_heatmap")
    plt.close(fig)


def plot_node_future(rows_by_variant: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]], meta: dict, out: Path) -> None:
    plt = _matplotlib()
    n = len(rows_by_variant)
    fig, axes = plt.subplots(n, 1, figsize=(3.6, 1.3 * n + 0.9), squeeze=False)
    stacked = {v: np.stack([future, current]) for v, (future, current, _, _) in rows_by_variant.items()}
    vmin, vmax = _shared_range(list(stacked.values()))
    threshold = vmin + 0.6 * (vmax - vmin)
    for ax, (variant, (future, current, destinations, pos)) in zip(axes[:, 0], rows_by_variant.items()):
        matrix = stacked[variant]
        image = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(2):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if matrix[i, j] > threshold else "#222222")
        ax.set_yticks([0, 1], [r"$\hat{h}_u$ (predicted future)", r"$h_{t-1,u}$ (current)"])
        ax.set_xticks(range(matrix.shape[1]))
        if ax is axes[-1, 0]:  # candidate ids only under the bottom panel
            ax.set_xticklabels(_tick_labels(destinations, pos), rotation=90)
        else:
            ax.set_xticklabels([])
        ax.add_patch(plt.Rectangle((pos - 0.5, -0.5), 1, 2, fill=False, lw=1.2, ec="#d62728"))
        ax.set_title(VARIANT_TITLES.get(variant, variant))
        ax.tick_params(length=0)
    axes[-1, 0].set_xlabel(r"cosine with future node latent $\bar{h}_{v_j}$ of each candidate")
    fig.subplots_adjust(hspace=0.45)
    fig.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.03, pad=0.02)
    _save(fig, out / "node_future")
    plt.close(fig)


def plot_pca(records_by_variant: dict[str, LatentRecords], metrics: dict[str, dict], out: Path, seed: int) -> None:
    plt = _matplotlib()
    n = len(records_by_variant)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, (variant, records) in zip(axes[0], records_by_variant.items()):
        q = _np(records.q_hat)
        labels = _np(records.labels)
        take = rng.choice(q.shape[0], size=min(6000, q.shape[0]), replace=False)
        x = q[take].astype(np.float64)
        x = x - x.mean(0, keepdims=True)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        proj = x @ vt[:2].T
        neg = labels[take] == 0
        ax.scatter(proj[neg, 0], proj[neg, 1], s=4, c="#b0b0b0", alpha=0.5, linewidths=0, label="no link")
        ax.scatter(proj[~neg, 0], proj[~neg, 1], s=4, c="#1f77b4", alpha=0.7, linewidths=0, label="link forms")
        ax.set_title(VARIANT_TITLES.get(variant, variant))
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.text(0.02, 0.98, f"latent-only AUC {metrics[variant]['latent_only_auc']:.3f}\n"
                f"eff. rank {metrics[variant]['effective_rank_q_hat']:.1f} / {metrics[variant]['q_dim']}",
                transform=ax.transAxes, va="top", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0][0].legend(loc="lower left", fontsize=7, frameon=False, markerscale=3)
    fig.tight_layout()
    fig.suptitle(r"PCA of context-predicted relation latents $\hat{q}_{uv}$ on the test set", fontsize=8, y=1.03)
    _save(fig, out / "latent_pca")
    plt.close(fig)


def plot_spectrum(records_by_variant: dict[str, LatentRecords], metrics: dict[str, dict], out: Path) -> None:
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    colors = {"full": "#1f77b4", "no_jepa": "#ff7f0e"}
    for variant, records in records_by_variant.items():
        erank, spectrum = effective_rank(records.q_hat)
        ax.plot(np.arange(1, len(spectrum) + 1), spectrum, lw=1.5, color=colors.get(variant),
                label=f"{VARIANT_TITLES.get(variant, variant)} (eff. rank {erank:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.set_title(r"Spectrum of $\hat{q}_{uv}$ (collapse check)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, out / "singular_values")
    plt.close(fig)


def plot_change_tracking(dynamics: dict[str, NodeDynamics], node_metrics: dict[str, dict], meta: dict, out: Path) -> None:
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(len(TURNOVER_LABELS))
    ax = axes[0]
    for name, dyn in dynamics.items():
        color, marker = METHOD_STYLE.get(name, ("#444444", "o"))
        rows = node_metrics[name]["relative_displacement_by_turnover"]
        mean = np.array([r["mean"] for r in rows])
        ci = np.array([r["ci95"] for r in rows])
        rho = node_metrics[name]["spearman_turnover_vs_displacement"]
        ax.errorbar(x, mean, yerr=np.nan_to_num(ci), color=color, marker=marker, ms=4, lw=1.5, capsize=2,
                    label=f"{VARIANT_TITLES.get(name, name)}  (Spearman {rho:.2f})")
    ax.set_xticks(x, TURNOVER_LABELS)
    ax.set_xlabel(r"neighbourhood turnover of $u$: $1-\mathrm{Jaccard}(N_{t-1}(u), N_t(u))$")
    ax.set_ylabel("relative latent displacement\n$(1-\\cos(z_{t-1,u}, z_{t,u}))$ / median inter-node dist.")
    ax.set_title("Latent displacement vs. neighbourhood turnover")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    for name, dyn in dynamics.items():
        if dyn.predictions is None:
            continue
        color, marker = METHOD_STYLE.get(name, ("#444444", "o"))
        pred = np.array([r["mean"] for r in node_metrics[name]["cos_prediction_by_turnover"]])
        persist = np.array([r["mean"] for r in node_metrics[name]["cos_persistence_by_turnover"]])
        short = "w/o JEPA losses" if name == "no_jepa" else VARIANT_TITLES.get(name, name)
        ax.plot(x, pred, color=color, marker=marker, ms=4, lw=1.5,
                label=rf"{short}: predicted $\hat h_u$")
        ax.plot(x, persist, color=color, marker=marker, ms=4, lw=1.2, ls="--", mfc="white",
                label=rf"{short}: persistence $\bar h_{{t-1,u}}$")
    ax.set_xticks(x, TURNOVER_LABELS)
    ax.set_xlabel("neighbourhood turnover of $u$")
    ax.set_ylabel(r"cosine with the actual future $\bar h_{t,u}$")
    ax.set_title("JEPA: predicted vs. persisted future")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.2)
    fig.suptitle(f"{meta['dataset']}: node-level dynamics over the test bins (seed {meta['seed']})", fontsize=8, y=1.02)
    fig.tight_layout()
    _save(fig, out / "change_tracking")
    plt.close(fig)


def _pca_transform(cloud: np.ndarray):
    center = cloud.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(cloud - center, full_matrices=False)
    basis = vt[:2].T
    scores = (cloud - center) @ basis
    scale = scores.std(0, keepdims=True) + 1e-12

    def transform(x: np.ndarray) -> np.ndarray:
        return ((np.atleast_2d(x) - center) @ basis) / scale

    return transform, scores / scale


def plot_trajectory(dynamics: dict[str, NodeDynamics], node: int, meta: dict, out: Path) -> None:
    plt = _matplotlib()
    names = list(dynamics)
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        dyn = dynamics[name]
        color, marker = METHOD_STYLE.get(name, ("#444444", "o"))
        last_nodes = np.flatnonzero(dyn.involved[-1])
        transform, cloud = _pca_transform(dyn.states[-1][last_nodes])
        ax.scatter(cloud[:, 0], cloud[:, 1], s=3, c="#c8c8c8", alpha=0.5, linewidths=0,
                   label=f"all involved nodes at bin {dyn.times[-1]}")
        points = transform(np.stack([dyn.first_state[node]] + [s[node] for s in dyn.states]))
        ax.plot(points[:, 0], points[:, 1], color=color, lw=1.2, alpha=0.9)
        ax.scatter(points[:, 0], points[:, 1], color=color, s=22, marker=marker, zorder=3, label=f"actual state of u={node}")
        for k, (px, py) in enumerate(points):
            ax.annotate(str(dyn.first_time if k == 0 else dyn.times[k - 1]), (px, py), textcoords="offset points",
                        xytext=(3, 3), fontsize=6, color="#333333")
        if dyn.predictions is not None:
            predicted = transform(np.stack([p[node] for p in dyn.predictions]))
            for k in range(len(predicted)):
                ax.plot([points[k, 0], predicted[k, 0]], [points[k, 1], predicted[k, 1]], color=color, lw=0.9, ls="--", alpha=0.8)
            ax.scatter(predicted[:, 0], predicted[:, 1], s=30, marker=marker, facecolors="white", edgecolors=color,
                       linewidths=1.2, zorder=4, label=r"predicted next state $\hat h_u$")
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        ax.text(0.02, 0.98, f"mean step {steps.mean():.2f} cloud-std\n"
                f"mean turnover {np.nanmean(np.where(dyn.stack('involved')[:, node], dyn.stack('turnover')[:, node], np.nan)):.2f}",
                transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_title(VARIANT_TITLES.get(name, name))
        ax.set_xlabel("PC 1 (cloud std units)")
        ax.set_ylabel("PC 2 (cloud std units)")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False, loc="best", markerscale=1.0)
    fig.suptitle(f"{meta['dataset']}: trajectory of node u={node} across the test bins (labels = bin index)", fontsize=8, y=1.02)
    fig.tight_layout()
    _save(fig, out / "trajectory")
    plt.close(fig)


def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if not np.isfinite(value) else f"{value:.4f}"
    return str(value)


def write_markdown(
    metrics: dict[str, dict],
    model_metrics: dict[str, dict],
    meta: dict,
    out: Path,
    node_metrics: dict[str, dict] | None = None,
    subset_metrics: dict[str, dict] | None = None,
    baseline: str | None = None,
    trajectory_node: int | None = None,
) -> None:
    keys = [
        ("model_ap", "model AP (K-candidate groups)"),
        ("model_auc", "model AUC (K-candidate groups)"),
        ("latent_only_ap", "AP of −d(q̂, q̄) alone"),
        ("latent_only_auc", "AUC of −d(q̂, q̄) alone"),
        ("retrieval_top1_positive_row", "top-1 retrieval of q̄ for the true pair"),
        ("retrieval_top1_all_rows", "top-1 retrieval, all candidates"),
        ("agreement_margin_diag_minus_offdiag", "diag − off-diag cosine"),
        ("positive_margin_vs_best_other", "true pair: own cell − best other"),
        ("positive_margin_positive_fraction", "fraction of groups with positive margin"),
        ("node_future_cos_auc", "AUC of cos(ĥ_u, h̄_v)"),
        ("node_current_cos_auc", "AUC of cos(h_{t-1,u}, h̄_v) (control)"),
        ("effective_rank_q_hat", "effective rank of q̂"),
        ("effective_rank_q_bar", "effective rank of q̄"),
    ]
    variants = list(metrics)
    lines = [f"# Latent analysis: {meta['dataset']} (seed {meta['seed']}, {meta['candidates']} candidates/group, {metrics[variants[0]]['groups']} test groups)", ""]
    lines.append("## DyGJEPA relation latents (pair level)")
    lines.append("")
    lines.append("| metric | " + " | ".join(VARIANT_TITLES.get(v, v) for v in variants) + " |")
    lines.append("|---|" + "---|" * len(variants))
    lines.append("| test AP / AUC (1:1 protocol, best checkpoint) | " + " | ".join(
        f"{model_metrics[v]['test']['ap']:.4f} / {model_metrics[v]['test']['auc']:.4f}" for v in variants) + " |")
    for key, label in keys:
        lines.append(f"| {label} | " + " | ".join(_fmt(metrics[v][key]) for v in variants) + " |")
    lines += ["", f"Case study: source u={meta['source']}, test bin {meta['target_time']}, "
              f"true destination {meta['positive_destination']} (median-margin group among those the full model ranks correctly)."]

    if node_metrics:
        methods = list(node_metrics)
        lines += ["", f"## Node-level dynamics over the test bins ({node_metrics[methods[0]]['windows']} windows)", ""]
        lines.append("Displacement = 1 − cos(z_{t-1,u}, z_{t,u}) of the same encoder at the last context bin and the target bin "
                     "(JEPA: target encoder; baseline: its SSL encoder on one bin); relative = divided by the method's median "
                     "inter-node distance at bin t. Turnover = 1 − Jaccard(N_{t-1}(u), N_t(u)).")
        lines.append("")
        lines.append("| metric | " + " | ".join(VARIANT_TITLES.get(m, m) for m in methods) + " |")
        lines.append("|---|" + "---|" * len(methods))
        if baseline is not None and baseline in model_metrics:
            lines.append("| test AP / AUC (1:1 protocol) | " + " | ".join(
                f"{model_metrics[m]['test']['ap']:.4f} / {model_metrics[m]['test']['auc']:.4f}" if m in model_metrics else "-" for m in methods) + " |")
        for key, label in [
            ("spearman_turnover_vs_displacement", "Spearman(turnover, displacement)"),
            ("spearman_turnover_vs_relative_displacement", "Spearman(turnover, relative displacement)"),
            ("mean_relative_displacement", "mean relative displacement"),
            ("relative_displacement_ratio_full_vs_no_change", "relative displacement: turnover 1 / turnover 0"),
            ("cos_prediction_vs_future", "cos(ĥ_u, h̄_{t,u}) predicted future"),
            ("cos_persistence_vs_future", "cos(h̄_{t-1,u}, h̄_{t,u}) persistence"),
            ("prediction_gain", "prediction gain (predicted − persistence)"),
            ("prediction_gain_positive_fraction", "fraction of nodes with positive gain"),
        ]:
            lines.append(f"| {label} | " + " | ".join(_fmt(node_metrics[m].get(key, "-")) for m in methods) + " |")
        lines += ["", "Relative displacement by turnover bin (mean, n):", ""]
        lines.append("| turnover | " + " | ".join(VARIANT_TITLES.get(m, m) for m in methods) + " |")
        lines.append("|---|" + "---|" * len(methods))
        for b, label in enumerate(TURNOVER_LABELS):
            lines.append(f"| {label} | " + " | ".join(
                f"{_fmt(node_metrics[m]['relative_displacement_by_turnover'][b]['mean'])} (n={node_metrics[m]['relative_displacement_by_turnover'][b]['n']})"
                for m in methods) + " |")
        if trajectory_node is not None:
            lines += ["", f"Trajectory figure: node u={trajectory_node} (present in the most test bins, highest mean turnover)."]

    if subset_metrics:
        methods = list(subset_metrics)
        lines += ["", "## Same K-candidate test queries, positives split by history", ""]
        lines.append("A positive (u, v) is *repeat* if the edge occurred in any earlier snapshot, *new* otherwise; "
                     "every group keeps its K−1 random negatives. Baseline scores come from its frozen encoder + probe on the identical pairs.")
        lines.append("")
        lines.append("| subset | " + " | ".join(f"{VARIANT_TITLES.get(m, m)} AP / AUC" for m in methods) + " |")
        lines.append("|---|" + "---|" * len(methods))
        for subset in ("all", "repeat", "new"):
            lines.append(f"| {subset} ({subset_metrics[methods[0]][subset]['groups']} groups) | " + " | ".join(
                f"{_fmt(subset_metrics[m][subset]['ap'])} / {_fmt(subset_metrics[m][subset]['auc'])}" for m in methods) + " |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/link_comparison_all.yaml"))
    parser.add_argument("--dataset", required=True, help="dataset name from the config, e.g. uci")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="cap rcps_training.epochs")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--baseline", default="cldg", choices=["none", *BASELINE_CLASSES],
                        help="snapshot-SSL method to compare the latent dynamics with (default cldg)")
    parser.add_argument("--baseline-pretrain-epochs", type=int, default=None, help="cap <baseline>_training.pretrain_epochs")
    parser.add_argument("--candidates", type=int, default=8, help="K = 1 true + K-1 random destinations per source")
    parser.add_argument("--max-groups", type=int, default=20000, help="cap on test groups collected")
    parser.add_argument("--chunk", type=int, default=2048, help="forward batch size during collection")
    parser.add_argument("--query-seed", type=int, default=2, help="seed of the K-candidate test queries (DyGLib test seed)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None, help="default results/latent/<dataset>")
    parser.add_argument("--reuse", action="store_true", help="load state_<variant>.pt from --out instead of training")
    args = parser.parse_args()

    out = args.out or Path("results/latent") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DYGJEPA_WANDB", "0")
    device = choose_device(args.device)
    cfg = dataset_config(args.config, args.dataset)
    graph, split, link_cfg, rcps_args, rcps_training = build_protocol(cfg, args.seed, device)
    if args.epochs is not None:
        rcps_training["epochs"] = int(args.epochs)
    baseline = None if args.baseline == "none" else args.baseline
    print(json.dumps({"dataset": args.dataset, "seed": args.seed, "device": device_description(device),
                      "test_windows": len(split.test), "epochs": rcps_training["epochs"],
                      "variants": args.variants, "baseline": baseline, "out": str(out)}), flush=True)

    records_by_variant: dict[str, LatentRecords] = {}
    metrics: dict[str, dict] = {}
    model_metrics: dict[str, dict] = {}
    groups_by_variant: dict[str, dict[int, np.ndarray]] = {}
    dynamics: dict[str, NodeDynamics] = {}
    node_metrics: dict[str, dict] = {}
    for variant in args.variants:
        model, trained = train_variant(variant, graph, split, rcps_args, rcps_training, args.seed, device, out, args.reuse)
        model_metrics[variant] = trained
        records = collect_latents(model, split.test, candidates=args.candidates, query_seed=args.query_seed,
                                  max_groups=args.max_groups, chunk=args.chunk)
        groups = group_rows(records)
        records_by_variant[variant] = records
        groups_by_variant[variant] = groups
        metrics[variant] = aggregate_metrics(records, groups)
        print(f"[{variant}] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics[variant].items()}), flush=True)
        encode_prev, encode_next = jepa_node_encoders(model)
        dynamics[variant] = collect_node_dynamics(encode_prev, encode_next, split.test, graph.num_nodes, args.seed)
        node_metrics[variant] = node_dynamics_metrics(dynamics[variant])
        print(f"[{variant}] node dynamics: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in node_metrics[variant].items() if not isinstance(v, list)}), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    reference = args.variants[0]
    subset_metrics: dict[str, dict] = {}
    repeat_flags_by_variant = {
        variant: repeat_edge_flags(graph.snapshots, records, split.test, graph.num_nodes)
        for variant, records in records_by_variant.items()
    }
    repeat_flags = repeat_flags_by_variant[reference]
    for variant, records in records_by_variant.items():
        subset_metrics[variant] = grouped_subset_metrics(
            records, groups_by_variant[variant], records.probability, repeat_flags_by_variant[variant]
        )
    if baseline is not None:
        baseline_model, trained = train_baseline(baseline, graph, split, cfg, link_cfg, args.seed, device, out,
                                                 args.reuse, args.baseline_pretrain_epochs)
        model_metrics[baseline] = trained
        encode_prev, encode_next = baseline_node_encoders(baseline_model)
        dynamics[baseline] = collect_node_dynamics(encode_prev, encode_next, split.test, graph.num_nodes, args.seed)
        node_metrics[baseline] = node_dynamics_metrics(dynamics[baseline])
        print(f"[{baseline}] node dynamics: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in node_metrics[baseline].items() if not isinstance(v, list)}), flush=True)
        probability = baseline_probabilities(baseline_model, split.test, records_by_variant[reference], device, args.chunk)
        subset_metrics[baseline] = grouped_subset_metrics(records_by_variant[reference], groups_by_variant[reference],
                                                          probability, repeat_flags)
        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # case study chosen on the first (full) variant; the same group ids exist in every variant
    rng = np.random.default_rng(args.seed)
    case_key = choose_case(records_by_variant[reference], groups_by_variant[reference], rng)
    matrices: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    node_rows: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    meta: dict = {}
    for variant, records in records_by_variant.items():
        rows = groups_by_variant[variant][case_key]
        ordered = rows[np.argsort(-records.labels[rows], kind="stable")]
        matrix, destinations, pos = group_matrix(records, rows)
        matrices[variant] = (matrix, destinations, pos)
        index = torch.as_tensor(ordered, dtype=torch.long)
        future = (F.normalize(records.h_hat_u[index], dim=-1) * F.normalize(records.h_bar_v[index], dim=-1)).sum(-1).numpy()
        current = (F.normalize(records.h_now_u[index], dim=-1) * F.normalize(records.h_bar_v[index], dim=-1)).sum(-1).numpy()
        node_rows[variant] = (future, current, destinations, pos)
        if not meta:
            meta = {
                "dataset": args.dataset, "seed": args.seed, "candidates": args.candidates,
                "source": int(records.pairs[ordered[0], 0]), "positive_destination": int(destinations[pos]),
                "target_time": int(records.target_time[ordered[0]]), "group": int(case_key),
            }
    plot_agreement(matrices, meta, out)
    plot_node_future(node_rows, meta, out)
    plot_pca(records_by_variant, metrics, out, args.seed)
    plot_spectrum(records_by_variant, metrics, out)
    plot_change_tracking(dynamics, node_metrics, meta, out)
    trajectory_node = choose_trajectory_node(dynamics[reference])
    trajectory_methods = {k: dynamics[k] for k in ([reference, baseline] if baseline else [reference])}
    plot_trajectory(trajectory_methods, trajectory_node, meta, out)
    meta["trajectory_node"] = trajectory_node
    meta["baseline"] = baseline
    (out / "metrics.json").write_text(json.dumps({
        "meta": meta, "latent": metrics, "model": model_metrics,
        "node_dynamics": node_metrics, "history_subsets": subset_metrics,
    }, indent=2, default=str))
    np.savez_compressed(out / "case_matrices.npz", **{f"{v}_matrix": m for v, (m, _, _) in matrices.items()},
                        destinations=next(iter(matrices.values()))[1])
    write_markdown(metrics, model_metrics, meta, out, node_metrics, subset_metrics, baseline, trajectory_node)
    print(f"figures and metrics written to {out}")


if __name__ == "__main__":
    main()
