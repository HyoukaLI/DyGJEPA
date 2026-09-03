import torch

from jepa_compare.data import Snapshot
from jepa_compare.snapshot_ssl_baselines import (
    CLDGLinkBaseline,
    DVGMAELinkBaseline,
    MaskDGNNLinkBaseline,
)


def _snapshots() -> list[Snapshot]:
    generator = torch.Generator().manual_seed(9)
    snapshots = []
    # Four sources [0,4), six destinations [4,10).  Each bin leaves many
    # destination-corrupted non-edges for deterministic negative sampling.
    pairs_by_time = [
        [(0, 4), (1, 5), (2, 6)],
        [(0, 5), (1, 6), (3, 7)],
        [(0, 6), (2, 7), (3, 8)],
        [(1, 7), (2, 8), (3, 9)],
        [(0, 8), (1, 9), (2, 4)],
        [(0, 9), (2, 5), (3, 4)],
    ]
    for time, rows in enumerate(pairs_by_time):
        directed = torch.tensor(rows, dtype=torch.long).T
        message = torch.cat([directed, directed.flip(0)], dim=1)
        snapshots.append(
            Snapshot(
                x=torch.randn(10, 5, generator=generator),
                edge_index=message,
                active=torch.ones(10, dtype=torch.bool),
                time=time,
                query_edge_index=directed,
                query_timestamps=torch.full((len(rows),), float(time)),
            )
        )
    return snapshots


def _exercise(model) -> None:
    snapshots = _snapshots()
    pretrain_optimizer = torch.optim.Adam(model.pretrain_parameters(), lr=1e-3)
    metrics = model.pretrain_epoch(snapshots[:4], pretrain_optimizer, 1.0, 42)
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    model.freeze_encoder()
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("probe.")
    )
    windows = [snapshots[0:3], snapshots[1:4]]
    probe_optimizer = torch.optim.Adam(model.probe.parameters(), lr=1e-3)
    probe_metrics = model.train_probe_epoch(
        windows, probe_optimizer, 1.0, pair_batch_size=8, seed=7
    )
    result = model.evaluate_windows(windows, pair_batch_size=8, query_seed=7)
    assert torch.isfinite(torch.tensor(probe_metrics["loss"]))
    assert result["examples"] == 12.0
    assert 0.0 <= result["ap"] <= 1.0
    assert 0.0 <= result["auc"] <= 1.0


def _link_kwargs() -> dict:
    return {
        "negative_ratio": 1.0,
        "max_positive_pairs": None,
        "new_edges_only": False,
        "undirected": False,
        "bipartite_source_count": 4,
        "probe_hidden_dim": 8,
    }


def test_cldg_pretrain_and_frozen_link_probe() -> None:
    _exercise(
        CLDGLinkBaseline(
            feature_dim=5,
            hidden_dim=8,
            embedding_dim=8,
            num_layers=1,
            num_spans=2,
            num_views=2,
            contrastive_batch_size=8,
            **_link_kwargs(),
        )
    )


def test_maskdgnn_pretrain_and_frozen_link_probe() -> None:
    _exercise(
        MaskDGNNLinkBaseline(
            feature_dim=5,
            hidden_dim=8,
            num_layers=1,
            window_size=2,
            pretrain_pair_limit=8,
            **_link_kwargs(),
        )
    )


def test_dvgmae_pretrain_and_frozen_link_probe() -> None:
    _exercise(
        DVGMAELinkBaseline(
            feature_dim=5,
            hidden_dim=8,
            num_layers=1,
            window_size=2,
            pretrain_pair_limit=8,
            **_link_kwargs(),
        )
    )
