from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import load_npz, make_synthetic
from .node_evaluation import final_probe, stratified_split, validation_probe
from .sg_jepa import SGJEPA


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def release_device_memory(device: torch.device) -> None:
    """Best-effort release of cached accelerator memory between long training runs."""
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def run(config: dict) -> dict[str, float]:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(config.get("device", "auto"))
    data_cfg = config["data"]
    if data_cfg.get("path"):
        graph = load_npz(data_cfg["path"])
    else:
        graph = make_synthetic(
            **{k: v for k, v in data_cfg.items() if k not in {"path", "fanout"}}, seed=seed
        )
    if data_cfg.get("fanout"):
        graph = graph.sample_neighbors(int(data_cfg["fanout"]), seed=seed)
    graph = graph.to(device)
    model = SGJEPA(feature_dim=graph.feature_dim, **config["model"]).to(device)
    train_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    split = None if graph.labels is None else stratified_split(
        graph.labels, train_cfg["train_ratio"],
        train_cfg.get("validation_ratio_within_train", 0.1), seed,
    )
    checkpoint_every = int(train_cfg.get("checkpoint_every", train_cfg["eval_every"]))
    selection_epochs = int(train_cfg.get("selection_probe_epochs", 100))
    selection_metric = train_cfg.get("selection_metric", "macro_f1")
    checkpoint_path = Path(train_cfg.get("checkpoint_path", "checkpoints/best.pt"))
    best_score, best_epoch, best_state = float("-inf"), 0, None
    metrics: dict[str, float] = {}

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss, metrics = model.loss(graph, batch_size=train_cfg.get("batch_size"))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
        optimizer.step()
        if epoch == 1 or epoch % train_cfg["eval_every"] == 0:
            print(json.dumps({"epoch": epoch, **metrics}))

        if split is not None and (epoch % checkpoint_every == 0 or epoch == train_cfg["epochs"]):
            model.eval()
            embeddings, node_ids = model.infer(graph)
            validation = validation_probe(
                embeddings, graph.labels[node_ids], split, selection_epochs, seed,
                train_cfg.get("probe_hidden_dim"),
            )
            print(json.dumps({"validation_probe": {"epoch": epoch, **validation}}))
            score = validation[selection_metric]
            if score > best_score:
                best_score, best_epoch, best_state = score, epoch, cpu_state_dict(model)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"epoch": epoch, "model_state": best_state,
                     "validation": validation, "config": config}, checkpoint_path,
                )

    if split is not None and best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        embeddings, node_ids = model.infer(graph)
        metrics.update(final_probe(
            embeddings, graph.labels[node_ids], split, train_cfg["probe_epochs"], seed,
            train_cfg.get("probe_hidden_dim"),
        ))
        print(json.dumps({"final_probe": {"best_epoch": best_epoch, **metrics}}))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the independent SG-JEPA reproduction")
    parser.add_argument("--config", type=Path, default=Path("configs/sg_node_synthetic.yaml"))
    parser.add_argument("--epochs", type=int, default=None, help="override training epochs")
    args = parser.parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    run(config)


if __name__ == "__main__":
    main()
