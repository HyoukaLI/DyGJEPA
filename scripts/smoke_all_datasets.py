from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from time import perf_counter

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_compare.compare_link_prediction import _dataset_configs, run as run_link
from jepa_compare.compare_node_prediction import run as run_node
from jepa_compare.data import load_npz, make_synthetic
from jepa_compare.train_sg_jepa import run as run_sg


LINK_CONFIG = ROOT / "configs" / "link_comparison_all.yaml"
NODE_SYNTHETIC = ROOT / "configs" / "node_comparison_synthetic.yaml"
SG_SYNTHETIC = ROOT / "configs" / "sg_node_synthetic.yaml"
DBLP_NPZ = ROOT / "data" / "processed" / "dblp.npz"


def _graph_stats(graph) -> dict[str, int | str | None]:
    queries = sum(
        0 if snap.query_edge_index is None else int(snap.query_edge_index.shape[1])
        for snap in graph.snapshots
    )
    edges = sum(int(snap.edge_index.shape[1]) for snap in graph.snapshots)
    return {
        "snapshots": len(graph.snapshots),
        "nodes": graph.num_nodes,
        "feature_dim": graph.feature_dim,
        "message_edges": edges,
        "query_events": queries,
        "labels": None if graph.labels is None else int(graph.labels.numel()),
        "kind": "bipartite" if graph.num_source_nodes is not None else "homogeneous",
        "source_nodes": graph.num_source_nodes,
    }


def _record(name: str, ok: bool, elapsed: float, detail: dict) -> dict:
    status = "ok" if ok else "fail"
    print(json.dumps({"smoke": name, "status": status, "seconds": round(elapsed, 2), **detail}))
    return {"name": name, "ok": ok, "seconds": elapsed, **detail}


def smoke_load_link(config_path: Path) -> list[dict]:
    config = yaml.safe_load(config_path.read_text())
    rows: list[dict] = []
    for name, dataset_config in _dataset_configs(config):
        started = perf_counter()
        path = Path(dataset_config["data"]["path"])
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            graph = load_npz(path)
            windows = list(graph.windows(int(dataset_config["common_model"]["window_size"])))
            if len(windows) < 3:
                raise ValueError(f"need at least 3 temporal windows, got {len(windows)}")
            detail = {"task": "load_link", **_graph_stats(graph), "windows": len(windows)}
            rows.append(_record(f"load:{name}", True, perf_counter() - started, detail))
        except Exception as exc:
            rows.append(
                _record(
                    f"load:{name}",
                    False,
                    perf_counter() - started,
                    {"task": "load_link", "error": str(exc), "trace": traceback.format_exc()},
                )
            )
    return rows


def smoke_link_edgebank(config_path: Path, output_dir: Path, max_positive_pairs: int) -> list[dict]:
    config = yaml.safe_load(config_path.read_text())
    config["device"] = "cpu"
    config["models"] = ["edgebank"]
    seed = config.get("seed", 42)
    config["seed"] = int(seed[0] if isinstance(seed, list) else seed)
    config.setdefault("link", {})["max_positive_pairs"] = max_positive_pairs
    config["output_dir"] = str(output_dir)
    config["summary_output_path"] = str(output_dir / "link_edgebank.json")
    rows: list[dict] = []
    for name, dataset_config in _dataset_configs(config):
        started = perf_counter()
        try:
            result = run_link(dataset_config)
            test = result["edgebank"]["test"]
            detail = {
                "task": "link_edgebank",
                "negative_ratio": dataset_config.get("link", {}).get("negative_ratio"),
                "ap": test.get("ap"),
                "auc": test.get("auc"),
                "mrr": test.get("mrr"),
            }
            rows.append(_record(f"link:{name}", True, perf_counter() - started, detail))
        except RuntimeError as exc:
            if "unable to sample enough negative links" not in str(exc):
                rows.append(
                    _record(
                        f"link:{name}",
                        False,
                        perf_counter() - started,
                        {"task": "link_edgebank", "error": str(exc), "trace": traceback.format_exc()},
                    )
                )
                continue
            fallback = dict(dataset_config)
            fallback["link"] = {**dict(dataset_config.get("link", {})), "negative_ratio": 4.0}
            try:
                result = run_link(fallback)
                test = result["edgebank"]["test"]
                detail = {
                    "task": "link_edgebank",
                    "ok_with_fallback": True,
                    "default_error": str(exc),
                    "negative_ratio": 4.0,
                    "ap": test.get("ap"),
                    "auc": test.get("auc"),
                    "mrr": test.get("mrr"),
                }
                rows.append(_record(f"link:{name}", True, perf_counter() - started, detail))
            except Exception as fallback_exc:
                rows.append(
                    _record(
                        f"link:{name}",
                        False,
                        perf_counter() - started,
                        {
                            "task": "link_edgebank",
                            "error": str(fallback_exc),
                            "default_error": str(exc),
                            "trace": traceback.format_exc(),
                        },
                    )
                )
        except Exception as exc:
            rows.append(
                _record(
                    f"link:{name}",
                    False,
                    perf_counter() - started,
                    {"task": "link_edgebank", "error": str(exc), "trace": traceback.format_exc()},
                )
            )
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return rows


def smoke_node_synthetic(output_dir: Path) -> dict:
    config = yaml.safe_load(NODE_SYNTHETIC.read_text())
    config["device"] = "cpu"
    config["output_path"] = str(output_dir / "node_synthetic.json")
    config["node_baselines"] = {"enabled": []}
    config["training"]["epochs"] = 1
    config["training"]["min_checkpoint_epoch"] = 1
    config["training"]["selection_probe_epochs"] = 5
    config["training"]["probe_epochs"] = 5
    started = perf_counter()
    try:
        result = run_node(config)
        return _record(
            "node:synthetic",
            True,
            perf_counter() - started,
            {"task": "node_synthetic", "models": sorted(result)},
        )
    except Exception as exc:
        return _record(
            "node:synthetic",
            False,
            perf_counter() - started,
            {"task": "node_synthetic", "error": str(exc), "trace": traceback.format_exc()},
        )


def smoke_sg_synthetic() -> dict:
    config = yaml.safe_load(SG_SYNTHETIC.read_text())
    config["device"] = "cpu"
    config["training"]["epochs"] = 1
    config["training"]["probe_epochs"] = 5
    started = perf_counter()
    try:
        metrics = run_sg(config)
        return _record(
            "sg:synthetic",
            True,
            perf_counter() - started,
            {"task": "sg_synthetic", "metrics": metrics},
        )
    except Exception as exc:
        return _record(
            "sg:synthetic",
            False,
            perf_counter() - started,
            {"task": "sg_synthetic", "error": str(exc), "trace": traceback.format_exc()},
        )


def smoke_load_dblp() -> dict:
    started = perf_counter()
    try:
        graph = load_npz(DBLP_NPZ)
        if graph.labels is None:
            raise ValueError("DBLP node task requires labels")
        windows = list(graph.windows(3))
        synthetic = make_synthetic(num_nodes=32, num_snapshots=6, feature_dim=8, num_classes=3)
        detail = {
            "task": "load_node",
            **_graph_stats(graph),
            "windows": len(windows),
            "synthetic_nodes": synthetic.num_nodes,
        }
        return _record("load:dblp", True, perf_counter() - started, detail)
    except Exception as exc:
        return _record(
            "load:dblp",
            False,
            perf_counter() - started,
            {"task": "load_node", "error": str(exc), "trace": traceback.format_exc()},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test every packaged dataset")
    parser.add_argument("--skip-train", action="store_true", help="only load and validate archives")
    parser.add_argument("--max-positive-pairs", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "smoke_all_datasets.json")
    args = parser.parse_args()

    output_dir = args.output.parent / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    rows.extend(smoke_load_link(LINK_CONFIG))
    rows.append(smoke_load_dblp())
    if not args.skip_train:
        rows.append(smoke_sg_synthetic())
        rows.append(smoke_node_synthetic(output_dir))
        rows.extend(smoke_link_edgebank(LINK_CONFIG, output_dir, args.max_positive_pairs))

    report = {
        "ok": all(row["ok"] for row in rows),
        "passed": sum(row["ok"] for row in rows),
        "failed": sum(not row["ok"] for row in rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": {k: report[k] for k in ("ok", "passed", "failed")}}))
    if not report["ok"]:
        for row in rows:
            if not row["ok"]:
                print(json.dumps({"failed": row["name"], "error": row.get("error")}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
