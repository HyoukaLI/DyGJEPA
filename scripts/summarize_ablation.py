#!/usr/bin/env python3
"""Collect the DyGJEPA ablation results into one table.

Reads the full-model results of the main run (results/link_comparison_<ds>.json)
and every variant under results/ablation/<variant>/link_comparison_<ds>.json,
then prints test AP/AUC (x100, mean +- std over seeds) with the difference to
the full model.  Missing files are shown as "-" so the table can be built while
runs are still in flight.

    python scripts/summarize_ablation.py                       # markdown, all datasets
    python scripts/summarize_ablation.py --datasets wikipedia enron canparl
    python scripts/summarize_ablation.py --metric auc --format latex
    python scripts/summarize_ablation.py --split validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_DATASETS = ["wikipedia", "reddit", "mooc", "enron", "uci", "canparl"]
# Paper order: parameter-free start, then one row per removed module.
DEFAULT_VARIANTS = [
    "prior_only",
    "no_history",
    "no_signature",
    "no_subgraph",
    "no_trajectories",
    "no_jepa",
    "no_id",
]
LABELS = {
    "full": "DyGJEPA (full)",
    "prior_only": "recurrence prior only (epoch 0)",
    "no_history": "w/o historical context",
    "no_signature": "w/o path signatures",
    "no_subgraph": "w/o pair subgraph context",
    "no_trajectories": "w/o node trajectories",
    "no_jepa": "w/o JEPA objectives",
    "no_id": "w/o ID embeddings",
}


def _load_metric(
    path: Path, split: str, metric: str, model: str = "rcps_jepa"
) -> tuple[float, float, int] | None:
    """Return (mean, std, n_seeds) for one result file, or None if absent."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "aggregate" in data:
        entry = data["aggregate"].get(model, {}).get(split, {}).get(metric)
        if entry is None:
            return None
        if isinstance(entry, dict):
            return float(entry["mean"]), float(entry["std"]), len(data.get("seeds", []))
        return float(entry), 0.0, 1
    entry = data.get(model, {}).get(split, {}).get(metric)
    if entry is None:
        return None
    return float(entry), 0.0, 1


def _cell(
    value: tuple[float, float, int] | None,
    full: tuple[float, float, int] | None,
    fmt: str,
    scale: float = 100.0,
) -> str:
    if value is None:
        return "-"
    mean, std, seeds = value
    pm = r"\pm" if fmt == "latex" else "±"
    text = f"{scale * mean:.2f} {pm} {scale * std:.2f}"
    if full is not None and value is not full:
        delta = scale * (mean - full[0])
        text += f" ({delta:+.2f})"
    if seeds and seeds != 5:
        text += f" [n={seeds}]"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--metric", default="ap", choices=["ap", "auc", "mrr", "mean_probability", "best_epoch"])
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    args = parser.parse_args()

    scale = 1.0 if args.metric == "best_epoch" else 100.0
    rows: list[tuple[str, list[str]]] = []
    full_values = {
        dataset: _load_metric(
            args.results / f"link_comparison_{dataset}.json", args.split, args.metric
        )
        for dataset in args.datasets
    }
    rows.append(
        (
            LABELS["full"],
            [_cell(full_values[d], full_values[d], args.format, scale) for d in args.datasets],
        )
    )
    for variant in args.variants:
        cells = []
        for dataset in args.datasets:
            value = _load_metric(
                args.results / "ablation" / variant / f"link_comparison_{dataset}.json",
                args.split,
                args.metric,
            )
            cells.append(_cell(value, full_values[dataset], args.format, scale))
        rows.append((LABELS.get(variant, variant), cells))

    unit = "" if scale == 1.0 else "x100, "
    title = f"{args.split} {args.metric.upper()} ({unit}mean ± std over seeds; Δ vs full in parentheses)"
    if args.format == "markdown":
        print(f"{title}\n")
        print("| Variant | " + " | ".join(args.datasets) + " |")
        print("|---|" + "---|" * len(args.datasets))
        for label, cells in rows:
            print(f"| {label} | " + " | ".join(cells) + " |")
    else:
        print(f"% {title}")
        print(r"\begin{tabular}{l" + "c" * len(args.datasets) + "}")
        print(r"\toprule")
        print("Variant & " + " & ".join(args.datasets) + r" \\")
        print(r"\midrule")
        for label, cells in rows:
            print(f"{label} & " + " & ".join(f"${c}$" if c != "-" else "-" for c in cells) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


if __name__ == "__main__":
    main()
