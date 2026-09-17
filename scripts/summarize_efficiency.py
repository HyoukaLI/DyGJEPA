#!/usr/bin/env python3
"""Efficiency table and figures from the comparison result files.

Every model of a comparison run carries an ``efficiency`` record next to its
metrics (see ``jepa_compare/efficiency.py``).  This script turns the records
of one result tree into

* a table per dataset: parameters, seconds per training epoch, epochs and
  wall-clock to the selected checkpoint, total training time, test inference
  time, peak GPU memory and the test AP of the same run;
* optional figures: a DyG-Mamba-style bubble chart (AP vs. time to the best
  checkpoint, bubble area = parameters) and TPNet-style bars of inference
  time and peak memory relative to DyGJEPA.

    python scripts/summarize_efficiency.py --results results/efficiency
    python scripts/summarize_efficiency.py --results results/efficiency --datasets enron wikipedia --format latex
    python scripts/summarize_efficiency.py --results results/efficiency --csv results/efficiency/efficiency.csv
    python scripts/summarize_efficiency.py --results results/efficiency --plot   # writes <results>/figures/
    python scripts/summarize_efficiency.py --results results/efficiency --plot --figure-tag main \
        --models dygformer cawn tgn tgat graphmixer tcl cldg dvgmae rcps_jepa       # main-text subset

Result files may be single-seed (``{model: {validation, test, efficiency}}``)
or multi-seed (``{"seeds", "runs", "aggregate"}``); aggregates use the mean.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

MODEL_ORDER = [
    "jodie",
    "dyrep",
    "tgat",
    "tgn",
    "cawn",
    "tcl",
    "graphmixer",
    "dygformer",
    "edgebank",
    "cldg",
    "maskdgnn",
    "dvgmae",
    "rcps_jepa",
    "jodie_author",
]
LABELS = {
    "jodie": "JODIE",
    "jodie_author": "JODIE (author)",
    "dyrep": "DyRep",
    "tgat": "TGAT",
    "tgn": "TGN",
    "cawn": "CAWN",
    "tcl": "TCL",
    "graphmixer": "GraphMixer",
    "dygformer": "DyGFormer",
    "edgebank": "EdgeBank",
    "cldg": "CLDG",
    "maskdgnn": "MaskDGNN",
    "dvgmae": "DVGMAE",
    "rcps_jepa": "DyGJEPA",
}
# Colour by model family, fixed order (categorical slots 1-3 of the validated
# palette plus a neutral): ours, continuous-time baselines, snapshot SSL
# baselines, parameter-free heuristic.
FAMILY = {
    "rcps_jepa": "ours",
    "cldg": "snapshot",
    "maskdgnn": "snapshot",
    "dvgmae": "snapshot",
    "edgebank": "heuristic",
}
FAMILY_COLOR = {
    "ours": "#2a78d6",
    "event": "#eb6834",
    "snapshot": "#1baf7a",
    "heuristic": "#6f6e6a",
}
COLUMNS = [
    ("parameters", "Params (K)", 1e-3, "{:.0f}"),
    ("train_epoch_seconds_mean", "s / epoch", 1.0, "{:.1f}"),
    ("best_epoch", "Best ep.", 1.0, "{:.0f}"),
    ("time_to_best_seconds", "Time to best (s)", 1.0, "{:.0f}"),
    ("train_seconds_total", "Train total (s)", 1.0, "{:.0f}"),
    ("test_seconds", "Test infer. (s)", 1.0, "{:.1f}"),
    ("peak_memory_mb", "Peak mem (MB)", 1.0, "{:.0f}"),
    ("test_ap", "Test AP", 100.0, "{:.2f}"),
]


def _mean(value):
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return float(value)


def load_records(path: Path) -> dict[str, dict[str, float]]:
    """Return {model: {efficiency fields + best_epoch + test_ap}} for one file."""
    data = json.loads(path.read_text())
    models = data["aggregate"] if "aggregate" in data else data
    records: dict[str, dict[str, float]] = {}
    for model, entry in models.items():
        efficiency = entry.get("efficiency")
        if not efficiency:
            continue
        record = {key: _mean(value) for key, value in efficiency.items()}
        test = entry.get("test", {})
        record["best_epoch"] = _mean(test.get("best_epoch", 0.0))
        record["test_ap"] = _mean(test["ap"]) if "ap" in test else math.nan
        # Snapshot SSL baselines pretrain first; count that time as training.
        record["train_seconds_total"] = record.get("train_seconds_total", 0.0) + record.get(
            "pretrain_seconds_total", 0.0
        )
        records[model] = record
    return records


def _ordered(records: dict[str, dict[str, float]]) -> list[str]:
    known = [name for name in MODEL_ORDER if name in records]
    return known + sorted(name for name in records if name not in MODEL_ORDER)


def _fmt(record: dict[str, float], key: str, scale: float, pattern: str) -> str:
    value = record.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return pattern.format(value * scale)


def print_table(dataset: str, records: dict[str, dict[str, float]], fmt: str, relative_to: str | None) -> None:
    columns = list(COLUMNS)
    if relative_to and relative_to in records:
        reference = records[relative_to]
        for record in records.values():
            record["relative_test_seconds"] = record["test_seconds"] / max(reference["test_seconds"], 1e-9)
            record["relative_time_to_best"] = record["time_to_best_seconds"] / max(
                reference["time_to_best_seconds"], 1e-9
            )
        columns.insert(6, ("relative_test_seconds", f"Infer. / {LABELS.get(relative_to, relative_to)}", 1.0, "{:.1f}x"))
        columns.insert(4, ("relative_time_to_best", f"Time / {LABELS.get(relative_to, relative_to)}", 1.0, "{:.1f}x"))
    header = ["Model"] + [title for _, title, _, _ in columns]
    rows = [
        [LABELS.get(name, name)] + [_fmt(records[name], key, scale, pattern) for key, _, scale, pattern in columns]
        for name in _ordered(records)
    ]
    if fmt == "markdown":
        print(f"### {dataset}\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for row in rows:
            print("| " + " | ".join(row) + " |")
        print()
    else:
        print(f"% {dataset}")
        print(r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}")
        print(r"\toprule")
        print(" & ".join(header).replace("%", r"\%") + r" \\")
        print(r"\midrule")
        for row in rows:
            print(" & ".join(cell.replace("x", r"$\times$") if cell.endswith("x") else cell for cell in row) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}\n")


def plot_dataset(
    dataset: str,
    records: dict[str, dict[str, float]],
    out_dir: Path,
    relative_to: str,
    tag: str = "",
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - plotting is optional
        print("matplotlib is not installed; skipping figures")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    names = _ordered(records)
    colors = [FAMILY_COLOR[FAMILY.get(name, "event")] for name in names]

    # 1. Bubble chart: AP vs. time to the selected checkpoint, area = parameters.
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    params = [max(records[n].get("parameters", 0.0), 1.0) for n in names]
    max_params = max(params)
    for name, color, size in zip(names, colors, params):
        record = records[name]
        x = max(record["time_to_best_seconds"], 1e-3)
        y = 100 * record["test_ap"]
        area = 40 + 400 * size / max_params
        ax.scatter(x, y, s=area, color=color, alpha=0.55, edgecolors=color, linewidths=1)
        ax.annotate(LABELS.get(name, name), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("time to best checkpoint (s, log)")
    ax.set_ylabel("test AP (%)")
    ax.set_title(f"{dataset}: accuracy vs. training cost (area = parameters)", fontsize=9)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = out_dir / f"{dataset}_bubble{tag}.{suffix}"
        fig.savefig(path, dpi=200)
        written.append(path)
    plt.close(fig)

    # 2. Relative inference time and peak memory (bars, log scale).
    reference = records.get(relative_to)
    for key, title, filename in (
        ("test_seconds", "test inference time", "inference"),
        ("peak_memory_mb", "peak GPU memory", "memory"),
    ):
        values = [records[n].get(key) for n in names]
        if any(v is None for v in values):
            continue
        fig, ax = plt.subplots(figsize=(4.6, 2.8))
        base = reference[key] if reference and reference.get(key) else 1.0
        rel = [v / max(base, 1e-9) for v in values]
        ax.bar(range(len(names)), rel, color=colors, width=0.7)
        for index, value in enumerate(rel):
            ax.annotate(f"{value:.1f}x", (index, value), ha="center", va="bottom", fontsize=6, xytext=(0, 1), textcoords="offset points")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=45, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel(f"{title}\n(relative to {LABELS.get(relative_to, relative_to)})", fontsize=8)
        ax.set_title(dataset, fontsize=9)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        for suffix in ("pdf", "png"):
            path = out_dir / f"{dataset}_{filename}{tag}.{suffix}"
            fig.savefig(path, dpi=200)
            written.append(path)
        plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path("results/efficiency"))
    parser.add_argument("--datasets", nargs="+", default=None, help="default: every link_comparison_<ds>.json in --results")
    parser.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    parser.add_argument("--relative-to", default="rcps_jepa")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="restrict the table and figures to these models (default: every model in the file)",
    )
    parser.add_argument("--plot", action="store_true", help="write figures to <results>/figures/")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="also write every record (all datasets, all fields) to this CSV for your own plots",
    )
    parser.add_argument("--figure-tag", default="", help="suffix for figure filenames, e.g. main")
    args = parser.parse_args()

    if args.datasets:
        files = [(name, args.results / f"link_comparison_{name}.json") for name in args.datasets]
    else:
        files = sorted(
            (path.stem.replace("link_comparison_", ""), path)
            for path in args.results.glob("link_comparison_*.json")
            if path.stem != "link_comparison_all" and "_seed" not in path.stem
        )
    csv_rows: list[dict[str, float | str]] = []
    for dataset, path in files:
        if not path.exists():
            print(f"{dataset}: {path} not found")
            continue
        records = load_records(path)
        if args.models:
            missing = [name for name in args.models if name not in records]
            if missing:
                print(f"{dataset}: no records for {missing}")
            records = {name: records[name] for name in args.models if name in records}
        if not records:
            print(f"{dataset}: no efficiency records in {path} (older run?)")
            continue
        print_table(dataset, records, args.format, args.relative_to)
        for name in _ordered(records):
            csv_rows.append({"dataset": dataset, "model": name, **records[name]})
        if args.plot:
            tag = f"_{args.figure_tag}" if args.figure_tag else ""
            for written in plot_dataset(
                dataset, records, args.results / "figures", args.relative_to, tag
            ):
                print(f"wrote {written}")
    if args.csv is not None and csv_rows:
        fields = ["dataset", "model"] + sorted(
            {key for row in csv_rows for key in row} - {"dataset", "model"}
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"wrote {args.csv} ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
