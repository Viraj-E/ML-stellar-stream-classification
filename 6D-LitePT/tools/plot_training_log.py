#!/usr/bin/env python3
"""
Summarise and plot LitePT pipe-delimited training logs.

Examples
--------
python plot_litept_log.py OUTS/base_adam_fullps_ftnmt_aug_memfix
python plot_litept_log.py OUTS/base_adam_fullps_ftnmt_aug_memfix/train_log_copy_on_15-06-2026::21h-37m-18.dat
python plot_litept_log.py OUTS/base_adam_fullps_ftnmt_aug_memfix --watch 60 --out live_metrics.png

The default directory mode builds the active lineage: rows from the newest log
plus older rows before the newest log's first epoch. This avoids showing stale
later epochs from an aborted run after resuming from a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METRIC_COLUMNS = [
    ("epoch", "Epoch", "{:.0f}"),
    ("train_loss", "Train loss", "{:.4f}"),
    ("train_iou", "Train IoU", "{:.4f}"),
    ("train_miou", "Train mIoU", "{:.4f}"),
    ("train_mcc", "Train MCC", "{:.4f}"),
    ("val_loss", "Val loss", "{:.4f}"),
    ("val_iou", "Val IoU @0.5", "{:.4f}"),
    ("val_miou", "Val mIoU @0.5", "{:.4f}"),
    ("best_val_iou_swept", "Best swept Val IoU", "{:.4f} @ {:.3g}"),
    ("val_mcc", "Val MCC @0.5", "{:.4f}"),
]

THRESHOLD_RE = re.compile(r"^(?P<split>train|val)_thr(?P<thr>[0-9]+p[0-9]+)_(?P<metric>.+)$")


@dataclass
class LogData:
    path: Path
    rows: list[dict[str, object]]
    mtime: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a compact LitePT epoch table and save multi-panel metric plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more train_log*.dat files, or a run directory containing train_log*.dat files.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path. Defaults to <log_dir>/litept_metrics.png for a directory, or next to the log file.",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default=None,
        help="Output format. Inferred from --out when omitted.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster outputs.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh every N seconds. Use 0 for a single run.",
    )
    parser.add_argument(
        "--merge-mode",
        choices=["lineage", "latest", "concat"],
        default="lineage",
        help=(
            "How to combine multiple logs. lineage uses the newest log plus older epochs before it; "
            "latest keeps the newest row per epoch; concat keeps all rows."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print the tables and summaries.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive matplotlib window after saving.",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="Only print the last N epoch rows in the compact table. Use 0 for all rows.",
    )
    return parser.parse_args()


def to_number(value: str) -> object:
    value = value.strip()
    if value == "":
        return math.nan
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and number.is_integer() and "e" not in value.lower() and "." not in value:
        return int(number)
    return number


def as_float(value: object, default: float = math.nan) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: int = -1) -> int:
    val = as_float(value)
    if math.isnan(val):
        return default
    return int(val)


def discover_logs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            paths.extend(sorted(path.glob("train_log*.dat"), key=lambda p: (p.stat().st_mtime, p.name)))
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(f"Input does not exist: {path}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def read_log(path: Path) -> LogData:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        try:
            header = next(reader)
        except StopIteration:
            return LogData(path=path, rows=[], mtime=path.stat().st_mtime)

        expected = len(header)
        for line_no, values in enumerate(reader, start=2):
            if len(values) != expected:
                print(
                    f"Warning: skipping partial/malformed row {line_no} in {path} "
                    f"({len(values)} fields, expected {expected})",
                    file=sys.stderr,
                )
                continue
            row = {key: to_number(value) for key, value in zip(header, values)}
            rows.append(row)
    return LogData(path=path, rows=rows, mtime=path.stat().st_mtime)


def combine_logs(logs: list[LogData], mode: str) -> list[dict[str, object]]:
    non_empty = [log for log in logs if log.rows]
    if not non_empty:
        return []

    ordered = sorted(non_empty, key=lambda log: (log.mtime, str(log.path)))

    if mode == "concat":
        rows = []
        for log in ordered:
            rows.extend(log.rows)
        return sorted(rows, key=lambda row: as_float(row.get("epoch")))

    if mode == "latest":
        by_epoch: dict[int, dict[str, object]] = {}
        for log in ordered:
            for row in log.rows:
                by_epoch[as_int(row.get("epoch"))] = row
        return [by_epoch[k] for k in sorted(by_epoch)]

    newest = ordered[-1]
    first_new_epoch = min(as_int(row.get("epoch")) for row in newest.rows)
    by_epoch: dict[int, dict[str, object]] = {}
    for log in ordered[:-1]:
        for row in log.rows:
            epoch = as_int(row.get("epoch"))
            if epoch < first_new_epoch:
                by_epoch[epoch] = row
    for row in newest.rows:
        by_epoch[as_int(row.get("epoch"))] = row
    return [by_epoch[k] for k in sorted(by_epoch)]


def threshold_from_name(raw: str) -> float:
    return float(raw.replace("p", "."))


def threshold_columns(rows: list[dict[str, object]], split: str, metric: str) -> list[tuple[float, str]]:
    if not rows:
        return []
    out: list[tuple[float, str]] = []
    for key in rows[0].keys():
        match = THRESHOLD_RE.match(key)
        if not match:
            continue
        if match.group("split") == split and match.group("metric") == metric:
            out.append((threshold_from_name(match.group("thr")), key))
    return sorted(out)


def best_swept_metric(row: dict[str, object], split: str, metric: str) -> tuple[float, float]:
    best_value = math.nan
    best_thr = math.nan
    for key, value in row.items():
        match = THRESHOLD_RE.match(key)
        if not match:
            continue
        if match.group("split") != split or match.group("metric") != metric:
            continue
        val = as_float(value)
        if math.isnan(val):
            continue
        if math.isnan(best_value) or val > best_value:
            best_value = val
            best_thr = threshold_from_name(match.group("thr"))
    return best_value, best_thr


def add_derived_columns(rows: list[dict[str, object]]) -> None:
    for row in rows:
        val_iou, val_iou_thr = best_swept_metric(row, "val", "iou")
        val_mcc, val_mcc_thr = best_swept_metric(row, "val", "mcc")
        train_iou, train_iou_thr = best_swept_metric(row, "train", "iou")
        row["best_val_iou_swept"] = val_iou
        row["best_val_iou_thr"] = val_iou_thr
        row["best_val_mcc_swept"] = val_mcc
        row["best_val_mcc_thr"] = val_mcc_thr
        row["best_train_iou_swept"] = train_iou
        row["best_train_iou_thr"] = train_iou_thr


def format_cell(row: dict[str, object], key: str, pattern: str) -> str:
    if key == "best_val_iou_swept":
        val = as_float(row.get(key))
        thr = as_float(row.get("best_val_iou_thr"))
        if math.isnan(val):
            return "-"
        return pattern.format(val, thr)
    val = as_float(row.get(key))
    if math.isnan(val):
        return "-"
    return pattern.format(val)


def make_epoch_table(rows: list[dict[str, object]], last: int = 0) -> str:
    view = rows[-last:] if last and last > 0 else rows
    headers = [label for _, label, _ in METRIC_COLUMNS]
    body = [
        [format_cell(row, key, pattern) for key, _, pattern in METRIC_COLUMNS]
        for row in view
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body)) if body else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_line(values: list[str]) -> str:
        return "| " + " | ".join(value.rjust(widths[i]) for i, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    return "\n".join([fmt_line(headers), sep, *(fmt_line(row) for row in body)])


def best_row(rows: list[dict[str, object]], key: str, maximize: bool = True) -> dict[str, object] | None:
    scored = [(as_float(row.get(key)), row) for row in rows]
    scored = [(value, row) for value, row in scored if not math.isnan(value)]
    if not scored:
        return None
    return (max if maximize else min)(scored, key=lambda item: item[0])[1]


def print_summary(rows: list[dict[str, object]], logs: list[LogData], mode: str, last: int = 0) -> None:
    loaded_rows = sum(len(log.rows) for log in logs)
    print(f"Loaded {loaded_rows} raw rows from {len(logs)} log file(s); merge mode: {mode}")
    for log in sorted(logs, key=lambda item: (item.mtime, str(item.path))):
        if log.rows:
            epochs = [as_int(row.get("epoch")) for row in log.rows]
            print(f"  {log.path}: {len(log.rows)} rows, epochs {min(epochs)}-{max(epochs)}")
        else:
            print(f"  {log.path}: 0 rows")
    print()
    print(make_epoch_table(rows, last=last))
    print()

    summaries = [
        ("Best Val mIoU @0.5", "val_miou", True, "{:.4f}"),
        ("Best Val MCC @0.5", "val_mcc", True, "{:.4f}"),
        ("Best Val IoU @0.5", "val_iou", True, "{:.4f}"),
        ("Best swept Val IoU", "best_val_iou_swept", True, "{:.4f} @ thr {:.3g}"),
        ("Best swept Val MCC", "best_val_mcc_swept", True, "{:.4f} @ thr {:.3g}"),
        ("Best Val loss", "val_loss", False, "{:.4f}"),
    ]
    print("Best metrics:")
    for label, key, maximize, pattern in summaries:
        row = best_row(rows, key, maximize=maximize)
        if row is None:
            continue
        value = as_float(row.get(key))
        epoch = as_int(row.get("epoch"))
        if key == "best_val_iou_swept":
            detail = pattern.format(value, as_float(row.get("best_val_iou_thr")))
        elif key == "best_val_mcc_swept":
            detail = pattern.format(value, as_float(row.get("best_val_mcc_thr")))
        else:
            detail = pattern.format(value)
        print(f"  {label}: {detail} at epoch {epoch}")

    latest = rows[-1]
    print()
    print(
        "Latest epoch: "
        f"{as_int(latest.get('epoch'))}, "
        f"train_loss={as_float(latest.get('train_loss')):.4f}, "
        f"train_iou={as_float(latest.get('train_iou')):.4f}, "
        f"train_miou={as_float(latest.get('train_miou')):.4f}, "
        f"val_loss={as_float(latest.get('val_loss')):.4f}, "
        f"val_iou@0.5={as_float(latest.get('val_iou')):.4f}, "
        f"val_miou@0.5={as_float(latest.get('val_miou')):.4f}, "
        f"best_swept_val_iou={as_float(latest.get('best_val_iou_swept')):.4f}"
        f"@{as_float(latest.get('best_val_iou_thr')):.3g}"
    )


def series(rows: list[dict[str, object]], key: str) -> list[float]:
    return [as_float(row.get(key)) for row in rows]


def plot_if_present(ax, epochs: list[float], rows: list[dict[str, object]], key: str, label: str, **kwargs) -> None:
    values = series(rows, key)
    if any(not math.isnan(value) for value in values):
        ax.plot(epochs, values, marker="o", linewidth=1.8, markersize=4, label=label, **kwargs)


def annotate_best(ax, epochs: list[float], rows: list[dict[str, object]], key: str, maximize: bool = True) -> None:
    row = best_row(rows, key, maximize=maximize)
    if row is None:
        return
    epoch = as_float(row.get("epoch"))
    value = as_float(row.get(key))
    if math.isnan(epoch) or math.isnan(value):
        return
    ax.scatter([epoch], [value], s=65, color="black", zorder=4)
    ax.annotate(
        f"e{int(epoch)} {value:.3f}",
        xy=(epoch, value),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color="black",
    )


def save_plots(rows: list[dict[str, object]], output: Path, fmt: str | None, dpi: int, show: bool) -> None:
    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not import matplotlib, skipping plots: {exc}", file=sys.stderr)
        return

    epochs = series(rows, "epoch")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(f"LitePT metrics: {output.stem}", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    plot_if_present(ax, epochs, rows, "train_loss", "train loss")
    plot_if_present(ax, epochs, rows, "val_loss", "val loss")
    annotate_best(ax, epochs, rows, "val_loss", maximize=False)
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    plot_if_present(ax, epochs, rows, "train_iou", "train IoU @0.5")
    plot_if_present(ax, epochs, rows, "val_iou", "val IoU @0.5")
    plot_if_present(ax, epochs, rows, "train_miou", "train mIoU @0.5", linestyle="--")
    plot_if_present(ax, epochs, rows, "val_miou", "val mIoU @0.5", linestyle="--")
    plot_if_present(ax, epochs, rows, "best_val_iou_swept", "best swept val IoU")
    annotate_best(ax, epochs, rows, "best_val_iou_swept", maximize=True)
    annotate_best(ax, epochs, rows, "val_miou", maximize=True)
    ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.0, label="mIoU target")
    ax.set_title("IoU and mIoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    plot_if_present(ax, epochs, rows, "train_mcc", "train MCC @0.5")
    plot_if_present(ax, epochs, rows, "val_mcc", "val MCC @0.5")
    plot_if_present(ax, epochs, rows, "best_val_mcc_swept", "best swept val MCC")
    annotate_best(ax, epochs, rows, "val_mcc", maximize=True)
    ax.set_title("MCC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MCC")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for key, label in [
        ("val_precision", "precision"),
        ("val_recall", "recall"),
        ("val_f1", "F1"),
        ("val_balanced_acc", "balanced acc"),
    ]:
        plot_if_present(ax, epochs, rows, key, label)
    ax.set_title("Validation @0.5")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_ylim(bottom=0, top=1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    for thr, key in threshold_columns(rows, "val", "iou"):
        ax.plot(epochs, series(rows, key), marker=".", linewidth=1.1, label=f"{thr:.3g}")
    ax.set_title("Validation IoU by Threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(title="thr", fontsize=7, ncol=4)

    ax = axes[2, 1]
    plot_if_present(ax, epochs, rows, "train_true_pos_frac", "train true pos frac")
    plot_if_present(ax, epochs, rows, "train_pred_pos_frac", "train pred pos frac")
    plot_if_present(ax, epochs, rows, "val_true_pos_frac", "val true pos frac")
    plot_if_present(ax, epochs, rows, "val_pred_pos_frac", "val pred pos frac")
    ax_lr = ax.twinx()
    plot_if_present(ax_lr, epochs, rows, "lr", "lr", color="black", linestyle="--")
    ax.set_title("Class Balance and LR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fraction")
    ax_lr.set_ylabel("LR")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines_lr, labels_lr = ax_lr.get_legend_handles_labels()
    ax.legend(lines + lines_lr, labels + labels_lr, fontsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"dpi": dpi}
    if fmt:
        save_kwargs["format"] = fmt
    fig.savefig(output, **save_kwargs)
    print(f"Saved plot: {output}")
    if show:
        plt.show()
    plt.close(fig)


def default_output_path(inputs: list[str], logs: list[LogData], requested: str | None) -> Path:
    if requested:
        return Path(requested).expanduser()
    first_input = Path(inputs[0]).expanduser()
    if first_input.is_dir():
        return first_input / "litept_metrics.png"
    if logs:
        return logs[0].path.with_name(logs[0].path.stem + "_metrics.png")
    return Path("litept_metrics.png")


def run_once(args: argparse.Namespace) -> int:
    paths = discover_logs(args.inputs)
    logs = [read_log(path) for path in paths]
    rows = combine_logs(logs, args.merge_mode)
    if not rows:
        print("No complete epoch rows found.", file=sys.stderr)
        return 1
    add_derived_columns(rows)
    print_summary(rows, logs, args.merge_mode, last=args.last)
    if not args.no_plot:
        output = default_output_path(args.inputs, logs, args.out)
        save_plots(rows, output, args.format, args.dpi, args.show)
    return 0


def main() -> int:
    args = parse_args()
    if args.watch and args.watch > 0:
        while True:
            os.system("clear")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            rc = run_once(args)
            print()
            print(f"Refreshing in {args.watch:g}s. Press Ctrl-C to stop.")
            try:
                time.sleep(args.watch)
            except KeyboardInterrupt:
                return rc
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
