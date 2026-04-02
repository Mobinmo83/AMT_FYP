"""
evaluate/compare.py — Cross-model results table.

Scans all runs/<run_name>/eval_<split>/summary_metrics.json files and
produces:
  1. A printed table (sorted by onset_f1 descending).
  2. A CSV file for importing into your dissertation.
  3. A grouped bar chart PNG comparing the main metrics.

Usage:
    python -m evaluate.compare \\
        --runs_dir /content/drive/MyDrive/piano_amt/runs \\
        --split    test \\
        --out_dir  /content/drive/MyDrive/piano_amt/comparison

    Or from a notebook:
        from evaluate.compare import compare_all_runs
        df = compare_all_runs(runs_dir=RUNS_DIR, split='test')
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Core comparison function
# ---------------------------------------------------------------------------

def compare_all_runs(
    runs_dir: Union[str, Path],
    split:    str = "test",
) -> List[Dict]:
    """
    Scan all runs under runs_dir and collect their summary metrics
    plus model config info.

    Looks for: runs/<run_name>/eval_<split>/summary_metrics.json
    Also loads: runs/<run_name>/config.json (if available)

    Args:
        runs_dir: Parent directory of all run directories.
        split:    Evaluation split name ("test", "validation").

    Returns:
        List of dicts, one per run, sorted by onset_f1 descending.
        Each dict has all keys from summary_metrics.json plus "run_name"
        and config keys (lr, batch_size, max_files, epochs, model_complexity).
    """
    runs_dir = Path(runs_dir)
    pattern  = f"*/eval_{split}/summary_metrics.json"
    files    = sorted(runs_dir.glob(pattern))

    if not files:
        print(f"No summary_metrics.json found under {runs_dir}/{pattern}")
        return []

    rows: List[Dict] = []
    for mf in files:
        run_name = mf.parent.parent.name
        with open(mf) as f:
            data = json.load(f)
        data["run_name"] = run_name

        # Load config.json for model/training info
        config_path = mf.parent.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            data["cfg_lr"]               = cfg.get("lr", None)
            data["cfg_batch_size"]       = cfg.get("batch_size", None)
            data["cfg_max_files"]        = cfg.get("max_files", None)
            data["cfg_epochs"]           = cfg.get("epochs", None)
            data["cfg_model_complexity"] = cfg.get("model_complexity", None)

        rows.append(data)

    rows.sort(key=lambda r: r.get("onset_f1", 0), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_DISPLAY_COLS = [
    ("run_name",                     "Run",              18),
    ("onset_precision",              "Onset P",          10),
    ("onset_recall",                 "Onset R",          10),
    ("onset_f1",                     "Onset F1",         10),
    ("frame_f1",                     "Frame F1",         10),
    ("note_with_offset_f1",          "Note+Off F1",      12),
    ("note_with_offset_vel_f1",      "N+O+V F1",         10),
    ("ea_offset_mae_ms",             "Off MAE(ms)",      12),
    ("ea_chord_completeness",        "Chord Comp",       11),
    ("n_files",                      "N files",           8),
    ("cfg_epochs",                   "Epochs",            8),
    ("cfg_max_files",                "Train files",      11),
]


def print_comparison_table(rows: List[Dict]) -> None:
    """Print a formatted ASCII table of comparison results."""
    if not rows:
        print("No results to display.")
        return

    # Header
    header = "  ".join(label.ljust(w) for _, label, w in _DISPLAY_COLS)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for row in rows:
        cells = []
        for key, _, width in _DISPLAY_COLS:
            val = row.get(key, "—")
            if val is None:
                cells.append("—".ljust(width))
            elif isinstance(val, float):
                cells.append(f"{val:.4f}".ljust(width))
            else:
                cells.append(str(val).ljust(width))
        print("  ".join(cells))

    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_comparison_csv(
    rows:     List[Dict],
    out_path: Union[str, Path],
) -> None:
    """Save comparison results to a CSV file."""
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(str(out_path), index=False)
        print(f"CSV saved → {out_path}")
    except ImportError:
        # Manual fallback
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(out_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# Grouped bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    rows:      List[Dict],
    save_path: Optional[Union[str, Path]] = None,
    title:     str = "Model comparison",
) -> None:
    """
    Grouped bar chart comparing main metrics across all runs.

    Metrics shown: onset_f1, frame_f1, note_with_offset_f1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        print("No data to plot.")
        return

    metrics = ["onset_f1", "frame_f1", "note_with_offset_f1"]
    labels  = ["Onset F1", "Frame F1", "Note+Offset F1"]
    run_names = [r["run_name"] for r in rows]

    x      = np.arange(len(metrics))
    width  = 0.8 / len(rows)
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))

    fig, ax = plt.subplots(figsize=(max(8, len(metrics)*2), 5))

    for i, (row, color) in enumerate(zip(rows, colors)):
        vals   = [row.get(m, 0) for m in metrics]
        offset = (i - len(rows)/2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width, label=row["run_name"],
                        color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare all AMT runs.")
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--split",    default="test")
    parser.add_argument("--out_dir",  default=None,
                        help="Directory to save CSV, PNG. "
                             "Defaults to runs_dir/comparison/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_dir) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = compare_all_runs(args.runs_dir, split=args.split)
    if not rows:
        return

    print_comparison_table(rows)

    save_comparison_csv(rows, out_dir / f"comparison_{args.split}.csv")

    plot_comparison_bar(
        rows,
        save_path=out_dir / f"comparison_{args.split}.png",
        title=f"Model comparison ({args.split})",
    )


if __name__ == "__main__":
    main()