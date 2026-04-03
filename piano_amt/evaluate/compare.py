"""
evaluate/compare.py — Cross-model results table.

Scans all runs/<run_name>/eval_<split>/summary_metrics.json files and
produces:
  1. A printed table (sorted by note_f1 descending — the primary metric).
  2. A CSV file for importing into your dissertation.
  3. A LaTeX table ready for copy-paste into dissertation.
  4. A grouped bar chart PNG comparing the main metrics.

Metric naming follows Hawthorne 2018a Table 1:
  "Note"              = onset + pitch match (note_f1)
  "Note w/ offset"    = + offset match      (note_with_offset_f1)
  "Note w/ off + vel" = + velocity match    (note_with_offset_vel_f1)
  "Frame"             = frame-level         (frame_f1)

Usage:
    python -m evaluate.compare \\
        --runs_dir /content/drive/MyDrive/piano_amt/runs \\
        --split    test \\
        --out_dir  /content/drive/MyDrive/piano_amt/comparison

    Or from a notebook:
        from evaluate.compare import compare_all_runs, print_comparison_table
        rows = compare_all_runs(runs_dir=RUNS_DIR, split='test')
        print_comparison_table(rows)
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
        List of dicts, one per run, sorted by note_f1 descending.
        Each dict has all keys from summary_metrics.json plus "run_name"
        and config keys.
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

    # Sort by note_f1 (primary metric), fall back to onset_f1 for old runs
    rows.sort(
        key=lambda r: r.get("note_f1", r.get("onset_f1", 0)),
        reverse=True,
    )
    return rows


# ---------------------------------------------------------------------------
# Printing — Primary table (paper-comparable)
# ---------------------------------------------------------------------------

# Columns for the primary comparison table (Hawthorne 2018a style)
_PRIMARY_COLS = [
    ("run_name",                     "Run",                 20),
    ("note_precision",               "Note P",               8),
    ("note_recall",                  "Note R",               8),
    ("note_f1",                      "Note F1",              8),
    ("note_with_offset_f1",          "N+Off F1",             9),
    ("note_with_offset_vel_f1",      "N+O+V F1",             9),
    ("frame_precision",              "Frame P",              8),
    ("frame_recall",                 "Frame R",              8),
    ("frame_f1",                     "Frame F1",             8),
]

# Columns for supplementary error analysis table
_SUPPLEMENTARY_COLS = [
    ("run_name",                     "Run",                 20),
    ("ea_offset_mae_ms",             "Off MAE(ms)",         11),
    ("ea_onset_mae_ms",              "On MAE(ms)",          10),
    ("ea_chord_completeness",        "Chord Comp",          10),
    ("ea_duplicate_note_rate",       "Dup Rate",             8),
    ("n_pred_notes",                 "Pred Notes",          10),
    ("n_gt_notes",                   "GT Notes",            10),
]

# Columns for training/run info
_INFO_COLS = [
    ("run_name",                     "Run",                 20),
    ("n_files",                      "Eval files",          10),
    ("cfg_epochs",                   "Epochs",               8),
    ("cfg_max_files",                "Train files",         10),
    ("cfg_model_complexity",         "Complexity",          10),
    ("model_parameters",             "Params",              12),
    ("eval_time_total_s",            "Eval time(s)",        12),
]


def _print_table(rows: List[Dict], cols: list, title: str) -> None:
    """Print a formatted ASCII table."""
    if not rows:
        print("No results to display.")
        return

    header = "  ".join(label.rjust(w) for _, label, w in cols)
    sep    = "-" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)

    for row in rows:
        cells = []
        for key, _, width in cols:
            val = row.get(key, None)
            if val is None:
                cells.append("—".rjust(width))
            elif isinstance(val, float):
                # Use different formats based on scale
                if "mae" in key.lower():
                    cells.append(f"{val:.1f}".rjust(width))
                elif val > 100:
                    cells.append(f"{val:.0f}".rjust(width))
                else:
                    cells.append(f"{val:.4f}".rjust(width))
            elif isinstance(val, int):
                cells.append(f"{val:,}".rjust(width))
            else:
                cells.append(str(val)[:width].rjust(width))
        print("  ".join(cells))

    print(sep + "\n")


def print_comparison_table(rows: List[Dict]) -> None:
    """
    Print all three comparison tables:
      1. Primary metrics (paper-comparable)
      2. Supplementary error analysis
      3. Run info
    """
    _print_table(rows, _PRIMARY_COLS,
                 "PRIMARY METRICS (Hawthorne 2018a Table 1 naming)")
    _print_table(rows, _SUPPLEMENTARY_COLS,
                 "SUPPLEMENTARY ERROR ANALYSIS (project-specific)")
    _print_table(rows, _INFO_COLS,
                 "RUN INFORMATION")


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
        # Reorder columns: run_name first, then primary metrics, then rest
        priority = ["run_name", "note_f1", "note_precision", "note_recall",
                     "note_with_offset_f1", "note_with_offset_vel_f1",
                     "frame_f1", "frame_precision", "frame_recall"]
        ordered = [c for c in priority if c in df.columns]
        rest    = [c for c in df.columns if c not in ordered]
        df = df[ordered + rest]
        df.to_csv(str(out_path), index=False)
        print(f"CSV saved → {out_path}")
    except ImportError:
        if not rows:
            return
        keys = list(rows[0].keys())
        with open(out_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")
        print(f"CSV saved → {out_path}")


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------

def save_latex_table(
    rows:      List[Dict],
    out_path:  Union[str, Path],
    caption:   str = "AMT evaluation results on MAESTRO v3.0.0",
    label:     str = "tab:amt_results",
) -> str:
    """
    Generate a LaTeX table matching Hawthorne 2018a Table 1 format.

    Saves to file and returns the LaTeX string.
    """
    if not rows:
        return ""

    metrics = [
        ("note_f1",                 "Note F1"),
        ("note_with_offset_f1",     "Note+Off F1"),
        ("note_with_offset_vel_f1", "N+O+V F1"),
        ("frame_f1",                "Frame F1"),
        ("ea_offset_mae_ms",        "Off MAE (ms)"),
        ("ea_chord_completeness",   "Chord Comp"),
    ]

    n_cols = len(metrics) + 1
    col_spec = "l" + "r" * len(metrics)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    # Header
    header = "    Model & " + " & ".join(label for _, label in metrics) + r" \\"
    lines.append(header)
    lines.append(r"    \midrule")

    # Data rows
    for row in rows:
        name = row.get("run_name", "?").replace("_", r"\_")
        vals = []
        for key, _ in metrics:
            v = row.get(key, None)
            if v is None:
                vals.append("—")
            elif "mae" in key.lower():
                vals.append(f"{v:.1f}")
            else:
                vals.append(f"{v:.3f}")
        lines.append(f"    {name} & " + " & ".join(vals) + r" \\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(out_path, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved → {out_path}")

    return latex


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

    Metrics shown: note_f1, note_with_offset_f1, note_with_offset_vel_f1, frame_f1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        print("No data to plot.")
        return

    metrics = ["note_f1", "note_with_offset_f1", "note_with_offset_vel_f1", "frame_f1"]
    labels  = ["Note F1", "Note+Off F1", "N+O+V F1", "Frame F1"]
    run_names = [r["run_name"] for r in rows]

    x      = np.arange(len(metrics))
    width  = 0.8 / max(len(rows), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(rows), 1)))

    fig, ax = plt.subplots(figsize=(max(10, len(metrics)*2.5), 5))

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
                        help="Directory to save CSV, PNG, LaTeX. "
                             "Defaults to runs_dir/comparison/")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_dir) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = compare_all_runs(args.runs_dir, split=args.split)
    if not rows:
        return

    print_comparison_table(rows)

    save_comparison_csv(rows, out_dir / f"comparison_{args.split}.csv")

    save_latex_table(
        rows,
        out_path=out_dir / f"comparison_{args.split}.tex",
        caption=f"AMT evaluation results on MAESTRO v3.0.0 ({args.split} split)",
    )

    plot_comparison_bar(
        rows,
        save_path=out_dir / f"comparison_{args.split}.png",
        title=f"Model comparison — MAESTRO v3.0.0 {args.split} split",
    )


if __name__ == "__main__":
    main()