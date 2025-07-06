"""Quick analysis of benchmark metrics.

Reads the CSV produced by *scripts/run_benchmark.py*, computes aggregated
statistics per mode, generates basic plots, and writes a self-contained HTML
report.

Usage
-----
    python analysis/inspect_results.py \
        --metrics bench_results/metrics.csv \
        --out_dir analysis/reports

Dependencies: pandas, matplotlib, jinja2 (optional, only for prettier HTML).
"""
from __future__ import annotations

import argparse
import base64
import io
import textwrap
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:  # noqa: D401
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _summary_table(df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
    agg_funcs = ["mean", "median", "std", "min", "max"]
    summary = df.groupby("mode")["best_reward"].agg(agg_funcs)
    summary["count"] = df.groupby("mode").size()
    return summary.reset_index()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Analyse benchmark metrics CSV")
    parser.add_argument("--metrics", type=Path, default=Path("bench_results/metrics.csv"))
    parser.add_argument("--out_dir", type=Path, default=Path("analysis/reports"))
    args = parser.parse_args()

    if not args.metrics.exists():
        raise FileNotFoundError(args.metrics)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    df = pd.read_csv(args.metrics)

    # ------------------------------------------------------------------
    # Summary stats
    summary = _summary_table(df)
    summary_path = args.out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary table to {summary_path}")

    # ------------------------------------------------------------------
    # Plot distribution per mode
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, grp in df.groupby("mode"):
        ax.hist(grp["best_reward"], bins=20, alpha=0.5, label=mode)
    ax.set_xlabel("Best reward")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    hist_b64 = _fig_to_base64(fig)

    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    df.boxplot(column="best_reward", by="mode", ax=ax2)
    ax2.set_title("Reward distribution by mode")
    ax2.set_ylabel("Best reward")
    fig2.suptitle("")
    box_b64 = _fig_to_base64(fig2)

    # ------------------------------------------------------------------
    # HTML report
    html_lines: list[str] = []
    html_lines.append("<html><head><meta charset='utf-8'><title>Benchmark Report</title></head><body>")
    html_lines.append("<h1>HyperCubeX-EAN – Benchmark Report</h1>")

    html_lines.append("<h2>Aggregated statistics</h2>")
    html_lines.append(summary.to_html(index=False, float_format="{:.4f}".format))

    html_lines.append("<h2>Histograms</h2>")
    html_lines.append(f"<img src='data:image/png;base64,{hist_b64}'/>")

    html_lines.append("<h2>Boxplot</h2>")
    html_lines.append(f"<img src='data:image/png;base64,{box_b64}'/>")

    # Top / bottom tasks
    html_lines.append("<h2>Top 10 Controller vs Heuristic</h2>")
    if set(df["mode"]) >= {"heuristic", "controller"}:
        pivot = df.pivot(index="task_id", columns="mode", values="best_reward")
        pivot["diff"] = pivot["controller"] - pivot["heuristic"]
        top10 = pivot.sort_values("diff", ascending=False).head(10)
        html_lines.append(top10.to_html(float_format="{:.4f}".format))
    else:
        html_lines.append("<p>Modes heuristic & controller both required.</p>")

    html_lines.append("</body></html>")
    report_path = args.out_dir / "report.html"
    report_path.write_text("\n".join(html_lines))
    print(f"Saved HTML report to {report_path}")


if __name__ == "__main__":
    main()
