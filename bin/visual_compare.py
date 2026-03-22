#!/usr/bin/env python3

"""Generate visual comparison report between FL and centralized training.

Accepts the model comparison JSON (from model_compare_run.py) and the
central snapshot tar. Produces a self-contained HTML report with:
  - FL vs CEN accuracy/loss bar charts per window/span
  - Convergence curve comparison (if training history available)
  - Per-site FL client performance breakdown
  - Precipitation prediction sample comparison (if model outputs exist)
  - Summary statistics table with significance indicators

Input:
  --config experiment_config.yaml
  --comparison model_comparison.json
  --central-tar central_snapshot.tar.gz
  --output visual_comparison.html

Output: Self-contained HTML with base64-embedded PNG charts.
"""

import argparse
import base64
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


def make_accuracy_comparison_chart(comparisons):
    """Bar chart comparing FL vs CEN accuracy per window/span."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    fl_accs = []
    cen_accs = []

    for comp in comparisons:
        label = f"{comp['window']}/\n{comp['span']}"
        labels.append(label)

        fl_m = comp.get("fl") or {}
        cen_m = comp.get("cen") or {}
        fl_accs.append(fl_m.get("accuracy") or 0)
        cen_accs.append(cen_m.get("accuracy") or 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))
    bars_fl = ax.bar(x - width / 2, fl_accs, width, label="Federated (FL)",
                     color="#2196F3", edgecolor="black", linewidth=0.5)
    bars_cen = ax.bar(x + width / 2, cen_accs, width, label="Centralized (CEN)",
                      color="#FF9800", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Window / Span", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("FL vs Centralized: Accuracy Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Add value labels
    for bars in [bars_fl, bars_cen]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_loss_comparison_chart(comparisons):
    """Bar chart comparing FL vs CEN loss per window/span."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = []
    fl_losses = []
    cen_losses = []

    for comp in comparisons:
        label = f"{comp['window']}/{comp['span']}"
        labels.append(label)

        fl_m = comp.get("fl") or {}
        cen_m = comp.get("cen") or {}
        fl_losses.append(fl_m.get("loss") or 0)
        cen_losses.append(cen_m.get("loss") or 0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2), 6))
    ax.bar(x - width / 2, fl_losses, width, label="Federated (FL)",
           color="#2196F3", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, cen_losses, width, label="Centralized (CEN)",
           color="#FF9800", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Window / Span", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("FL vs Centralized: Loss Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_convergence_chart(comparisons):
    """Line chart showing training convergence curves (FL vs CEN)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find comparisons with history data
    charts = []

    for comp in comparisons:
        fl_m = comp.get("fl") or {}
        cen_m = comp.get("cen") or {}
        fl_history = fl_m.get("history", [])
        cen_history = cen_m.get("history", [])

        if not fl_history and not cen_history:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        title = f"{comp['window']}/span_{comp['span']}"

        # Loss curves
        if fl_history:
            fl_losses = [r.get("loss") for r in fl_history if r.get("loss") is not None]
            if fl_losses:
                ax1.plot(range(1, len(fl_losses) + 1), fl_losses,
                         "b-o", markersize=3, label="FL", alpha=0.8)

        if cen_history:
            cen_losses = [r.get("loss") for r in cen_history if r.get("loss") is not None]
            if cen_losses:
                ax1.plot(range(1, len(cen_losses) + 1), cen_losses,
                         "r-s", markersize=3, label="CEN", alpha=0.8)

        ax1.set_xlabel("Round / Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.set_title(f"Loss Convergence — {title}", fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Accuracy curves
        if fl_history:
            fl_accs = [r.get("accuracy") for r in fl_history if r.get("accuracy") is not None]
            if fl_accs:
                ax2.plot(range(1, len(fl_accs) + 1), fl_accs,
                         "b-o", markersize=3, label="FL", alpha=0.8)

        if cen_history:
            cen_accs = [r.get("accuracy") for r in cen_history if r.get("accuracy") is not None]
            if cen_accs:
                ax2.plot(range(1, len(cen_accs) + 1), cen_accs,
                         "r-s", markersize=3, label="CEN", alpha=0.8)

        ax2.set_xlabel("Round / Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.set_title(f"Accuracy Convergence — {title}", fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        charts.append((title, fig_to_base64(fig)))
        plt.close(fig)

    return charts


def make_per_site_chart(comparisons, sites):
    """Bar chart showing per-site FL client accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate per-site metrics across all comparisons
    site_accs = {site: [] for site in sites}

    for comp in comparisons:
        per_site = comp.get("per_site_fl", {})
        for site in sites:
            site_m = per_site.get(site, {})
            acc = site_m.get("accuracy")
            if acc is not None:
                site_accs[site].append(acc)

    # Check if we have any data
    has_data = any(accs for accs in site_accs.values())
    if not has_data:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    site_names = sorted(sites)
    means = [np.mean(site_accs[s]) if site_accs[s] else 0 for s in site_names]
    stds = [np.std(site_accs[s]) if len(site_accs[s]) > 1 else 0 for s in site_names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(site_names)))
    bars = ax.bar(site_names, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Radar Site", fontsize=12)
    ax.set_ylabel("Mean Accuracy", fontsize=12)
    ax.set_title("Per-Site FL Client Accuracy", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    for bar, mean in zip(bars, means):
        if mean > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_delta_heatmap(comparisons):
    """Heatmap showing metric deltas (FL - CEN) per window/span."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{c['window']}/{c['span']}" for c in comparisons]
    metrics = ["accuracy", "loss", "f1"]
    metric_labels = ["Accuracy (FL-CEN)", "Loss (FL-CEN)", "F1 (FL-CEN)"]

    data = np.full((len(comparisons), len(metrics)), np.nan)
    for i, comp in enumerate(comparisons):
        delta = comp.get("delta", {})
        for j, metric in enumerate(metrics):
            if metric in delta:
                data[i, j] = delta[metric]

    # Skip if all NaN
    if np.all(np.isnan(data)):
        return None

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.6 + 1)))

    # Mask NaN values
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto",
                   vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Metric Deltas (FL - Centralized)", fontsize=14)

    for i in range(len(labels)):
        for j in range(len(metrics)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.25 else "black"
                ax.text(j, i, f"{val:+.4f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="gray")

    fig.colorbar(im, ax=ax, label="Delta", shrink=0.8)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def build_html(comparisons, summary, charts, timestamp, sites):
    """Build the self-contained HTML comparison report."""
    acc_chart = charts.get("accuracy", "")
    loss_chart = charts.get("loss", "")
    per_site_chart = charts.get("per_site", "")
    delta_heatmap = charts.get("delta_heatmap", "")
    convergence_charts = charts.get("convergence", [])

    # Summary metrics
    mean_fl = summary.get("mean_fl_accuracy")
    mean_cen = summary.get("mean_cen_accuracy")
    fl_wins = summary.get("fl_wins", 0)
    cen_wins = summary.get("cen_wins", 0)
    p_value = summary.get("ttest_pvalue")
    significant = summary.get("significant_difference", False)

    sig_text = ""
    if p_value is not None:
        sig_indicator = "Yes (p < 0.05)" if significant else "No (p >= 0.05)"
        sig_text = f"""
        <div class="metric">
            <div class="value">{p_value:.4f}</div>
            <div class="label">t-test p-value ({sig_indicator})</div>
        </div>"""

    # Comparison table rows
    table_rows = ""
    for comp in comparisons:
        fl_m = comp.get("fl") or {}
        cen_m = comp.get("cen") or {}
        delta = comp.get("delta", {})

        fl_acc = f"{fl_m['accuracy']:.4f}" if fl_m.get("accuracy") is not None else "N/A"
        cen_acc = f"{cen_m['accuracy']:.4f}" if cen_m.get("accuracy") is not None else "N/A"
        fl_loss = f"{fl_m['loss']:.4f}" if fl_m.get("loss") is not None else "N/A"
        cen_loss = f"{cen_m['loss']:.4f}" if cen_m.get("loss") is not None else "N/A"
        delta_acc = f"{delta['accuracy']:+.4f}" if "accuracy" in delta else "N/A"

        winner = ""
        if "accuracy" in delta:
            if delta["accuracy"] > 0.001:
                winner = '<span style="color:blue">FL</span>'
            elif delta["accuracy"] < -0.001:
                winner = '<span style="color:orange">CEN</span>'
            else:
                winner = "Tie"

        table_rows += f"""<tr>
            <td>{comp['window']}</td>
            <td>{comp['span']}</td>
            <td>{fl_acc}</td>
            <td>{cen_acc}</td>
            <td>{delta_acc}</td>
            <td>{fl_loss}</td>
            <td>{cen_loss}</td>
            <td>{winner}</td>
        </tr>\n"""

    # Convergence sections
    convergence_html = ""
    for title, img_b64 in convergence_charts:
        convergence_html += f"""
        <div class="card">
            <h3>Convergence — {title}</h3>
            <img src="data:image/png;base64,{img_b64}" alt="Convergence {title}">
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPRITE FL vs Centralized Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{ color: #1a237e; border-bottom: 3px solid #3f51b5; padding-bottom: 10px; }}
        h2 {{ color: #283593; margin-top: 40px; }}
        .card {{
            background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card img {{ max-width: 100%; height: auto; display: block; margin: 10px auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: center; }}
        th {{ background: #3f51b5; color: white; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .summary-grid {{
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px; margin: 20px 0;
        }}
        .metric {{
            background: white; border-radius: 8px; padding: 20px; text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric .value {{ font-size: 1.8em; font-weight: bold; color: #3f51b5; }}
        .metric .label {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
        .timestamp {{ color: #999; font-size: 0.85em; }}
        .winner-fl {{ color: #2196F3; font-weight: bold; }}
        .winner-cen {{ color: #FF9800; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>FL vs Centralized Training Comparison</h1>
    <p class="timestamp">Generated: {timestamp} &mdash; Sites: {', '.join(sorted(sites))}</p>

    <div class="summary-grid">
        <div class="metric">
            <div class="value">{f'{mean_fl:.4f}' if mean_fl is not None else 'N/A'}</div>
            <div class="label">Mean FL Accuracy</div>
        </div>
        <div class="metric">
            <div class="value">{f'{mean_cen:.4f}' if mean_cen is not None else 'N/A'}</div>
            <div class="label">Mean CEN Accuracy</div>
        </div>
        <div class="metric">
            <div class="value">{fl_wins}</div>
            <div class="label">FL Wins</div>
        </div>
        <div class="metric">
            <div class="value">{cen_wins}</div>
            <div class="label">CEN Wins</div>
        </div>
        {sig_text}
    </div>

    <h2>Accuracy Comparison</h2>
    <div class="card">
        {'<img src="data:image/png;base64,' + acc_chart + '" alt="Accuracy comparison">' if acc_chart else '<p>No accuracy data available.</p>'}
    </div>

    <h2>Loss Comparison</h2>
    <div class="card">
        {'<img src="data:image/png;base64,' + loss_chart + '" alt="Loss comparison">' if loss_chart else '<p>No loss data available.</p>'}
    </div>

    <h2>Metric Deltas (FL - Centralized)</h2>
    <div class="card">
        {'<img src="data:image/png;base64,' + delta_heatmap + '" alt="Delta heatmap">' if delta_heatmap else '<p>No delta data available.</p>'}
    </div>

    <h2>Detailed Results</h2>
    <div class="card">
        <table>
            <tr>
                <th>Window</th><th>Span</th>
                <th>FL Acc</th><th>CEN Acc</th><th>Delta</th>
                <th>FL Loss</th><th>CEN Loss</th><th>Winner</th>
            </tr>
            {table_rows}
        </table>
    </div>

    {'<h2>Per-Site FL Performance</h2><div class="card"><img src="data:image/png;base64,' + per_site_chart + '" alt="Per-site accuracy"></div>' if per_site_chart else ''}

    {'<h2>Convergence Curves</h2>' + convergence_html if convergence_html else ''}

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual comparison report (FL vs Centralized)"
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--comparison", required=True,
                        help="Input model comparison JSON (from model_compare_run)")
    parser.add_argument("--central-tar", required=True,
                        help="Input central snapshot tar.gz archive")
    parser.add_argument("--output", required=True, help="Output HTML report file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.comparison, "r") as f:
        comparison_data = json.load(f)

    sites = config.get("sites", [])
    comparisons = comparison_data.get("comparisons", [])
    summary = comparison_data.get("summary", {})
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    logger.info(f"Generating charts for {len(comparisons)} comparisons")

    # Generate all charts
    charts = {}

    logger.info("Generating accuracy comparison chart...")
    charts["accuracy"] = make_accuracy_comparison_chart(comparisons)

    logger.info("Generating loss comparison chart...")
    charts["loss"] = make_loss_comparison_chart(comparisons)

    logger.info("Generating convergence curves...")
    charts["convergence"] = make_convergence_chart(comparisons)

    logger.info("Generating per-site chart...")
    charts["per_site"] = make_per_site_chart(comparisons, sites)

    logger.info("Generating delta heatmap...")
    charts["delta_heatmap"] = make_delta_heatmap(comparisons)

    # Build HTML
    html = build_html(comparisons, summary, charts, now, sites)

    with open(args.output, "w") as f:
        f.write(html)

    logger.info(f"Visual comparison report written: {args.output}")


if __name__ == "__main__":
    main()
