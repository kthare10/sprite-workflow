#!/usr/bin/env python3

"""Generate visual summary report for the SPRITE pipeline.

Produces an HTML report containing:
  - Pipeline summary: sequence counts per site (bar chart),
    data completeness overview, DAG structure diagram
  - Precipitation sample maps: spatial heatmaps from sample .nc files
    for each site

Accepts the central snapshot tar, FL configs tar, and experiment config.
Outputs a self-contained HTML file with embedded PNG images.
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
from collections import defaultdict
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


def make_sequence_bar_chart(site_seq_counts):
    """Create a bar chart of sequence counts per site."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = sorted(site_seq_counts.keys())
    counts = [site_seq_counts[s] for s in sites]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sites)))
    bars = ax.bar(sites, counts, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Radar Site", fontsize=12)
    ax.set_ylabel("Sequence Count", fontsize=12)
    ax.set_title("Sequences per Site in Central Snapshot", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_completeness_heatmap(site_split_counts):
    """Create a heatmap showing data split distribution per site."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = sorted(site_split_counts.keys())
    splits = sorted({s for counts in site_split_counts.values() for s in counts})

    data = np.zeros((len(sites), len(splits)))
    for i, site in enumerate(sites):
        for j, split in enumerate(splits):
            data[i, j] = site_split_counts[site].get(split, 0)

    fig, ax = plt.subplots(figsize=(8, max(4, len(sites) * 0.6 + 1)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits, fontsize=11)
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels(sites, fontsize=11)
    ax.set_title("Sequence Count by Site and Split", fontsize=14)

    for i in range(len(sites)):
        for j in range(len(splits)):
            val = int(data[i, j])
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > data.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, label="Count", shrink=0.8)
    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_dag_diagram(sites):
    """Create a visual DAG diagram of the pipeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, max(6, len(sites) * 0.8 + 2)))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1, len(sites) + 1)
    ax.axis("off")
    ax.set_title("SPRITE FL Workflow DAG", fontsize=16, fontweight="bold", pad=20)

    # Colors for stages
    stage_colors = {
        "download": "#4CAF50",
        "preproc": "#2196F3",
        "snapshot": "#FF9800",
        "central": "#9C27B0",
        "configs": "#F44336",
        "report": "#607D8B",
    }

    box_w, box_h = 0.9, 0.5

    def draw_box(x, y, label, color, fontsize=8):
        rect = mpatches.FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.05", facecolor=color, edgecolor="black",
            linewidth=1.2, alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2 - box_w / 2, y2), xytext=(x1 + box_w / 2, y1),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    # Per-site rows
    for i, site in enumerate(sites):
        y = len(sites) - i - 0.5
        draw_box(0.5, y, f"download\n{site}", stage_colors["download"], 7)
        draw_box(1.8, y, f"preproc\n{site}", stage_colors["preproc"], 7)
        draw_box(3.1, y, f"snapshot\n{site}", stage_colors["snapshot"], 7)
        draw_arrow(0.5, y, 1.8, y)
        draw_arrow(1.8, y, 3.1, y)

    # Fan-in nodes
    mid_y = (len(sites) - 1) / 2
    draw_box(4.4, mid_y, "central\nsnapshot", stage_colors["central"], 7)
    draw_box(5.3, mid_y, "prepare\nconfigs", stage_colors["configs"], 7)
    draw_box(6.2, mid_y, "finalize\nreport", stage_colors["report"], 7)

    # Arrows from snapshots to central
    for i in range(len(sites)):
        y = len(sites) - i - 0.5
        draw_arrow(3.1, y, 4.4, mid_y)

    draw_arrow(4.4, mid_y, 5.3, mid_y)
    draw_arrow(5.3, mid_y, 6.2, mid_y)

    plt.tight_layout()
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def make_precip_maps(central_dir, sites, max_samples=1):
    """Create spatial precipitation heatmaps from sample .nc files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    maps = {}
    for site in sorted(sites):
        # Find a sample .nc file for this site in the central snapshot
        nc_path = None
        for root, dirs, files in os.walk(central_dir):
            for f in files:
                if f.endswith(".nc") and site.lower() in root.lower():
                    nc_path = os.path.join(root, f)
                    break
                # Also check for __SITE suffix in directory names
                if f.endswith(".nc"):
                    rel = os.path.relpath(root, central_dir)
                    if f"__{site}" in rel:
                        nc_path = os.path.join(root, f)
                        break
            if nc_path:
                break

        if not nc_path:
            logger.info(f"No sample .nc found for {site}, skipping precipitation map")
            continue

        try:
            import xarray as xr
            ds = xr.open_dataset(nc_path)

            # Get the first data variable
            var_name = list(ds.data_vars)[0]
            data = ds[var_name].values

            # Handle 2D or higher-dimensional data
            while data.ndim > 2:
                data = data[0]

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(data, cmap="Blues", interpolation="nearest",
                           aspect="auto")
            ax.set_title(f"Precipitation Sample — {site}", fontsize=14)

            # Try to add coordinate labels
            if "latitude" in ds.coords:
                lat = ds["latitude"].values
                lon = ds["longitude"].values
                ax.set_xlabel("Longitude Index", fontsize=11)
                ax.set_ylabel("Latitude Index", fontsize=11)
                # Add a few tick labels
                ny, nx = data.shape
                if ny > 0 and nx > 0:
                    yticks = np.linspace(0, ny - 1, min(5, ny), dtype=int)
                    xticks = np.linspace(0, nx - 1, min(5, nx), dtype=int)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels([f"{lat[i]:.1f}" for i in yticks])
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([f"{lon[i]:.1f}" for i in xticks], rotation=45)
                    ax.set_xlabel("Longitude", fontsize=11)
                    ax.set_ylabel("Latitude", fontsize=11)

            fig.colorbar(im, ax=ax, label=f"{var_name} (mm/hr)", shrink=0.8)
            plt.tight_layout()
            maps[site] = fig_to_base64(fig)
            plt.close(fig)
            ds.close()
            logger.info(f"Generated precipitation map for {site}")

        except Exception as e:
            logger.warning(f"Failed to generate precipitation map for {site}: {e}")

    return maps


def build_html_report(dag_img, bar_img, heatmap_img, precip_maps,
                      site_seq_counts, config_summary, timestamp):
    """Build a self-contained HTML report with embedded images."""
    precip_sections = ""
    for site, img_b64 in sorted(precip_maps.items()):
        precip_sections += f"""
        <div class="card">
            <h3>Precipitation Sample &mdash; {site}</h3>
            <img src="data:image/png;base64,{img_b64}" alt="Precipitation map for {site}">
        </div>
        """

    total_seqs = sum(site_seq_counts.values())
    site_rows = ""
    for site in sorted(site_seq_counts):
        site_rows += f"<tr><td>{site}</td><td>{site_seq_counts[site]}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPRITE FL Pipeline Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1a237e;
            border-bottom: 3px solid #3f51b5;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #283593;
            margin-top: 40px;
        }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .card img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 10px auto;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 14px;
            text-align: left;
        }}
        th {{
            background: #3f51b5;
            color: white;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .metric {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric .value {{
            font-size: 2em;
            font-weight: bold;
            color: #3f51b5;
        }}
        .metric .label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.85em;
        }}
    </style>
</head>
<body>
    <h1>SPRITE Federated Learning Pipeline Report</h1>
    <p class="timestamp">Generated: {timestamp}</p>

    <div class="summary-grid">
        <div class="metric">
            <div class="value">{len(site_seq_counts)}</div>
            <div class="label">Radar Sites</div>
        </div>
        <div class="metric">
            <div class="value">{total_seqs}</div>
            <div class="label">Total Sequences</div>
        </div>
        <div class="metric">
            <div class="value">{config_summary.get('fl_runs', 0)}</div>
            <div class="label">FL Training Runs</div>
        </div>
        <div class="metric">
            <div class="value">{config_summary.get('cen_runs', 0)}</div>
            <div class="label">Centralized Runs</div>
        </div>
    </div>

    <h2>Workflow DAG</h2>
    <div class="card">
        <img src="data:image/png;base64,{dag_img}" alt="Workflow DAG">
    </div>

    <h2>Pipeline Summary</h2>

    <div class="card">
        <h3>Sequences per Site</h3>
        <img src="data:image/png;base64,{bar_img}" alt="Sequence bar chart">
        <table>
            <tr><th>Site</th><th>Sequences</th></tr>
            {site_rows}
            <tr style="font-weight:bold"><td>Total</td><td>{total_seqs}</td></tr>
        </table>
    </div>

    <div class="card">
        <h3>Data Split Distribution</h3>
        <img src="data:image/png;base64,{heatmap_img}" alt="Split heatmap">
    </div>

    <h2>Precipitation Sample Maps</h2>
    {precip_sections if precip_sections else '<div class="card"><p>No .nc sample files found in central snapshot.</p></div>'}

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual summary report for the SPRITE pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--central-tar", required=True,
                        help="Input central snapshot tar.gz archive")
    parser.add_argument("--configs-tar", required=True,
                        help="Input FL configs tar.gz archive")
    parser.add_argument("--output", required=True, help="Output HTML report file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    sites = config.get("sites", [])
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Extract central snapshot
    central_tmp = tempfile.mkdtemp(prefix="sprite_viz_central_")
    logger.info(f"Extracting {args.central_tar}")
    with tarfile.open(args.central_tar, "r:gz") as tar:
        tar.extractall(central_tmp)

    # Count sequences per site from central snapshot directory names
    site_seq_counts = defaultdict(int)
    site_split_counts = defaultdict(lambda: defaultdict(int))

    for root, dirs, files in os.walk(central_tmp):
        for d in dirs:
            # Sequence dirs have __SITE suffix: seq-16-1-0.0__KBOX
            if "__" in d and d.split("__")[-1] in sites:
                site_name = d.split("__")[-1]
                site_seq_counts[site_name] += 1
                # Determine split from path
                rel = os.path.relpath(root, central_tmp)
                parts = rel.split(os.sep)
                # Structure: {window}/v001/{span}/{split}
                if len(parts) >= 4:
                    split_name = parts[3]
                    site_split_counts[site_name][split_name] += 1

    # Ensure all configured sites appear
    for site in sites:
        if site not in site_seq_counts:
            site_seq_counts[site] = 0

    logger.info(f"Site sequence counts: {dict(site_seq_counts)}")

    # Extract configs tar to count runs
    configs_tmp = tempfile.mkdtemp(prefix="sprite_viz_configs_")
    logger.info(f"Extracting {args.configs_tar}")
    with tarfile.open(args.configs_tar, "r:gz") as tar:
        tar.extractall(configs_tmp)

    fl_runs = 0
    cen_runs = 0
    for root, dirs, files in os.walk(configs_tmp):
        if "config.yaml" in files:
            rel = os.path.relpath(root, configs_tmp)
            if rel.startswith("fl/"):
                fl_runs += 1
            elif rel.startswith("cen/"):
                cen_runs += 1

    config_summary = {"fl_runs": fl_runs, "cen_runs": cen_runs}

    # Generate visualizations
    logger.info("Generating DAG diagram...")
    dag_img = make_dag_diagram(sites)

    logger.info("Generating sequence bar chart...")
    bar_img = make_sequence_bar_chart(dict(site_seq_counts))

    logger.info("Generating completeness heatmap...")
    heatmap_img = make_completeness_heatmap(dict(site_split_counts))

    logger.info("Generating precipitation sample maps...")
    precip_maps = make_precip_maps(central_tmp, sites)

    # Cleanup temp dirs
    shutil.rmtree(central_tmp, ignore_errors=True)
    shutil.rmtree(configs_tmp, ignore_errors=True)

    # Build HTML report
    logger.info("Building HTML report...")
    html = build_html_report(
        dag_img, bar_img, heatmap_img, precip_maps,
        dict(site_seq_counts), config_summary, now,
    )

    with open(args.output, "w") as f:
        f.write(html)

    logger.info(f"Visual report written: {args.output}")


if __name__ == "__main__":
    main()
