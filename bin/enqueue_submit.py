#!/usr/bin/env python3

"""Write run configs and submit FL/centralized training jobs.

Generates per-run configuration files (config.yaml, clients.map.json,
flwr.override.toml) for federated learning and centralized training
runs. Records job entries in sprite.sqlite and writes _RUNNING markers.

Corresponds to orchestrator Steps G-H: enqueue_jobs() + submit_jobs().
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NORM_TAG = "no-normalization"
PRECIP_TAG = "precip-min:0.0_precip-max:128"
VERSION = "v001"


def write_fl_configs(run_dir, config, sites, snapshot_root, window, span_key):
    """Generate FL run configuration files."""
    os.makedirs(run_dir, exist_ok=True)

    fl_cfg = config.get("fl", {})
    span_dir_name = f"span_{span_key}"

    # Central data root
    central_data_root = os.path.join(
        snapshot_root, "central", NORM_TAG, PRECIP_TAG,
        window, VERSION, span_dir_name,
    )

    # Per-site data roots
    site_configs = []
    for site in sites:
        site_data_root = os.path.join(
            snapshot_root, "sites", site, NORM_TAG, PRECIP_TAG,
            window, VERSION, span_dir_name,
        )
        site_configs.append({
            "site": site,
            "data_root": site_data_root,
        })

    # config.yaml (matching reference format)
    run_config = {
        "data": {
            "central": {"data_root": central_data_root},
            "sites": site_configs,
        },
        "fl": {
            "server_addr": fl_cfg.get("server_addr", "127.0.0.1:8080"),
            "rounds": fl_cfg.get("rounds", 50),
            "local_steps": fl_cfg.get("local_steps", 2),
            "clients_concurrency": fl_cfg.get("clients_concurrency", 4),
            "min_clients_to_start": fl_cfg.get("min_clients_to_start", 2),
            "uv_activate": fl_cfg.get("uv_activate", ""),
            "clients_map_template": fl_cfg.get("clients_map_template", ""),
        },
        "window": window,
        "span": span_key,
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # clients.map.json (matching reference format)
    run_mark = f"{window}_{span_key}_run001"
    clients_map = {}
    for i, site in enumerate(sites):
        site_data_dir = os.path.join(
            snapshot_root, "sites", site, NORM_TAG, PRECIP_TAG,
            window, VERSION, span_dir_name,
        )
        clients_map[str(i)] = {
            "radar_id": site,
            "data_dir": site_data_dir,
            "taken": True,
            "log": "gradients",
            "log_freq": 100,
            "run_mark": run_mark,
        }
    # Default entry
    clients_map["default"] = {
        "log": "gradients",
        "log_freq": 100,
        "run_mark": run_mark,
    }
    with open(os.path.join(run_dir, "clients.map.json"), "w") as f:
        json.dump(clients_map, f, indent=2)

    # flwr.override.toml (matching reference format)
    mapping_file = os.path.join(run_dir, "clients.map.json")
    data_root_train = os.path.join(central_data_root, "train")
    toml_lines = [
        '[tool.flwr.app.config]',
        f'mapping_file = "{mapping_file}"',
        f'data-root    = "{data_root_train}"',
        '',
    ]
    with open(os.path.join(run_dir, "flwr.override.toml"), "w") as f:
        f.write("\n".join(toml_lines))

    logger.info(f"FL configs written to {run_dir}")


def write_cen_configs(run_dir, config, snapshot_root, window, span_key):
    """Generate centralized run configuration files."""
    os.makedirs(run_dir, exist_ok=True)

    span_dir_name = f"span_{span_key}"
    central_data_root = os.path.join(
        snapshot_root, "central", NORM_TAG, PRECIP_TAG,
        window, VERSION, span_dir_name,
    )

    run_config = {
        "data": {
            "data_root": central_data_root,
        },
        "window": window,
        "span": span_key,
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    logger.info(f"CEN configs written to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Enqueue and submit training jobs")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--central-snapshot-marker", required=True,
                        help="Input central snapshot marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.central_snapshot_marker, "r") as f:
        cs_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    runs_root = config.get("paths", {}).get("runs_root", "")
    db_dir = config.get("paths", {}).get("db_dir", "")
    db_path = os.path.join(db_dir, "sprite.sqlite") if db_dir else "sprite.sqlite"
    snapshot_root = config.get("paths", {}).get("snapshot_root", "")
    fl_cfg = config.get("fl", {})
    slurm_cfg = config.get("slurm", {})
    sites = config.get("sites", [])
    windows_cfg = config.get("windows", [])
    enabled_windows = [w["name"] for w in windows_cfg if w.get("enabled", False)]

    logger.info(f"Runs root: {runs_root}")
    logger.info(f"FL rounds: {fl_cfg.get('rounds', 'N/A')}")
    logger.info(f"Slurm partition: {slurm_cfg.get('partition', 'N/A')}")

    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    now = datetime.utcnow().isoformat() + "Z"
    submitted_jobs = []

    # Extract unique (window, span_key) pairs from central snapshot marker
    central_splits = cs_marker.get("central_splits", {})
    span_combos = set()
    for combo_key in central_splits:
        parts = combo_key.split("/")
        if len(parts) == 3:
            # e.g. "1mo/span_2022-02/train" -> window="1mo", span_key="2022-02"
            span_key = parts[1].replace("span_", "")
            span_combos.add((parts[0], span_key))

    # Fallback: compute from config if no central_splits info
    if not span_combos:
        scan = config.get("scan", {})
        start = str(scan.get("start", "2022-02-01"))
        for w in enabled_windows:
            span_combos.add((w, start[:7]))

    for window, span_key in sorted(span_combos):
        span_dir_name = f"span_{span_key}"

        # FL run
        fl_run_dir = os.path.join(runs_root, "fl", window, span_dir_name, "run_001")
        write_fl_configs(fl_run_dir, config, sites, snapshot_root, window, span_key)

        # Write _RUNNING marker
        with open(os.path.join(fl_run_dir, "_RUNNING"), "w") as f:
            pass

        fl_payload = {
            "run_dir": fl_run_dir,
            "fl_server_data_dir": os.path.join(
                snapshot_root, "central", NORM_TAG, PRECIP_TAG,
                window, VERSION, span_dir_name,
            ),
        }
        cursor = conn.execute(
            """INSERT INTO jobs (kind, window, span, site, status, slurm_id,
                                retries, payload_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            ("train_fl", window, span_key, "", "queued", None,
             0, json.dumps(fl_payload), now, now),
        )
        fl_job_id = cursor.lastrowid
        submitted_jobs.append({
            "job_id": fl_job_id,
            "kind": "train_fl",
            "window": window,
            "span": span_key,
            "run_dir": fl_run_dir,
        })

        # CEN run
        cen_run_dir = os.path.join(runs_root, "cen", window, span_dir_name, "run_001")
        write_cen_configs(cen_run_dir, config, snapshot_root, window, span_key)

        # Write _RUNNING marker
        with open(os.path.join(cen_run_dir, "_RUNNING"), "w") as f:
            pass

        cen_payload = {
            "run_dir": cen_run_dir,
        }
        cursor = conn.execute(
            """INSERT INTO jobs (kind, window, span, site, status, slurm_id,
                                retries, payload_json, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            ("train_cen", window, span_key, "", "queued", None,
             0, json.dumps(cen_payload), now, now),
        )
        cen_job_id = cursor.lastrowid
        submitted_jobs.append({
            "job_id": cen_job_id,
            "kind": "train_cen",
            "window": window,
            "span": span_key,
            "run_dir": cen_run_dir,
        })

        logger.info(f"Enqueued FL job {fl_job_id} and CEN job {cen_job_id} "
                     f"for {window}/{span_dir_name}")

    conn.commit()
    conn.close()

    marker = {
        "stage": "enqueue_submit",
        "status": "success",
        "timestamp": now,
        "runs_root": runs_root,
        "submitted_jobs": submitted_jobs,
        "fl_rounds": fl_cfg.get("rounds"),
        "slurm_partition": slurm_cfg.get("partition"),
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
