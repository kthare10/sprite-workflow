#!/usr/bin/env python3

"""Generate FL and centralized training configuration files.

Accepts the central snapshot tar archive, inspects available
windows/spans/splits, and generates per-run configuration files
(config.yaml, clients.map.json, flwr.override.toml) with relative
data paths only. Packages all config files into an output tar.

No SQLite, no absolute paths, no _RUNNING markers.
Replaces the old enqueue_submit.py.
"""

import argparse
import json
import logging
import os
import shutil
import tarfile
import tempfile
from datetime import datetime

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

VERSION = "v001"


def write_fl_configs(run_dir, config, sites, window, span_key):
    """Generate FL run configuration files with relative paths."""
    os.makedirs(run_dir, exist_ok=True)

    fl_cfg = config.get("fl", {})
    span_dir_name = f"span_{span_key}"
    preproc_cfg = config.get("preprocessor", {})
    norm_tag = preproc_cfg.get("normalization", "no-normalization")
    precip_tag = preproc_cfg.get("precip_range", "precip-min:0.0_precip-max:128")

    # Relative data paths (relative to where central tar is extracted)
    central_data_rel = f"central/{norm_tag}/{precip_tag}/{window}/{VERSION}/{span_dir_name}"

    # Per-site relative data paths
    site_configs = []
    for site in sites:
        site_data_rel = f"sites/{site}/{norm_tag}/{precip_tag}/{window}/{VERSION}/{span_dir_name}"
        site_configs.append({
            "site": site,
            "data_root": site_data_rel,
        })

    # config.yaml
    run_config = {
        "data": {
            "central": {"data_root": central_data_rel},
            "sites": site_configs,
        },
        "fl": {
            "server_addr": fl_cfg.get("server_addr", "127.0.0.1:8080"),
            "rounds": fl_cfg.get("rounds", 50),
            "local_steps": fl_cfg.get("local_steps", 2),
            "clients_concurrency": fl_cfg.get("clients_concurrency", 4),
            "min_clients_to_start": fl_cfg.get("min_clients_to_start", 2),
        },
        "window": window,
        "span": span_key,
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    # clients.map.json
    run_mark = f"{window}_{span_key}_run001"
    clients_map = {}
    for i, site in enumerate(sites):
        site_data_rel = f"sites/{site}/{norm_tag}/{precip_tag}/{window}/{VERSION}/{span_dir_name}"
        clients_map[str(i)] = {
            "radar_id": site,
            "data_dir": site_data_rel,
            "taken": True,
            "log": "gradients",
            "log_freq": 100,
            "run_mark": run_mark,
        }
    clients_map["default"] = {
        "log": "gradients",
        "log_freq": 100,
        "run_mark": run_mark,
    }
    with open(os.path.join(run_dir, "clients.map.json"), "w") as f:
        json.dump(clients_map, f, indent=2)

    # flwr.override.toml
    mapping_file = "clients.map.json"
    data_root_train = f"{central_data_rel}/train"
    toml_lines = [
        '[tool.flwr.app.config]',
        f'mapping_file = "{mapping_file}"',
        f'data-root    = "{data_root_train}"',
        '',
    ]
    with open(os.path.join(run_dir, "flwr.override.toml"), "w") as f:
        f.write("\n".join(toml_lines))

    logger.info(f"FL configs written to {run_dir}")


def write_cen_configs(run_dir, config, window, span_key):
    """Generate centralized run configuration files with relative paths."""
    os.makedirs(run_dir, exist_ok=True)

    preproc_cfg = config.get("preprocessor", {})
    norm_tag = preproc_cfg.get("normalization", "no-normalization")
    precip_tag = preproc_cfg.get("precip_range", "precip-min:0.0_precip-max:128")
    span_dir_name = f"span_{span_key}"
    central_data_rel = f"central/{norm_tag}/{precip_tag}/{window}/{VERSION}/{span_dir_name}"

    run_config = {
        "data": {
            "data_root": central_data_rel,
        },
        "window": window,
        "span": span_key,
    }
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(run_config, f, default_flow_style=False)

    logger.info(f"CEN configs written to {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate FL/centralized training configs from central snapshot"
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--central-tar", required=True,
                        help="Input central snapshot tar.gz archive")
    parser.add_argument("--output", required=True, help="Output configs tar.gz archive")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    sites = config.get("sites", [])
    fl_cfg = config.get("fl", {})

    # Extract central tar to inspect available windows/spans
    central_tmp = tempfile.mkdtemp(prefix="sprite_central_cfg_")
    logger.info(f"Extracting {args.central_tar} to inspect structure")
    with tarfile.open(args.central_tar, "r:gz") as tar:
        tar.extractall(central_tmp)

    # Discover (window, span_key) pairs from extracted structure
    # Structure: {window}/v001/span_{YYYY-MM}/{split}/...
    span_combos = set()
    for window_dir in sorted(os.listdir(central_tmp)):
        window_path = os.path.join(central_tmp, window_dir)
        if not os.path.isdir(window_path):
            continue
        version_path = os.path.join(window_path, "v001")
        if not os.path.isdir(version_path):
            continue
        for span_dir in sorted(os.listdir(version_path)):
            if span_dir.startswith("span_"):
                span_key = span_dir.replace("span_", "")
                span_combos.add((window_dir, span_key))

    shutil.rmtree(central_tmp, ignore_errors=True)

    if not span_combos:
        # Fallback: compute from config
        scan = config.get("scan", {})
        windows_cfg = config.get("windows", [])
        enabled_windows = [w["name"] for w in windows_cfg if w.get("enabled", False)]
        start = str(scan.get("start", "2022-02-01"))
        for w in enabled_windows:
            span_combos.add((w, start[:7]))

    logger.info(f"Discovered {len(span_combos)} (window, span) combinations")

    # Generate configs
    configs_dir = tempfile.mkdtemp(prefix="sprite_configs_")

    for window, span_key in sorted(span_combos):
        span_dir_name = f"span_{span_key}"

        # FL run
        fl_run_dir = os.path.join(configs_dir, "fl", window, span_dir_name, "run_001")
        write_fl_configs(fl_run_dir, config, sites, window, span_key)

        # CEN run
        cen_run_dir = os.path.join(configs_dir, "cen", window, span_dir_name, "run_001")
        write_cen_configs(cen_run_dir, config, window, span_key)

        logger.info(f"Generated configs for {window}/{span_dir_name}")

    # Package into tar.gz archive
    logger.info(f"Creating output archive: {args.output}")
    with tarfile.open(args.output, "w:gz") as tar:
        for root, dirs, files in os.walk(configs_dir):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, configs_dir)
                tar.add(full_path, arcname=arcname)

    shutil.rmtree(configs_dir, ignore_errors=True)

    logger.info(f"Output archive written: {args.output}")


if __name__ == "__main__":
    main()
