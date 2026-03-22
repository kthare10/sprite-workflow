#!/usr/bin/env python3

"""Merge all site snapshots into a central snapshot.

Accepts per-site snapshot tar archives and merges them into a single
central snapshot tar. Sequences from different sites are distinguished
by appending __{SITE} to the directory name.

No symlinks, no absolute paths.
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


def main():
    parser = argparse.ArgumentParser(description="Merge site snapshots into central snapshot")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site-tars", nargs="+", required=True,
                        help="Per-site snapshot tar.gz archives")
    parser.add_argument("--sites", nargs="+", required=True,
                        help="Site names corresponding to --site-tars (same order)")
    parser.add_argument("--output", required=True, help="Output central snapshot tar.gz archive")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"Sites: {args.sites}")
    logger.info(f"Site tars: {args.site_tars}")

    if len(args.site_tars) != len(args.sites):
        logger.error("Number of --site-tars must match number of --sites")
        import sys
        sys.exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    preproc_cfg = config.get("preprocessor", {})
    norm_tag = preproc_cfg.get("normalization", "no-normalization")
    precip_tag = preproc_cfg.get("precip_range", "precip-min:0.0_precip-max:128")

    # Output temp directory for central snapshot
    central_dir = tempfile.mkdtemp(prefix="sprite_central_")
    now = datetime.utcnow().isoformat() + "Z"

    # Track (window, span_dir, split) -> seq_count for manifests
    combo_counts = {}

    for site, tar_path in zip(args.sites, args.site_tars):
        # Extract each site tar to temp dir
        site_tmp = tempfile.mkdtemp(prefix=f"sprite_site_{site.lower()}_")
        logger.info(f"Extracting {tar_path} for site {site}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(site_tmp)

        # Walk the extracted site snapshot
        # Structure: {window}/v001/span_{YYYY-MM}/{split}/seq-*/*
        for root, dirs, files in os.walk(site_tmp):
            rel = os.path.relpath(root, site_tmp)
            parts = rel.split(os.sep)

            # We're looking for sequence dirs at depth 4:
            # {window}/v001/{span_dir}/{split}/{seq_dir}
            if len(parts) >= 4:
                window, version, span_dir, split = parts[0], parts[1], parts[2], parts[3]

                if len(parts) == 5:
                    # This is a sequence directory
                    seq_name = parts[4]
                    dst_seq_name = f"{seq_name}__{site}"
                    dst_path = os.path.join(
                        central_dir, window, version, span_dir, split, dst_seq_name
                    )
                    if not os.path.exists(dst_path):
                        shutil.copytree(root, dst_path)

                    combo_key = (window, span_dir, split)
                    combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1

                elif len(parts) == 4 and files:
                    # This is the split directory level - skip manifest files,
                    # they'll be regenerated
                    pass

        shutil.rmtree(site_tmp, ignore_errors=True)

    # Write central MANIFEST.json for each combo
    for (window, span_dir, split), seq_count in sorted(combo_counts.items()):
        combo_dir = os.path.join(central_dir, window, "v001", span_dir, split)
        os.makedirs(combo_dir, exist_ok=True)

        span_key = span_dir.replace("span_", "")
        manifest = {
            "level": "central",
            "split": split,
            "window": window,
            "span": span_key,
            "version": "v001",
            "norm_tag": f"{norm_tag}/{precip_tag}",
            "seq_count": seq_count,
            "sites": sorted(args.sites),
            "created_at": now,
            "success": True,
        }
        with open(os.path.join(combo_dir, "MANIFEST.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"  central/{window}/{span_dir}/{split}: "
            f"{seq_count} sequences from {len(args.sites)} sites"
        )

    # Package into tar.gz archive
    logger.info(f"Creating output archive: {args.output}")
    with tarfile.open(args.output, "w:gz") as tar:
        for root, dirs, files in os.walk(central_dir):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, central_dir)
                tar.add(full_path, arcname=arcname)

    shutil.rmtree(central_dir, ignore_errors=True)

    logger.info(f"Output archive written: {args.output}")


if __name__ == "__main__":
    main()
