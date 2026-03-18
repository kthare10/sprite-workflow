#!/usr/bin/env python3

"""Audit and freeze monthly data for a single site.

For each month in the scan range, verifies data completeness against the
configured missing_ratio_tol thresholds. Produces MANIFEST.json + _SUCCESS
or _PARTIAL markers under the freeze_manifests directory.

Corresponds to orchestrator Step C: freeze_months() -> audit_and_freeze_month.
"""

import argparse
import calendar
import hashlib
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def ts_to_split(ts_str, ratio=0.2, seed=2025):
    """Deterministically assign a timestamp to train or test split (must match download.py)."""
    h = hashlib.md5(f"{seed}:{ts_str}".encode()).hexdigest()
    return "test" if (int(h, 16) % 1000) < int(ratio * 1000) else "train"


def expected_timestamps_for_month(year, month, step_minutes=2):
    """Generate all expected YYYYMMDD_HHMMSS timestamps for a month."""
    days_in_month = calendar.monthrange(year, month)[1]
    timestamps = []
    for day in range(1, days_in_month + 1):
        day_str = f"{year:04d}{month:02d}{day:02d}"
        t = datetime(year, month, day)
        end = t + timedelta(days=1)
        while t < end:
            ts = f"{day_str}_{t.strftime('%H%M%S')}"
            timestamps.append((day_str, ts))
            t += timedelta(minutes=step_minutes)
    return timestamps


def main():
    parser = argparse.ArgumentParser(description="Freeze monthly data for a site")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--site", required=True, help="Site identifier (e.g. KBOX)")
    parser.add_argument("--plan-spans-marker", required=True, help="Input plan_spans marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")
    logger.info(f"Site: {args.site}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.plan_spans_marker, "r") as f:
        spans_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    db_dir = config.get("paths", {}).get("db_dir", "")
    db_path = os.path.join(db_dir, "sprite.sqlite") if db_dir else "sprite.sqlite"
    freeze_cfg = config.get("freeze", {})
    datasource = freeze_cfg.get("datasource", "s3")
    freeze_lag_days = freeze_cfg.get("freeze_lag_days", 3)
    missing_tol = freeze_cfg.get("missing_ratio_tol", {"train": 0.25, "test": 0.85})
    spans = spans_marker.get("spans", [])
    splits = spans_marker.get("splits", ["train", "test"])

    split_ratio = 0.2
    split_seed = 2025
    step_minutes = 2

    manifest_base = os.path.join(db_dir, "freeze_manifests") if db_dir else "freeze_manifests"

    logger.info(f"Datasource: {datasource}")
    logger.info(f"Freeze lag days: {freeze_lag_days}")
    logger.info(f"Missing ratio tolerance: {missing_tol}")
    logger.info(f"DB path: {db_path}")

    conn = sqlite3.connect(db_path, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=120000")
    now = datetime.utcnow().isoformat() + "Z"

    frozen_months = []

    for span_info in spans:
        span_label = span_info["span"]      # e.g. "span_2022-02"
        window = span_info["window"]         # e.g. "1mo"

        # Extract YYYY-MM from span label
        month_str = span_label.replace("span_", "")  # "2022-02"
        try:
            year = int(month_str.split("-")[0])
            month = int(month_str.split("-")[1])
        except (ValueError, IndexError):
            logger.warning(f"Cannot parse month from span: {span_label}")
            continue

        yyyymm = f"{year:04d}{month:02d}"
        period_key = f"{year:04d}-{month:02d}"  # "2022-02" for DB and manifest

        # Generate expected timestamps
        expected_ts = expected_timestamps_for_month(year, month, step_minutes)

        for split in splits:
            # Filter expected to this split
            split_expected = [
                (day, ts) for (day, ts) in expected_ts
                if ts_to_split(ts, ratio=split_ratio, seed=split_seed) == split
            ]

            # Insert into expected table
            exp_batch = [
                (args.site, split, day, ts, datasource, now)
                for (day, ts) in split_expected
            ]
            conn.executemany(
                """INSERT OR IGNORE INTO expected
                   (site, split, day, ts, source, first_seen)
                   VALUES (?,?,?,?,?,?)""",
                exp_batch,
            )

            # Query actual files from raw_files
            actual_count = conn.execute(
                """SELECT COUNT(*) FROM raw_files
                   WHERE site=? AND split=? AND day LIKE ?""",
                (args.site, split, f"{yyyymm}%"),
            ).fetchone()[0]

            expected_count = len(split_expected)
            missing_count = max(0, expected_count - actual_count)
            missing_ratio = missing_count / expected_count if expected_count > 0 else 0.0

            tol = missing_tol.get(split, 0.25)
            status = "success" if missing_ratio <= tol else "partial"
            marker_file = "_SUCCESS" if status == "success" else "_PARTIAL"

            # Get list of missing timestamps
            actual_ts_set = set()
            for row in conn.execute(
                "SELECT ts FROM raw_files WHERE site=? AND split=? AND day LIKE ?",
                (args.site, split, f"{yyyymm}%"),
            ):
                actual_ts_set.add(row[0])

            missing_list = [ts for (_, ts) in split_expected if ts not in actual_ts_set]

            # Write manifest (matching reference format exactly)
            manifest_dir = os.path.join(manifest_base, window, args.site, split, yyyymm)
            os.makedirs(manifest_dir, exist_ok=True)

            manifest = {
                "site": args.site,
                "split": split,
                "period_type": window,
                "period_key": period_key,
                "E_count": expected_count,
                "A_count": actual_count,
                "missing": missing_list,
                "missing_ratio": missing_ratio,
                "freeze_lag_days": freeze_lag_days,
                "missing_ratio_tol": tol,
                "version": "v001",
                "created_at": now,
                "status": status,
            }
            with open(os.path.join(manifest_dir, "MANIFEST.json"), "w") as f:
                json.dump(manifest, f, indent=2)

            # Write status marker (empty file)
            with open(os.path.join(manifest_dir, marker_file), "w") as f:
                pass

            # Insert into freezes table
            conn.execute(
                """INSERT OR REPLACE INTO freezes
                   (period_type, period_key, site, split, version,
                    status, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (window, period_key, args.site, split, "v001",
                 status, now, now),
            )

            logger.info(
                f"  {args.site}/{split}/{yyyymm}: "
                f"E={expected_count} A={actual_count} "
                f"missing={missing_ratio:.2%} -> {status}"
            )

        frozen_months.append({
            "span": span_label,
            "window": window,
            "month": yyyymm,
            "period_key": period_key,
            "status": "success",
        })

    conn.commit()
    conn.close()

    marker = {
        "stage": "freeze",
        "status": "success",
        "timestamp": now,
        "site": args.site,
        "datasource": datasource,
        "frozen_months": frozen_months,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
