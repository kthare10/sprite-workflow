#!/home/ubuntu/sprite-venv/bin/python3

"""Index raw .nc files into SQLite inventory.

Walks the raw_store directory tree and registers every .nc file in the
sprite.sqlite database (raw_files table). This provides a queryable index
of all available data partitioned by site, split, and date.

Corresponds to orchestrator Step A: inventory() -> walk_and_index_raw.
"""

import argparse
import hashlib
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

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raw_files(
  site TEXT, split TEXT, day TEXT, ts TEXT, path TEXT, size INTEGER, sha1 TEXT,
  status TEXT, first_seen TEXT, last_seen TEXT,
  PRIMARY KEY (site, split, day, ts));

CREATE TABLE IF NOT EXISTS expected(
  site TEXT, split TEXT, day TEXT, ts TEXT, source TEXT, first_seen TEXT,
  PRIMARY KEY (site, split, day, ts));

CREATE TABLE IF NOT EXISTS freezes(
  period_type TEXT, period_key TEXT, site TEXT, split TEXT, version TEXT,
  status TEXT, created_at TEXT, updated_at TEXT,
  PRIMARY KEY (period_type, period_key, site, split, version));

CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT, window TEXT, span TEXT, site TEXT,
  status TEXT, slurm_id TEXT, retries INTEGER DEFAULT 0,
  payload_json TEXT, created_at TEXT, updated_at TEXT);

CREATE TABLE IF NOT EXISTS seq_index(
  site TEXT, split TEXT, seq_name TEXT,
  tmin TEXT, tmax TEXT,
  PRIMARY KEY (site, split, seq_name));
"""


def sha1_file(path):
    """Compute SHA1 hash of a file."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def init_db(db_path):
    """Create/open the SQLite database with WAL mode and all tables."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def main():
    parser = argparse.ArgumentParser(description="Index raw .nc files into SQLite")
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--download-marker", required=True, help="Input download marker")
    parser.add_argument("--output-marker", required=True, help="Output JSON marker file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.download_marker, "r") as f:
        dl_marker = json.load(f)

    out_dir = os.path.dirname(args.output_marker)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    raw_root = config.get("paths", {}).get("raw_root", "")
    db_dir = config.get("paths", {}).get("db_dir", "")
    db_path = os.path.join(db_dir, "sprite.sqlite") if db_dir else "sprite.sqlite"
    sites = config.get("sites", [])

    logger.info(f"Raw root: {raw_root}")
    logger.info(f"DB path: {db_path}")
    logger.info(f"Sites: {sites}")

    conn = init_db(db_path)
    now = datetime.utcnow().isoformat() + "Z"
    indexed_files = 0

    # Walk raw_store/{SITE}/{split}/{YYYYMMDD}/*.nc
    for site in sites:
        site_dir = os.path.join(raw_root, site)
        if not os.path.isdir(site_dir):
            logger.warning(f"Site directory not found: {site_dir}")
            continue

        for split in os.listdir(site_dir):
            split_dir = os.path.join(site_dir, split)
            if not os.path.isdir(split_dir) or split not in ("train", "test"):
                continue

            for day_dir_name in sorted(os.listdir(split_dir)):
                day_path = os.path.join(split_dir, day_dir_name)
                if not os.path.isdir(day_path):
                    continue

                batch = []
                for nc_file in sorted(os.listdir(day_path)):
                    if not nc_file.endswith(".nc"):
                        continue

                    nc_path = os.path.join(day_path, nc_file)
                    size = os.path.getsize(nc_path)
                    sha1 = sha1_file(nc_path)

                    # Parse ts from filename: YYYYMMDD_HHMMSS.nc
                    ts = nc_file.replace(".nc", "")
                    day = day_dir_name

                    batch.append((site, split, day, ts, nc_path, size, sha1,
                                  "ok", now, now))

                if batch:
                    conn.executemany(
                        """INSERT OR REPLACE INTO raw_files
                           (site, split, day, ts, path, size, sha1,
                            status, first_seen, last_seen)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        batch,
                    )
                    indexed_files += len(batch)

                if indexed_files % 5000 == 0 and indexed_files > 0:
                    conn.commit()
                    logger.info(f"Indexed {indexed_files} files so far...")

    conn.commit()
    conn.close()

    logger.info(f"Inventory complete: {indexed_files} files indexed")

    marker = {
        "stage": "inventory",
        "status": "success",
        "timestamp": now,
        "raw_root": raw_root,
        "db_path": db_path,
        "sites": sites,
        "indexed_files": indexed_files,
    }

    with open(args.output_marker, "w") as f:
        json.dump(marker, f, indent=2)

    logger.info(f"Marker written: {args.output_marker}")


if __name__ == "__main__":
    main()
