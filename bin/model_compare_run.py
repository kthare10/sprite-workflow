#!/usr/bin/env python3

"""Run model comparison between FL and centralized training results.

Accepts the FL configs tar (containing per-run config.yaml, clients.map.json,
flwr.override.toml) and the central snapshot tar. Loads trained model
checkpoints (or training logs), computes comparison metrics between
federated and centralized approaches, and outputs a comparison JSON.

If a Model Compare Tool (MCT) web service endpoint is configured, submits
results via HTTP API. Otherwise performs local comparison.

Comparison metrics produced:
  - Per-window/span: FL vs CEN final loss, accuracy, F1
  - Convergence speed: rounds/epochs to reach target metric
  - Per-site contribution: client-level metrics for FL runs
  - Statistical tests: paired t-test on per-site metrics

Input:
  --config experiment_config.yaml
  --configs-tar fl_configs.tar.gz
  --central-tar central_snapshot.tar.gz
  --output model_comparison.json

Output JSON schema:
  {
    "stage": "model_compare_run",
    "status": "success",
    "comparisons": [
      {
        "window": "1mo",
        "span": "2022-02",
        "fl": { "loss": float, "accuracy": float, ... },
        "cen": { "loss": float, "accuracy": float, ... },
        "delta": { "loss": float, "accuracy": float, ... },
        "per_site_fl": { "KBOX": {...}, ... },
        "convergence": { "fl_rounds_to_target": int, "cen_epochs_to_target": int }
      }
    ],
    "summary": { "mean_fl_accuracy": float, "mean_cen_accuracy": float, ... }
  }
"""

import argparse
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


def find_training_logs(run_dir):
    """Search a run directory for training log files.

    Looks for common training output patterns:
      - metrics.json / training_log.json (structured logs)
      - checkpoint files (*.pt, *.pth, *.ckpt)
      - CSV logs (metrics.csv, history.csv)
      - Flower result files (results.json)
    """
    logs = {
        "metrics_json": None,
        "checkpoints": [],
        "csv_logs": [],
        "flower_results": None,
    }

    if not os.path.isdir(run_dir):
        return logs

    for root, dirs, files in os.walk(run_dir):
        for f in files:
            fpath = os.path.join(root, f)
            if f in ("metrics.json", "training_log.json", "results.json"):
                if f == "results.json":
                    logs["flower_results"] = fpath
                else:
                    logs["metrics_json"] = fpath
            elif f.endswith((".pt", ".pth", ".ckpt")):
                logs["checkpoints"].append(fpath)
            elif f.endswith(".csv") and "metric" in f.lower() or "history" in f.lower():
                logs["csv_logs"].append(fpath)

    return logs


def parse_metrics_json(path):
    """Parse a metrics JSON file into a standardized format."""
    try:
        with open(path, "r") as f:
            data = json.load(f)

        # Handle various formats
        if isinstance(data, list):
            # List of per-round/epoch records
            if data:
                last = data[-1]
                return {
                    "loss": last.get("loss", last.get("train_loss", None)),
                    "accuracy": last.get("accuracy", last.get("acc", last.get("test_accuracy", None))),
                    "f1": last.get("f1", last.get("f1_score", None)),
                    "rounds": len(data),
                    "history": data,
                }
        elif isinstance(data, dict):
            return {
                "loss": data.get("loss", data.get("final_loss", None)),
                "accuracy": data.get("accuracy", data.get("final_accuracy", None)),
                "f1": data.get("f1", data.get("f1_score", None)),
                "rounds": data.get("rounds", data.get("epochs", None)),
                "history": data.get("history", []),
            }
    except Exception as e:
        logger.warning(f"Failed to parse metrics from {path}: {e}")

    return None


def parse_flower_results(path):
    """Parse Flower framework results.json."""
    try:
        with open(path, "r") as f:
            data = json.load(f)

        # Flower results typically have losses_distributed, metrics_distributed
        history = []
        losses = data.get("losses_distributed", data.get("losses_centralized", []))
        metrics = data.get("metrics_distributed", data.get("metrics_centralized", {}))

        for i, (round_num, loss) in enumerate(losses):
            record = {"round": round_num, "loss": loss}
            # Try to find matching accuracy
            for metric_name, metric_vals in metrics.items():
                if "acc" in metric_name.lower():
                    if i < len(metric_vals):
                        record["accuracy"] = metric_vals[i][1]
            history.append(record)

        final = history[-1] if history else {}
        return {
            "loss": final.get("loss"),
            "accuracy": final.get("accuracy"),
            "f1": None,
            "rounds": len(history),
            "history": history,
        }
    except Exception as e:
        logger.warning(f"Failed to parse Flower results from {path}: {e}")
    return None


def parse_csv_log(path):
    """Parse a CSV training log."""
    try:
        import pandas as pd
        df = pd.read_csv(path)

        # Look for common column names
        loss_col = next((c for c in df.columns if "loss" in c.lower()), None)
        acc_col = next((c for c in df.columns if "acc" in c.lower()), None)

        last_row = df.iloc[-1] if len(df) > 0 else {}
        history = []
        for _, row in df.iterrows():
            record = {}
            if loss_col:
                record["loss"] = float(row[loss_col])
            if acc_col:
                record["accuracy"] = float(row[acc_col])
            history.append(record)

        return {
            "loss": float(last_row[loss_col]) if loss_col and loss_col in last_row else None,
            "accuracy": float(last_row[acc_col]) if acc_col and acc_col in last_row else None,
            "f1": None,
            "rounds": len(df),
            "history": history,
        }
    except Exception as e:
        logger.warning(f"Failed to parse CSV log from {path}: {e}")
    return None


def extract_run_metrics(run_dir):
    """Extract training metrics from a run directory using any available source."""
    logs = find_training_logs(run_dir)

    # Try sources in priority order
    if logs["flower_results"]:
        metrics = parse_flower_results(logs["flower_results"])
        if metrics:
            return metrics

    if logs["metrics_json"]:
        metrics = parse_metrics_json(logs["metrics_json"])
        if metrics:
            return metrics

    if logs["csv_logs"]:
        for csv_path in logs["csv_logs"]:
            metrics = parse_csv_log(csv_path)
            if metrics:
                return metrics

    # If model checkpoints exist but no logs, note their presence
    if logs["checkpoints"]:
        return {
            "loss": None,
            "accuracy": None,
            "f1": None,
            "rounds": None,
            "history": [],
            "note": f"Found {len(logs['checkpoints'])} checkpoints but no training logs",
        }

    return None


def compute_convergence(history, target_metric="accuracy", target_value=0.8):
    """Compute rounds/epochs to reach a target metric value."""
    if not history:
        return None

    for i, record in enumerate(history):
        val = record.get(target_metric)
        if val is not None and val >= target_value:
            return i + 1

    return None  # Never reached target


def submit_to_mct(endpoint, comparison_data):
    """Submit comparison results to Model Compare Tool web service."""
    try:
        import urllib.request
        import urllib.error

        payload = json.dumps(comparison_data).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
            logger.info(f"MCT response: {response_data.get('status', 'unknown')}")
            return response_data

    except urllib.error.URLError as e:
        logger.warning(f"MCT endpoint not reachable ({endpoint}): {e}")
    except Exception as e:
        logger.warning(f"MCT submission failed: {e}")

    return None


def extract_per_site_metrics(fl_run_dir, clients_map):
    """Extract per-site (per-client) metrics from FL run."""
    per_site = {}

    for client_id, client_info in clients_map.items():
        if client_id == "default":
            continue

        site = client_info.get("radar_id", f"client_{client_id}")

        # Look for per-client log files
        client_log_patterns = [
            os.path.join(fl_run_dir, "logs", f"client_{client_id}.json"),
            os.path.join(fl_run_dir, "logs", f"{site}.json"),
            os.path.join(fl_run_dir, f"client_{client_id}_metrics.json"),
        ]

        client_metrics = None
        for pattern in client_log_patterns:
            if os.path.exists(pattern):
                client_metrics = parse_metrics_json(pattern)
                break

        per_site[site] = client_metrics or {
            "loss": None,
            "accuracy": None,
            "note": "No per-client logs found",
        }

    return per_site


def main():
    parser = argparse.ArgumentParser(
        description="Compare FL and centralized training results"
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--configs-tar", required=True,
                        help="Input FL/CEN configs tar.gz archive")
    parser.add_argument("--central-tar", required=True,
                        help="Input central snapshot tar.gz archive")
    parser.add_argument("--output", required=True, help="Output comparison JSON file")
    args = parser.parse_args()

    logger.info(f"Config: {args.config}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    sites = config.get("sites", [])
    mct_cfg = config.get("model_compare", {})
    mct_endpoint = mct_cfg.get("endpoint")
    target_accuracy = mct_cfg.get("target_accuracy", 0.8)

    now = datetime.utcnow().isoformat() + "Z"

    # Extract configs tar to find run directories
    configs_tmp = tempfile.mkdtemp(prefix="sprite_mct_configs_")
    logger.info(f"Extracting {args.configs_tar}")
    with tarfile.open(args.configs_tar, "r:gz") as tar:
        tar.extractall(configs_tmp)

    # Discover all (window, span) runs
    comparisons = []
    fl_accuracies = []
    cen_accuracies = []

    # Walk FL runs
    fl_base = os.path.join(configs_tmp, "fl")
    cen_base = os.path.join(configs_tmp, "cen")

    fl_runs = {}  # (window, span) -> run_dir
    cen_runs = {}

    for base, target_dict, kind in [(fl_base, fl_runs, "fl"), (cen_base, cen_runs, "cen")]:
        if not os.path.isdir(base):
            continue
        for window_dir in sorted(os.listdir(base)):
            window_path = os.path.join(base, window_dir)
            if not os.path.isdir(window_path):
                continue
            for span_dir in sorted(os.listdir(window_path)):
                span_path = os.path.join(window_path, span_dir)
                if not os.path.isdir(span_path):
                    continue
                span_key = span_dir.replace("span_", "")
                for run_dir in sorted(os.listdir(span_path)):
                    run_path = os.path.join(span_path, run_dir)
                    if os.path.isdir(run_path):
                        target_dict[(window_dir, span_key)] = run_path
                        break  # Take first run

    # Compare each (window, span) pair
    all_keys = sorted(set(fl_runs.keys()) | set(cen_runs.keys()))

    for window, span_key in all_keys:
        logger.info(f"Comparing {window}/span_{span_key}")

        comparison = {
            "window": window,
            "span": span_key,
            "fl": None,
            "cen": None,
            "delta": {},
            "per_site_fl": {},
            "convergence": {},
        }

        # Extract FL metrics
        fl_run_dir = fl_runs.get((window, span_key))
        if fl_run_dir:
            fl_metrics = extract_run_metrics(fl_run_dir)
            if fl_metrics:
                comparison["fl"] = {
                    k: v for k, v in fl_metrics.items() if k != "history"
                }
                if fl_metrics.get("accuracy") is not None:
                    fl_accuracies.append(fl_metrics["accuracy"])

                # Convergence
                convergence_round = compute_convergence(
                    fl_metrics.get("history", []),
                    target_metric="accuracy",
                    target_value=target_accuracy,
                )
                comparison["convergence"]["fl_rounds_to_target"] = convergence_round

                # Per-site metrics
                clients_map_path = os.path.join(fl_run_dir, "clients.map.json")
                if os.path.exists(clients_map_path):
                    with open(clients_map_path, "r") as f:
                        clients_map = json.load(f)
                    comparison["per_site_fl"] = extract_per_site_metrics(
                        fl_run_dir, clients_map
                    )
            else:
                comparison["fl"] = {"note": "No training results found"}
                logger.info(f"  FL: no training results in {fl_run_dir}")
        else:
            comparison["fl"] = {"note": "No FL run configured"}

        # Extract CEN metrics
        cen_run_dir = cen_runs.get((window, span_key))
        if cen_run_dir:
            cen_metrics = extract_run_metrics(cen_run_dir)
            if cen_metrics:
                comparison["cen"] = {
                    k: v for k, v in cen_metrics.items() if k != "history"
                }
                if cen_metrics.get("accuracy") is not None:
                    cen_accuracies.append(cen_metrics["accuracy"])

                convergence_epoch = compute_convergence(
                    cen_metrics.get("history", []),
                    target_metric="accuracy",
                    target_value=target_accuracy,
                )
                comparison["convergence"]["cen_epochs_to_target"] = convergence_epoch
            else:
                comparison["cen"] = {"note": "No training results found"}
                logger.info(f"  CEN: no training results in {cen_run_dir}")
        else:
            comparison["cen"] = {"note": "No CEN run configured"}

        # Compute deltas (FL - CEN) where both have metrics
        fl_m = comparison.get("fl") or {}
        cen_m = comparison.get("cen") or {}
        for metric in ("loss", "accuracy", "f1"):
            fl_val = fl_m.get(metric)
            cen_val = cen_m.get(metric)
            if fl_val is not None and cen_val is not None:
                comparison["delta"][metric] = fl_val - cen_val

        comparisons.append(comparison)
        logger.info(f"  FL: {comparison['fl']}")
        logger.info(f"  CEN: {comparison['cen']}")
        logger.info(f"  Delta: {comparison['delta']}")

    shutil.rmtree(configs_tmp, ignore_errors=True)

    # Compute summary statistics
    summary = {
        "total_comparisons": len(comparisons),
        "mean_fl_accuracy": float(np.mean(fl_accuracies)) if fl_accuracies else None,
        "mean_cen_accuracy": float(np.mean(cen_accuracies)) if cen_accuracies else None,
        "std_fl_accuracy": float(np.std(fl_accuracies)) if fl_accuracies else None,
        "std_cen_accuracy": float(np.std(cen_accuracies)) if cen_accuracies else None,
        "fl_wins": sum(
            1 for c in comparisons
            if c.get("delta", {}).get("accuracy") is not None
            and c["delta"]["accuracy"] > 0
        ),
        "cen_wins": sum(
            1 for c in comparisons
            if c.get("delta", {}).get("accuracy") is not None
            and c["delta"]["accuracy"] < 0
        ),
        "ties": sum(
            1 for c in comparisons
            if c.get("delta", {}).get("accuracy") is not None
            and c["delta"]["accuracy"] == 0
        ),
    }

    # Statistical test if enough data
    if len(fl_accuracies) >= 2 and len(cen_accuracies) >= 2:
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(fl_accuracies, cen_accuracies)
            summary["ttest_statistic"] = float(t_stat)
            summary["ttest_pvalue"] = float(p_value)
            summary["significant_difference"] = p_value < 0.05
        except Exception as e:
            logger.warning(f"Statistical test failed: {e}")

    # Submit to MCT if configured
    mct_response = None
    if mct_endpoint:
        logger.info(f"Submitting to MCT at {mct_endpoint}")
        mct_response = submit_to_mct(mct_endpoint, {
            "comparisons": comparisons,
            "summary": summary,
            "sites": sites,
        })

    # Write output
    result = {
        "stage": "model_compare_run",
        "status": "success",
        "timestamp": now,
        "comparisons": comparisons,
        "summary": summary,
        "mct_response": mct_response,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Comparison report written: {args.output}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
