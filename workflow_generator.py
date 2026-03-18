#!/usr/bin/env python3

"""
Pegasus workflow generator for SPRITE Federated Learning pipeline.

Implements a multi-step pipeline for MRMS radar precipitation data:
  1. Download raw MRMS data from AWS S3
  2. Index raw .nc files into SQLite inventory
  3. Compute window/span combinations from config
  4. Per-site parallel branches:
     a. Freeze (audit & freeze monthly data)
     b. Preprocess frozen data
     c. Create site snapshot
  5. Merge all site snapshots into central snapshot
  6. Enqueue and submit FL/centralized training jobs
  7. Poll job status and retry failures
  8. Generate final report

Usage:
    ./workflow_generator.py \\
        --download-config download_config.yaml \\
        --experiment-config experiment_config.yaml \\
        --output workflow.yml

    ./workflow_generator.py \\
        --download-config download_config.yaml \\
        --experiment-config experiment_config.yaml \\
        -e condorpool --output workflow.yml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from Pegasus.api import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Per-tool resource configuration
TOOL_CONFIGS = {
    "download":          {"memory": "8 GB",  "cores": 2},
    "inventory":         {"memory": "2 GB",  "cores": 1},
    "plan_spans":        {"memory": "1 GB",  "cores": 1},
    "freeze":            {"memory": "2 GB",  "cores": 1},
    "preproc":           {"memory": "8 GB",  "cores": 2},
    "snapshot":          {"memory": "4 GB",  "cores": 1},
    "central_snapshot":  {"memory": "4 GB",  "cores": 1},
    "enqueue_submit":    {"memory": "2 GB",  "cores": 1},
    "poll_retry":        {"memory": "2 GB",  "cores": 1},
    "finalize_report":   {"memory": "1 GB",  "cores": 1},
}


class SpriteFlWorkflow:
    """Pegasus workflow for SPRITE Federated Learning pipeline."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "sprite_fl"

    # Config file paths (set from CLI args)
    download_config_path = None
    experiment_config_path = None

    # Parsed configs
    download_config = None
    experiment_config = None

    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")

    def load_configs(self):
        """Load and parse YAML config files."""
        with open(self.download_config_path, "r") as f:
            self.download_config = yaml.safe_load(f)
        with open(self.experiment_config_path, "r") as f:
            self.experiment_config = yaml.safe_load(f)

    @property
    def sites(self):
        """Return list of site names from experiment config."""
        return self.experiment_config.get("sites", [])

    def write(self):
        """Write all catalogs and workflow to files."""
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"

    # ------------------------------------------------------------------
    # Site Catalog
    # ------------------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="condorpool"):
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(
                Directory.SHARED_SCRATCH, self.shared_scratch_dir
            ).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(
                Directory.LOCAL_STORAGE, self.local_storage_dir
            ).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        if exec_site_name == "local":
            self.sc.add_sites(local)
        else:
            exec_site = (
                Site(exec_site_name)
                .add_condor_profile(universe="vanilla")
                .add_pegasus_profile(style="condor")
            )
            self.sc.add_sites(local, exec_site)

    # ------------------------------------------------------------------
    # Transformation Catalog
    # ------------------------------------------------------------------
    def create_transformation_catalog(self, exec_site_name="condorpool"):
        self.tc = TransformationCatalog()

        # Use container only for remote execution sites
        container = None
        if exec_site_name != "local":
            container = Container(
                "sprite_fl_container",
                container_type=Container.SINGULARITY,
                image="docker://kthare10/sprite-fl:latest",
                image_site="docker_hub",
            )
            self.tc.add_containers(container)

        transformations = []
        for tool_name, config in TOOL_CONFIGS.items():
            if exec_site_name == "local":
                # For local execution, use installed transformation
                tx = Transformation(
                    tool_name,
                    site="local",
                    pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                    is_stageable=False,
                )
            else:
                tx = Transformation(
                    tool_name,
                    site=exec_site_name,
                    pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                    is_stageable=True,
                )
                if container:
                    tx.container = container
            tx.add_pegasus_profile(
                memory=config["memory"], cores=config.get("cores", 1)
            )
            transformations.append(tx)

        self.tc.add_transformations(*transformations)

    # ------------------------------------------------------------------
    # Replica Catalog
    # ------------------------------------------------------------------
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()

        # Register config files as input replicas
        if self.download_config_path:
            self.rc.add_replica(
                "local", "download_config.yaml",
                "file://" + os.path.abspath(self.download_config_path)
            )
        if self.experiment_config_path:
            self.rc.add_replica(
                "local", "experiment_config.yaml",
                "file://" + os.path.abspath(self.experiment_config_path)
            )

    # ------------------------------------------------------------------
    # Workflow DAG
    # ------------------------------------------------------------------
    def create_workflow(self):
        """Create the SPRITE FL workflow DAG."""
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # ============================================================
        # Input File objects (registered in Replica Catalog)
        # ============================================================
        dl_config = File("download_config.yaml")
        exp_config = File("experiment_config.yaml")

        # ============================================================
        # Step 1: Download MRMS data from AWS S3
        # ============================================================
        download_marker = File("download_marker.json")

        download_job = (
            Job("download", _id="download", node_label="download")
            .add_args(
                "--config", dl_config,
                "--output-marker", download_marker,
            )
            .add_inputs(dl_config)
            .add_outputs(download_marker, stage_out=True,
                         register_replica=False)
        )
        self.wf.add_jobs(download_job)

        # ============================================================
        # Step 2: Inventory — index raw .nc files into SQLite
        # ============================================================
        inventory_marker = File("inventory_marker.json")

        inventory_job = (
            Job("inventory", _id="inventory", node_label="inventory")
            .add_args(
                "--config", exp_config,
                "--download-marker", download_marker,
                "--output-marker", inventory_marker,
            )
            .add_inputs(exp_config, download_marker)
            .add_outputs(inventory_marker, stage_out=True,
                         register_replica=False)
        )
        self.wf.add_jobs(inventory_job)

        # ============================================================
        # Step 3: Plan spans — compute window/span combinations
        # ============================================================
        plan_spans_marker = File("plan_spans_marker.json")

        plan_spans_job = (
            Job("plan_spans", _id="plan_spans", node_label="plan_spans")
            .add_args(
                "--config", exp_config,
                "--inventory-marker", inventory_marker,
                "--output-marker", plan_spans_marker,
            )
            .add_inputs(exp_config, inventory_marker)
            .add_outputs(plan_spans_marker, stage_out=True,
                         register_replica=False)
        )
        self.wf.add_jobs(plan_spans_job)

        # ============================================================
        # Steps 4-6: Per-site parallel branches
        #   freeze → preproc → snapshot (for each site)
        # ============================================================
        site_snapshot_markers = []

        for site in self.sites:
            site_lower = site.lower()

            # Step 4: Freeze — audit & freeze monthly data (per-site)
            freeze_marker = File(f"freeze_{site_lower}_marker.json")

            freeze_job = (
                Job("freeze", _id=f"freeze_{site_lower}",
                    node_label=f"freeze_{site_lower}")
                .add_args(
                    "--config", exp_config,
                    "--site", site,
                    "--plan-spans-marker", plan_spans_marker,
                    "--output-marker", freeze_marker,
                )
                .add_inputs(exp_config, plan_spans_marker)
                .add_outputs(freeze_marker, stage_out=True,
                             register_replica=False)
            )
            self.wf.add_jobs(freeze_job)

            # Step 5: Preprocess frozen data (per-site)
            preproc_marker = File(f"preproc_{site_lower}_marker.json")

            preproc_job = (
                Job("preproc", _id=f"preproc_{site_lower}",
                    node_label=f"preproc_{site_lower}")
                .add_args(
                    "--config", exp_config,
                    "--site", site,
                    "--freeze-marker", freeze_marker,
                    "--output-marker", preproc_marker,
                )
                .add_inputs(exp_config, freeze_marker)
                .add_outputs(preproc_marker, stage_out=True,
                             register_replica=False)
            )
            self.wf.add_jobs(preproc_job)

            # Step 6a: Site snapshot (per-site)
            snapshot_marker = File(f"snapshot_{site_lower}_marker.json")

            snapshot_job = (
                Job("snapshot", _id=f"snapshot_{site_lower}",
                    node_label=f"snapshot_{site_lower}")
                .add_args(
                    "--config", exp_config,
                    "--site", site,
                    "--preproc-marker", preproc_marker,
                    "--output-marker", snapshot_marker,
                )
                .add_inputs(exp_config, preproc_marker)
                .add_outputs(snapshot_marker, stage_out=True,
                             register_replica=False)
            )
            self.wf.add_jobs(snapshot_job)

            site_snapshot_markers.append(snapshot_marker)

        # ============================================================
        # Step 6b: Central snapshot — merge all site snapshots
        # ============================================================
        central_snapshot_marker = File("central_snapshot_marker.json")

        central_snapshot_job = (
            Job("central_snapshot", _id="central_snapshot",
                node_label="central_snapshot")
            .add_args(
                "--config", exp_config,
                "--output-marker", central_snapshot_marker,
            )
        )
        # Add all site snapshot markers as inputs
        central_snapshot_job.add_inputs(exp_config)
        for marker in site_snapshot_markers:
            central_snapshot_job.add_args("--site-marker", marker)
            central_snapshot_job.add_inputs(marker)
        central_snapshot_job.add_outputs(
            central_snapshot_marker, stage_out=True, register_replica=False
        )
        self.wf.add_jobs(central_snapshot_job)

        # ============================================================
        # Step 7: Enqueue and submit FL/centralized training jobs
        # ============================================================
        enqueue_submit_marker = File("enqueue_submit_marker.json")

        enqueue_submit_job = (
            Job("enqueue_submit", _id="enqueue_submit",
                node_label="enqueue_submit")
            .add_args(
                "--config", exp_config,
                "--central-snapshot-marker", central_snapshot_marker,
                "--output-marker", enqueue_submit_marker,
            )
            .add_inputs(exp_config, central_snapshot_marker)
            .add_outputs(enqueue_submit_marker, stage_out=True,
                         register_replica=False)
        )
        self.wf.add_jobs(enqueue_submit_job)

        # ============================================================
        # Step 8: Poll job status and retry failures
        # ============================================================
        poll_retry_marker = File("poll_retry_marker.json")

        poll_retry_job = (
            Job("poll_retry", _id="poll_retry", node_label="poll_retry")
            .add_args(
                "--config", exp_config,
                "--enqueue-submit-marker", enqueue_submit_marker,
                "--output-marker", poll_retry_marker,
            )
            .add_inputs(exp_config, enqueue_submit_marker)
            .add_outputs(poll_retry_marker, stage_out=True,
                         register_replica=False)
        )
        self.wf.add_jobs(poll_retry_job)

        # ============================================================
        # Step 9: Finalize report
        # ============================================================
        final_report = File("final_report.json")

        finalize_job = (
            Job("finalize_report", _id="finalize_report",
                node_label="finalize_report")
            .add_args(
                "--config", exp_config,
                "--poll-retry-marker", poll_retry_marker,
                "--output-marker", final_report,
            )
            .add_inputs(exp_config, poll_retry_marker)
            .add_outputs(final_report, stage_out=True,
                         register_replica=False)
        )
        self.wf.add_jobs(finalize_job)


# ======================================================================
# main() — CLI argument parsing
# ======================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SPRITE Federated Learning Pegasus Workflow Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --download-config download_config.yaml \\
           --experiment-config experiment_config.yaml

  %(prog)s --download-config download_config.yaml \\
           --experiment-config experiment_config.yaml \\
           -e condorpool --output workflow.yml
""",
    )

    # Standard Pegasus arguments
    parser.add_argument(
        "-s", "--skip-sites-catalog", action="store_true",
        help="Skip site catalog creation",
    )
    parser.add_argument(
        "-e", "--execution-site-name", metavar="STR", type=str,
        default="condorpool",
        help="Execution site name (default: condorpool)",
    )
    parser.add_argument(
        "-o", "--output", metavar="STR", type=str, default="workflow.yml",
        help="Output file (default: workflow.yml)",
    )

    # SPRITE-specific arguments
    parser.add_argument(
        "--download-config", required=True,
        help="Path to download_config.yaml",
    )
    parser.add_argument(
        "--experiment-config", required=True,
        help="Path to experiment_config.yaml",
    )

    args = parser.parse_args()

    # Input validation
    for path_arg, label in [
        (args.download_config, "Download config"),
        (args.experiment_config, "Experiment config"),
    ]:
        if not os.path.exists(path_arg):
            print(f"Error: {label} file not found: {path_arg}")
            sys.exit(1)

    logger.info("=" * 70)
    logger.info("SPRITE FL WORKFLOW GENERATOR")
    logger.info("=" * 70)
    logger.info(f"Download config:  {args.download_config}")
    logger.info(f"Experiment config: {args.experiment_config}")
    logger.info(f"Execution site:   {args.execution_site_name}")
    logger.info(f"Output file:      {args.output}")
    logger.info("=" * 70)

    try:
        workflow = SpriteFlWorkflow(dagfile=args.output)
        workflow.download_config_path = args.download_config
        workflow.experiment_config_path = args.experiment_config
        workflow.load_configs()

        logger.info(f"Sites: {workflow.sites}")

        workflow.create_pegasus_properties()

        if not args.skip_sites_catalog:
            workflow.create_sites_catalog(
                exec_site_name=args.execution_site_name
            )

        workflow.create_transformation_catalog(
            exec_site_name=args.execution_site_name
        )
        workflow.create_replica_catalog()
        workflow.create_workflow()
        workflow.write()

        logger.info(f"\nWorkflow written to {args.output}")
        logger.info(
            f"Submit: pegasus-plan --submit "
            f"-s {args.execution_site_name} -o local {args.output}"
        )

    except Exception as e:
        logger.error(f"Failed to generate workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
