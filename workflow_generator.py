#!/usr/bin/env python3

"""
Pegasus workflow generator for SPRITE Federated Learning pipeline.

Implements a Pegasus-native pipeline for MRMS radar precipitation data
using tar archives as the data transfer unit between jobs:

  For each site (KBOX, KBYX, KENX, KLGX, KTLX, KVNX, PAHG):
    1. download_{site}  -> {site}_raw.tar.gz
    2. preproc_{site}   -> {site}_sequences.tar.gz
    3. snapshot_{site}   -> {site}_snapshot.tar.gz
  Then:
    4. central_snapshot  -> central_snapshot.tar.gz
    5. prepare_configs   -> fl_configs.tar.gz
    6. finalize_report   -> final_report.json

No absolute paths, no bind mounts, no SQLite, no __DATA_ROOT__ tokens.
All inter-job data flows through explicit Pegasus File objects.

Usage:
    ./workflow_generator.py \\
        --download-config download_config.yaml \\
        --experiment-config experiment_config.yaml \\
        --output workflow.yml
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
    "preproc":           {"memory": "8 GB",  "cores": 2},
    "snapshot":          {"memory": "4 GB",  "cores": 1},
    "central_snapshot":  {"memory": "4 GB",  "cores": 1},
    "prepare_configs":   {"memory": "2 GB",  "cores": 1},
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
        """Load and parse YAML config files directly (no token substitution)."""
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
        self.props["pegasus.data.configuration"] = "condorio"
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

        container = Container(
            "sprite_fl_container",
            container_type=Container.SINGULARITY,
            image="docker://kthare10/sprite-fl:latest",
            image_site="docker_hub",
        )
        # No --bind /home/ubuntu, no DATA_ROOT env var
        self.tc.add_containers(container)

        transformations = []
        for tool_name, config in TOOL_CONFIGS.items():
            tx = Transformation(
                tool_name,
                site=exec_site_name,
                pfn=os.path.join(self.wf_dir, f"bin/{tool_name}.py"),
                is_stageable=True,
                container=container,
            )
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

        # Register config files as input replicas (original files, no token resolution)
        self.rc.add_replica(
            "local", "download_config.yaml",
            "file://" + os.path.abspath(self.download_config_path)
        )
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
        # Per-site fan-out: download -> preproc -> snapshot
        # ============================================================
        site_snapshot_tars = []

        for site in self.sites:
            site_lower = site.lower()

            # File objects for this site's pipeline
            raw_tar = File(f"{site_lower}_raw.tar.gz")
            seq_tar = File(f"{site_lower}_sequences.tar.gz")
            snap_tar = File(f"{site_lower}_snapshot.tar.gz")

            # Step 1: Download MRMS data for this site
            download_job = (
                Job("download", _id=f"download_{site_lower}",
                    node_label=f"download_{site_lower}")
                .add_args(
                    "--config", dl_config,
                    "--site", site,
                    "--output", raw_tar,
                )
                .add_inputs(dl_config)
                .add_outputs(raw_tar, stage_out=False, register_replica=False)
            )

            # Step 2: Preprocess (inventory + spans + freeze + grouping)
            preproc_job = (
                Job("preproc", _id=f"preproc_{site_lower}",
                    node_label=f"preproc_{site_lower}")
                .add_args(
                    "--config", exp_config,
                    "--site", site,
                    "--raw-tar", raw_tar,
                    "--output", seq_tar,
                )
                .add_inputs(exp_config, raw_tar)
                .add_outputs(seq_tar, stage_out=False, register_replica=False)
            )

            # Step 3: Create site snapshot
            snapshot_job = (
                Job("snapshot", _id=f"snapshot_{site_lower}",
                    node_label=f"snapshot_{site_lower}")
                .add_args(
                    "--config", exp_config,
                    "--site", site,
                    "--sequences-tar", seq_tar,
                    "--output", snap_tar,
                )
                .add_inputs(exp_config, seq_tar)
                .add_outputs(snap_tar, stage_out=False, register_replica=False)
            )

            site_snapshot_tars.append(snap_tar)
            self.wf.add_jobs(download_job, preproc_job, snapshot_job)

        # ============================================================
        # Fan-in: Central snapshot — merge all site snapshots
        # ============================================================
        central_tar = File("central_snapshot.tar.gz")

        central_job = (
            Job("central_snapshot", _id="central_snapshot",
                node_label="central_snapshot")
            .add_args("--config", exp_config)
        )
        # Add --site-tars and --sites arguments
        central_job.add_args("--site-tars")
        for snap_tar in site_snapshot_tars:
            central_job.add_args(snap_tar)
        central_job.add_args("--sites")
        for site in self.sites:
            central_job.add_args(site)
        central_job.add_args("--output", central_tar)

        central_job.add_inputs(exp_config, *site_snapshot_tars)
        central_job.add_outputs(central_tar, stage_out=False,
                                register_replica=False)
        self.wf.add_jobs(central_job)

        # ============================================================
        # Prepare FL/centralized training configs
        # ============================================================
        configs_tar = File("fl_configs.tar.gz")

        prepare_job = (
            Job("prepare_configs", _id="prepare_configs",
                node_label="prepare_configs")
            .add_args(
                "--config", exp_config,
                "--central-tar", central_tar,
                "--output", configs_tar,
            )
            .add_inputs(exp_config, central_tar)
            .add_outputs(configs_tar, stage_out=False, register_replica=False)
        )
        self.wf.add_jobs(prepare_job)

        # ============================================================
        # Finalize report
        # ============================================================
        report = File("final_report.json")

        finalize_job = (
            Job("finalize_report", _id="finalize_report",
                node_label="finalize_report")
            .add_args(
                "--config", exp_config,
                "--central-tar", central_tar,
                "--configs-tar", configs_tar,
                "--output", report,
            )
            .add_inputs(exp_config, central_tar, configs_tar)
            .add_outputs(report, stage_out=True, register_replica=False)
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
