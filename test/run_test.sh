#!/usr/bin/env bash
#
# Test the SPRITE pipeline end-to-end with minimal data.
#
# Uses 1 site (KBOX), 1 day, hourly steps (~24 download tasks).
# Runs each bin/*.py stage locally (no Pegasus scheduler needed).
#
# Usage:
#   cd sprite-workflow
#   bash test/run_test.sh
#
# Requirements:
#   pip install -r requirements.txt matplotlib
#   Internet access for S3 downloads (or skip download with --skip-download)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DIR="$WORKFLOW_DIR/test"
WORK_DIR="$WORKFLOW_DIR/test/workdir"

DL_CONFIG="$TEST_DIR/test_download_config.yaml"
EXP_CONFIG="$TEST_DIR/test_experiment_config.yaml"

SKIP_DOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
    esac
done

SEP="$(printf '=%.0s' {1..60})"

step() {
    echo ""
    echo "$SEP"
    echo "  STEP: $1"
    echo "$SEP"
}

# Clean previous test run
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ---- Step 0: Generate Pegasus workflow YAML (validation only) ----
step "Generate workflow YAML (validation)"
python3 "$WORKFLOW_DIR/workflow_generator.py" \
    --download-config "$DL_CONFIG" \
    --experiment-config "$EXP_CONFIG" \
    --skip-sites-catalog \
    --output "$WORK_DIR/workflow.yml"
echo "  Generated: $WORK_DIR/workflow.yml"
echo "  Jobs in DAG:"
grep -c 'type: job' "$WORK_DIR/workflow.yml" || echo "  (could not count)"

# ---- Step 1: Download ----
SITE="KBOX"
SITE_LOWER="kbox"
RAW_TAR="$WORK_DIR/${SITE_LOWER}_raw.tar.gz"

if [ "$SKIP_DOWNLOAD" = true ] && [ -f "$RAW_TAR" ]; then
    step "Download (SKIPPED — using existing $RAW_TAR)"
else
    step "Download KBOX (1 day, hourly)"
    python3 "$WORKFLOW_DIR/bin/download.py" \
        --config "$DL_CONFIG" \
        --site "$SITE" \
        --output "$RAW_TAR"
fi
echo "  Output: $RAW_TAR ($(du -h "$RAW_TAR" | cut -f1))"

# ---- Step 2: Preprocess ----
step "Preprocess KBOX"
SEQ_TAR="$WORK_DIR/${SITE_LOWER}_sequences.tar.gz"
python3 "$WORKFLOW_DIR/bin/preproc.py" \
    --config "$EXP_CONFIG" \
    --site "$SITE" \
    --raw-tar "$RAW_TAR" \
    --output "$SEQ_TAR"
echo "  Output: $SEQ_TAR ($(du -h "$SEQ_TAR" | cut -f1))"

# ---- Step 3: Snapshot ----
step "Snapshot KBOX"
SNAP_TAR="$WORK_DIR/${SITE_LOWER}_snapshot.tar.gz"
python3 "$WORKFLOW_DIR/bin/snapshot.py" \
    --config "$EXP_CONFIG" \
    --site "$SITE" \
    --sequences-tar "$SEQ_TAR" \
    --output "$SNAP_TAR"
echo "  Output: $SNAP_TAR ($(du -h "$SNAP_TAR" | cut -f1))"

# ---- Step 4: Central Snapshot ----
step "Central Snapshot"
CENTRAL_TAR="$WORK_DIR/central_snapshot.tar.gz"
python3 "$WORKFLOW_DIR/bin/central_snapshot.py" \
    --config "$EXP_CONFIG" \
    --site-tars "$SNAP_TAR" \
    --sites "$SITE" \
    --output "$CENTRAL_TAR"
echo "  Output: $CENTRAL_TAR ($(du -h "$CENTRAL_TAR" | cut -f1))"

# ---- Step 5: Prepare Configs ----
step "Prepare Configs"
CONFIGS_TAR="$WORK_DIR/fl_configs.tar.gz"
python3 "$WORKFLOW_DIR/bin/prepare_configs.py" \
    --config "$EXP_CONFIG" \
    --central-tar "$CENTRAL_TAR" \
    --output "$CONFIGS_TAR"
echo "  Output: $CONFIGS_TAR ($(du -h "$CONFIGS_TAR" | cut -f1))"

# ---- Step 6: Model Compare ----
step "Model Compare Run"
MODEL_CMP="$WORK_DIR/model_comparison.json"
python3 "$WORKFLOW_DIR/bin/model_compare_run.py" \
    --config "$EXP_CONFIG" \
    --configs-tar "$CONFIGS_TAR" \
    --central-tar "$CENTRAL_TAR" \
    --output "$MODEL_CMP"
echo "  Output: $MODEL_CMP"

# ---- Step 7: Visual Compare ----
step "Visual Compare"
VIS_CMP="$WORK_DIR/visual_comparison.html"
python3 "$WORKFLOW_DIR/bin/visual_compare.py" \
    --config "$EXP_CONFIG" \
    --comparison "$MODEL_CMP" \
    --central-tar "$CENTRAL_TAR" \
    --output "$VIS_CMP"
echo "  Output: $VIS_CMP ($(du -h "$VIS_CMP" | cut -f1))"

# ---- Step 8: Finalize Report ----
step "Finalize Report"
REPORT="$WORK_DIR/final_report.json"
python3 "$WORKFLOW_DIR/bin/finalize_report.py" \
    --config "$EXP_CONFIG" \
    --central-tar "$CENTRAL_TAR" \
    --configs-tar "$CONFIGS_TAR" \
    --output "$REPORT"
echo "  Output: $REPORT"
python3 -m json.tool "$REPORT"

# ---- Step 9: Visualize ----
step "Visualize (Pipeline Report)"
VIZ="$WORK_DIR/pipeline_report.html"
python3 "$WORKFLOW_DIR/bin/visualize.py" \
    --config "$EXP_CONFIG" \
    --central-tar "$CENTRAL_TAR" \
    --configs-tar "$CONFIGS_TAR" \
    --output "$VIZ"
echo "  Output: $VIZ ($(du -h "$VIZ" | cut -f1))"

# ---- Summary ----
step "TEST COMPLETE"
echo ""
echo "  All outputs in: $WORK_DIR/"
echo ""
ls -lh "$WORK_DIR/"
echo ""
echo "  To view HTML reports:"
echo "    open $VIS_CMP"
echo "    open $VIZ"
echo ""
echo "  To re-run without re-downloading:"
echo "    bash test/run_test.sh --skip-download"
