#!/usr/bin/env bash
#
# Compare SPRITE workflow output against a reference dataset.
#
# Usage:
#   ./bin/compare_output.sh <output_dir> <reference_dir>
#
# Example (local):
#   ./bin/compare_output.sh /home/ubuntu/data/MRMS/S3_V2 /home/ubuntu/S3_V2
#
# Example (via SSH):
#   ssh pegasus 'bash -s' < bin/compare_output.sh /home/ubuntu/data/MRMS/S3_V2 /home/ubuntu/S3_V2
#

set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 <output_dir> <reference_dir>}"
REF_DIR="${2:?Usage: $0 <output_dir> <reference_dir>}"

OUTPUT_DB="${OUTPUT_DIR}/_meta/sprite.sqlite"
REF_DB="${REF_DIR}/_meta/sprite.sqlite"

SEP="$(printf '=%.0s' {1..70})"

section() {
    echo ""
    echo "$SEP"
    echo "  $1"
    echo "$SEP"
}

compare_query() {
    local label="$1"
    local query="$2"
    echo ""
    echo "--- $label ---"
    echo "  OUTPUT:"
    sqlite3 "$OUTPUT_DB" "$query" 2>/dev/null | sed 's/^/    /' || echo "    (query failed)"
    echo "  REFERENCE:"
    sqlite3 "$REF_DB" "$query" 2>/dev/null | sed 's/^/    /' || echo "    (query failed)"
}

# ---- Header ----
echo "$SEP"
echo "  SPRITE Workflow Comparison"
echo "  Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  Output:    $OUTPUT_DIR"
echo "  Reference: $REF_DIR"
echo "$SEP"

# ---- 1. SQLite Table Row Counts ----
section "1. SQLite Table Row Counts"
compare_query "raw_files" "SELECT 'raw_files', COUNT(*) FROM raw_files;"
compare_query "expected" "SELECT 'expected', COUNT(*) FROM expected;"
compare_query "freezes" "SELECT 'freezes', COUNT(*) FROM freezes;"
compare_query "seq_index" "SELECT 'seq_index', COUNT(*) FROM seq_index;"
compare_query "jobs" "SELECT 'jobs', COUNT(*) FROM jobs;"

# ---- 2. Raw Files per Site ----
section "2. Raw Files per Site"
compare_query "raw_files GROUP BY site" "SELECT site, COUNT(*) FROM raw_files GROUP BY site ORDER BY site;"

# ---- 3. Raw Files per Month ----
section "3. Raw Files per Month"
compare_query "raw_files GROUP BY month" "SELECT SUBSTR(day,1,6) AS month, COUNT(*) FROM raw_files GROUP BY month ORDER BY month;"

# ---- 4. Splits in raw_files ----
section "4. Splits"
compare_query "DISTINCT split" "SELECT DISTINCT split FROM raw_files ORDER BY split;"

# ---- 5. Freezes Table ----
section "5. Freezes"
compare_query "freezes" "SELECT period_type, period_key, site, split, version, status FROM freezes ORDER BY site, split, period_key;"

# ---- 6. Seq Index per Site/Split ----
section "6. Seq Index per Site/Split"
compare_query "seq_index GROUP BY site, split" "SELECT site, split, COUNT(*) FROM seq_index GROUP BY site, split ORDER BY site, split;"

# ---- 7. Jobs Table ----
section "7. Jobs"
compare_query "jobs" "SELECT id, kind, window, span, site, status, retries FROM jobs ORDER BY id;"

# ---- 8. Raw Store Directory Sites ----
section "8. Raw Store Sites"
echo "  OUTPUT:"
ls "$OUTPUT_DIR/raw_store/" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"
echo "  REFERENCE:"
ls "$REF_DIR/raw_store/" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"

# ---- 9. Top-Level Directory Structure ----
section "9. Top-Level Directory Structure"
echo "  OUTPUT:"
ls "$OUTPUT_DIR/" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"
echo "  REFERENCE:"
ls "$REF_DIR/" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"

# ---- 10. Sequence Naming Sample (first site/train) ----
section "10. Sequence Naming Sample"
compare_query "seq_index sample (train, first 5)" "SELECT seq_name FROM seq_index WHERE split='train' ORDER BY site, seq_name LIMIT 5;"

# ---- 11. Freeze Manifests (sample) ----
section "11. Freeze Manifest Sample"
SAMPLE_MANIFEST_OUT=$(find "$OUTPUT_DIR/_meta/freeze_manifests" -name MANIFEST.json 2>/dev/null | head -1)
SAMPLE_MANIFEST_REF=$(find "$REF_DIR/_meta/freeze_manifests" -name MANIFEST.json 2>/dev/null | head -1)
echo "  OUTPUT: ${SAMPLE_MANIFEST_OUT:-not found}"
if [ -n "$SAMPLE_MANIFEST_OUT" ]; then
    python3 -c "import json; m=json.load(open('$SAMPLE_MANIFEST_OUT')); print(json.dumps({k:v for k,v in m.items() if k != 'missing'}, indent=2))" 2>/dev/null | sed 's/^/    /' || echo "    (parse failed)"
fi
echo "  REFERENCE: ${SAMPLE_MANIFEST_REF:-not found}"
if [ -n "$SAMPLE_MANIFEST_REF" ]; then
    python3 -c "import json; m=json.load(open('$SAMPLE_MANIFEST_REF')); print(json.dumps({k:v for k,v in m.items() if k != 'missing'}, indent=2))" 2>/dev/null | sed 's/^/    /' || echo "    (parse failed)"
fi

# ---- 12. Snapshot Manifests (sample) ----
section "12. Site Snapshot Manifest Sample"
SNAP_OUT=$(find "$OUTPUT_DIR/snapshots/sites" -name MANIFEST.json 2>/dev/null | head -1)
SNAP_REF=$(find "$REF_DIR/snapshots/sites" -name MANIFEST.json 2>/dev/null | head -1)
echo "  OUTPUT: ${SNAP_OUT:-not found}"
if [ -n "$SNAP_OUT" ]; then
    cat "$SNAP_OUT" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' || echo "    (parse failed)"
fi
echo "  REFERENCE: ${SNAP_REF:-not found}"
if [ -n "$SNAP_REF" ]; then
    cat "$SNAP_REF" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' || echo "    (parse failed)"
fi

# ---- 13. Central Snapshot Manifest (sample) ----
section "13. Central Snapshot Manifest Sample"
CSNAP_OUT=$(find "$OUTPUT_DIR/snapshots/central" -name MANIFEST.json 2>/dev/null | head -1)
CSNAP_REF=$(find "$REF_DIR/snapshots/central" -name MANIFEST.json 2>/dev/null | head -1)
echo "  OUTPUT: ${CSNAP_OUT:-not found}"
if [ -n "$CSNAP_OUT" ]; then
    cat "$CSNAP_OUT" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' || echo "    (parse failed)"
fi
echo "  REFERENCE: ${CSNAP_REF:-not found}"
if [ -n "$CSNAP_REF" ]; then
    cat "$CSNAP_REF" | python3 -m json.tool 2>/dev/null | sed 's/^/    /' || echo "    (parse failed)"
fi

# ---- 14. FL Run Configs ----
section "14. FL Run Config"
FL_OUT=$(find "$OUTPUT_DIR/runs/fl" -name config.yaml 2>/dev/null | head -1)
FL_REF=$(find "$REF_DIR/runs/fl" -name config.yaml 2>/dev/null | head -1)
echo "  OUTPUT: ${FL_OUT:-not found}"
[ -n "$FL_OUT" ] && cat "$FL_OUT" | sed 's/^/    /' || echo "    (not found)"
echo "  REFERENCE: ${FL_REF:-not found}"
[ -n "$FL_REF" ] && cat "$FL_REF" | sed 's/^/    /' || echo "    (not found)"

# ---- 15. clients.map.json ----
section "15. clients.map.json"
CM_OUT=$(find "$OUTPUT_DIR/runs/fl" -name clients.map.json 2>/dev/null | head -1)
CM_REF=$(find "$REF_DIR/runs/fl" -name clients.map.json 2>/dev/null | head -1)
echo "  OUTPUT: ${CM_OUT:-not found}"
[ -n "$CM_OUT" ] && python3 -m json.tool "$CM_OUT" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"
echo "  REFERENCE: ${CM_REF:-not found}"
[ -n "$CM_REF" ] && python3 -m json.tool "$CM_REF" 2>/dev/null | sed 's/^/    /' || echo "    (not found)"

# ---- 16. flwr.override.toml ----
section "16. flwr.override.toml"
FO_OUT=$(find "$OUTPUT_DIR/runs/fl" -name flwr.override.toml 2>/dev/null | head -1)
FO_REF=$(find "$REF_DIR/runs/fl" -name flwr.override.toml 2>/dev/null | head -1)
echo "  OUTPUT: ${FO_OUT:-not found}"
[ -n "$FO_OUT" ] && cat "$FO_OUT" | sed 's/^/    /' || echo "    (not found)"
echo "  REFERENCE: ${FO_REF:-not found}"
[ -n "$FO_REF" ] && cat "$FO_REF" | sed 's/^/    /' || echo "    (not found)"

# ---- 17. .nc File Sizes (sample) ----
section "17. .nc File Sizes (sample from first site/train day)"
FIRST_SITE=$(ls "$OUTPUT_DIR/raw_store/" 2>/dev/null | head -1)
if [ -n "$FIRST_SITE" ]; then
    FIRST_DAY_OUT=$(ls "$OUTPUT_DIR/raw_store/$FIRST_SITE/train/" 2>/dev/null | head -1)
    FIRST_DAY_REF=$(ls "$REF_DIR/raw_store/$FIRST_SITE/train/" 2>/dev/null | head -1)
    echo "  OUTPUT ($FIRST_SITE/train/${FIRST_DAY_OUT:-?}):"
    [ -n "$FIRST_DAY_OUT" ] && ls -la "$OUTPUT_DIR/raw_store/$FIRST_SITE/train/$FIRST_DAY_OUT/"*.nc 2>/dev/null | head -5 | sed 's/^/    /' || echo "    (no files)"
    echo "  REFERENCE ($FIRST_SITE/train/${FIRST_DAY_REF:-?}):"
    [ -n "$FIRST_DAY_REF" ] && ls -la "$REF_DIR/raw_store/$FIRST_SITE/train/$FIRST_DAY_REF/"*.nc 2>/dev/null | head -5 | sed 's/^/    /' || echo "    (no files)"
fi

# ---- Summary ----
section "SUMMARY"
OUT_RAW=$(sqlite3 "$OUTPUT_DB" "SELECT COUNT(*) FROM raw_files;" 2>/dev/null || echo "?")
REF_RAW=$(sqlite3 "$REF_DB" "SELECT COUNT(*) FROM raw_files;" 2>/dev/null || echo "?")
OUT_SEQ=$(sqlite3 "$OUTPUT_DB" "SELECT COUNT(*) FROM seq_index;" 2>/dev/null || echo "?")
REF_SEQ=$(sqlite3 "$REF_DB" "SELECT COUNT(*) FROM seq_index;" 2>/dev/null || echo "?")
OUT_FREEZE=$(sqlite3 "$OUTPUT_DB" "SELECT COUNT(*) FROM freezes;" 2>/dev/null || echo "?")
REF_FREEZE=$(sqlite3 "$REF_DB" "SELECT COUNT(*) FROM freezes;" 2>/dev/null || echo "?")
OUT_JOBS=$(sqlite3 "$OUTPUT_DB" "SELECT COUNT(*) FROM jobs;" 2>/dev/null || echo "?")
REF_JOBS=$(sqlite3 "$REF_DB" "SELECT COUNT(*) FROM jobs;" 2>/dev/null || echo "?")

printf "  %-15s %10s %10s\n" "Metric" "Output" "Reference"
printf "  %-15s %10s %10s\n" "----------" "------" "---------"
printf "  %-15s %10s %10s\n" "raw_files" "$OUT_RAW" "$REF_RAW"
printf "  %-15s %10s %10s\n" "seq_index" "$OUT_SEQ" "$REF_SEQ"
printf "  %-15s %10s %10s\n" "freezes" "$OUT_FREEZE" "$REF_FREEZE"
printf "  %-15s %10s %10s\n" "jobs" "$OUT_JOBS" "$REF_JOBS"

echo ""
echo "Done."
