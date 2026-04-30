#!/usr/bin/env bash
set -euo pipefail

# Runs TotalSegmentator liver pseudo-labeling on this repo dataset.
# Defaults to CPU so it works on laptops without GPU.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEVICE="${1:-cpu}"
CT_NIFTI_DIR="$ROOT_DIR/pipeline/work/ct_nifti"
PSEUDO_DIR="$ROOT_DIR/pipeline/work/pseudolabels"

if ! command -v TotalSegmentator >/dev/null 2>&1; then
  echo "TotalSegmentator command not found. Install with: pip install TotalSegmentator"
  exit 1
fi

echo "[1/2] Convert dataset/*.npy -> NIfTI"
python "$ROOT_DIR/pipeline/scripts/prepare_nifti_from_npy.py" \
  --input-dir "$ROOT_DIR/dataset" \
  --output-dir "$CT_NIFTI_DIR" \
  --default-spacing 1.0 1.0 1.0 \
  --clip -1024 2000

echo "[2/2] Run TotalSegmentator liver on all scans (device=$DEVICE)"
bash "$ROOT_DIR/pipeline/scripts/run_totalseg_liver_batch.sh" \
  "$CT_NIFTI_DIR" \
  "$PSEUDO_DIR" \
  "$DEVICE"

echo "Done. Liver masks are in: $PSEUDO_DIR/liver_masks"
