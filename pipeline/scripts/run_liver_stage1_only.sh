#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  run_liver_stage1_only.sh <input_nifti_dir> [output_root] [device]

Arguments:
  input_nifti_dir  Folder with input CT files (.nii or .nii.gz)
  output_root      Output root folder (default: pipeline/work/stage1)
  device           cpu|gpu|mps (default: cpu)

What this does:
  1) Standardize input files to *_0000.nii.gz
  2) Run TotalSegmentator (liver only)
  3) Save examples with labels as binary masks (0 background, 1 liver)

Outputs:
  <output_root>/ct_nifti/*.nii.gz
  <output_root>/pseudolabels/liver_masks/*.nii.gz
  <output_root>/examples/images/*_0000.nii.gz
  <output_root>/examples/labels/*.nii.gz
EOF
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_NIFTI_DIR="$1"
OUTPUT_ROOT="${2:-$ROOT_DIR/pipeline/work/stage1}"
DEVICE="${3:-cpu}"

CT_NIFTI_DIR="$OUTPUT_ROOT/ct_nifti"
PSEUDO_DIR="$OUTPUT_ROOT/pseudolabels"
EXAMPLES_DIR="$OUTPUT_ROOT/examples"

if ! command -v TotalSegmentator >/dev/null 2>&1; then
  echo "TotalSegmentator command not found. Install with: pip install TotalSegmentator"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"

echo "[1/3] Prepare input NIfTI files"
python "$ROOT_DIR/pipeline/scripts/prepare_nifti_from_nii.py" \
  --input-dir "$INPUT_NIFTI_DIR" \
  --output-dir "$CT_NIFTI_DIR"

echo "[2/3] Run TotalSegmentator liver-only (device=$DEVICE)"
bash "$ROOT_DIR/pipeline/scripts/run_totalseg_liver_batch.sh" \
  "$CT_NIFTI_DIR" \
  "$PSEUDO_DIR" \
  "$DEVICE"

echo "[3/3] Build minimal examples (0 background, 1 liver)"
python "$ROOT_DIR/pipeline/scripts/build_liver_examples.py" \
  --ct-nifti-dir "$CT_NIFTI_DIR" \
  --liver-mask-dir "$PSEUDO_DIR/liver_masks" \
  --output-dir "$EXAMPLES_DIR" \
  --overwrite

echo "Done. Examples saved in: $EXAMPLES_DIR"
