#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  run_totalseg_liver_batch.sh <input_nifti_dir> <output_dir> [device]

Arguments:
  input_nifti_dir  Directory with *_0000.nii.gz CT files
  output_dir       Directory where TotalSegmentator outputs are written
  device           cpu|gpu|mps (default: cpu)

Outputs:
  <output_dir>/totalseg_raw/<case_id>/...   # raw TotalSegmentator outputs
  <output_dir>/liver_masks/<case_id>.nii.gz # extracted liver masks
EOF
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
DEVICE="${3:-cpu}"
RAW_DIR="$OUTPUT_DIR/totalseg_raw"
LIVER_DIR="$OUTPUT_DIR/liver_masks"

mkdir -p "$RAW_DIR" "$LIVER_DIR"
shopt -s nullglob

for ct in "$INPUT_DIR"/*_0000.nii.gz; do
  base="$(basename "$ct")"
  case_id="${base%_0000.nii.gz}"
  case_out="$RAW_DIR/$case_id"
  out_mask="$LIVER_DIR/$case_id.nii.gz"

  if [[ -f "$out_mask" ]]; then
    echo "[skip] $case_id (liver mask exists)"
    continue
  fi

  mkdir -p "$case_out"
  echo "[run] $case_id"

  TotalSegmentator \
    -i "$ct" \
    -o "$case_out" \
    --roi_subset liver \
    --device "$DEVICE"

  liver_candidate=""
  if [[ -f "$case_out/liver.nii.gz" ]]; then
    liver_candidate="$case_out/liver.nii.gz"
  elif [[ -f "$case_out/segmentations/liver.nii.gz" ]]; then
    liver_candidate="$case_out/segmentations/liver.nii.gz"
  else
    # fallback: find any liver.nii.gz recursively
    found="$(find "$case_out" -type f -name 'liver.nii.gz' | head -n 1 || true)"
    if [[ -n "$found" ]]; then
      liver_candidate="$found"
    fi
  fi

  if [[ -z "$liver_candidate" || ! -f "$liver_candidate" ]]; then
    echo "[error] Could not find liver.nii.gz for $case_id in $case_out" >&2
    exit 2
  fi

  cp "$liver_candidate" "$out_mask"
  echo "[ok] $out_mask"
done

count=$(find "$LIVER_DIR" -maxdepth 1 -type f -name '*.nii.gz' | wc -l)
echo "Done. Liver masks: $count"
