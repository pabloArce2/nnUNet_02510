#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  run_monai_ssl_pretrain.sh <input_nifti_dir> [output_root] [device]

Arguments:
  input_nifti_dir  Folder with CT .nii/.nii.gz files
  output_root      Output root (default: pipeline/work/monai)
  device           auto|cpu|cuda|mps (default: auto)

This runs:
  1) prepare_nifti_from_nii.py
  2) create_monai_splits.py (no labels required)
  3) train_monai_ssl.py
EOF
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_NIFTI_DIR="$1"
OUTPUT_ROOT="${2:-$ROOT_DIR/pipeline/work/monai}"
DEVICE="${3:-auto}"

CT_DIR="$OUTPUT_ROOT/ct_nifti"
SPLIT_JSON="$OUTPUT_ROOT/splits.json"
SSL_OUT="$OUTPUT_ROOT/models/ssl"

mkdir -p "$OUTPUT_ROOT"

echo "[1/3] Preparing CT files"
python "$ROOT_DIR/pipeline/scripts/prepare_nifti_from_nii.py" \
  --input-dir "$INPUT_NIFTI_DIR" \
  --output-dir "$CT_DIR"

echo "[2/3] Building split JSON"
python "$ROOT_DIR/pipeline/scripts/create_monai_splits.py" \
  --ct-dir "$CT_DIR" \
  --output-json "$SPLIT_JSON"

echo "[3/3] SSL pretraining"
python "$ROOT_DIR/pipeline/scripts/train_monai_ssl.py" \
  --split-json "$SPLIT_JSON" \
  --output-dir "$SSL_OUT" \
  --device "$DEVICE"

SSL_RUN_DIR="$SSL_OUT"
if [[ -f "$SSL_OUT/latest_run.txt" ]]; then
  SSL_RUN_DIR="$(cat "$SSL_OUT/latest_run.txt")"
fi
echo "Done. SSL model: $SSL_RUN_DIR/ssl_best.pt"
