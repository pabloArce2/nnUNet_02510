#!/usr/bin/env bash
set -euo pipefail

# MONAI pipeline wrapper.
# Runs SSL always; supervised stages run only when LABEL_DIR is provided.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-$ROOT_DIR/pipeline/work/monai_all}"
INPUT_NPY_DIR="${INPUT_NPY_DIR:-$ROOT_DIR/dataset/pigs_npy}"
INPUT_NIFTI_DIR="${INPUT_NIFTI_DIR:-}"
DEVICE="${DEVICE:-auto}"
LABEL_DIR="${LABEL_DIR:-}"
CORRECTED_LABEL_DIR="${CORRECTED_LABEL_DIR:-}"

CT_DIR="$WORK_DIR/ct_nifti"
SPLIT_JSON="$WORK_DIR/splits.json"
SSL_DIR="$WORK_DIR/models/ssl"

mkdir -p "$WORK_DIR"

if [[ -n "$INPUT_NIFTI_DIR" ]]; then
  echo "[1/4] Preparing CT NIfTI from existing NIfTI folder"
  python "$ROOT_DIR/pipeline/scripts/dataset/prepare_nifti_from_nii.py" \
    --input-dir "$INPUT_NIFTI_DIR" \
    --output-dir "$CT_DIR"
else
  echo "[1/4] Preparing CT NIfTI from .npy dataset"
  python "$ROOT_DIR/pipeline/scripts/npy/prepare_nifti_from_npy.py" \
    --input-dir "$INPUT_NPY_DIR" \
    --output-dir "$CT_DIR" \
    --default-spacing 1.0 1.0 1.0 \
    --clip -1024 2000
fi

SPLIT_CMD=(
  python "$ROOT_DIR/pipeline/scripts/monai/create_monai_splits.py"
  --ct-dir "$CT_DIR"
  --output-json "$SPLIT_JSON"
)

if [[ -n "$LABEL_DIR" ]]; then
  SPLIT_CMD+=(--label-dir "$LABEL_DIR")
fi
if [[ -n "$CORRECTED_LABEL_DIR" ]]; then
  SPLIT_CMD+=(--corrected-label-dir "$CORRECTED_LABEL_DIR")
fi

echo "[2/4] Building split file"
"${SPLIT_CMD[@]}"

echo "[3/4] SSL pretraining"
python "$ROOT_DIR/pipeline/scripts/monai/train_monai_ssl.py" \
  --split-json "$SPLIT_JSON" \
  --output-dir "$SSL_DIR" \
  --device "$DEVICE"

if [[ -z "$LABEL_DIR" ]]; then
  echo "[4/4] Skipping supervised stages: LABEL_DIR is not set"
  echo "Done. SSL checkpoint: $SSL_DIR/ssl_best.pt"
  exit 0
fi

BASE_OUT="$WORK_DIR/experiments/baseline"
SSL_OUT="$WORK_DIR/experiments/ssl_finetune"

echo "[4/4] Supervised training (baseline + SSL-init)"

bash "$ROOT_DIR/pipeline/scripts/run/run_monai_supervised_cycle.sh" \
  "$CT_DIR" \
  "$LABEL_DIR" \
  "$BASE_OUT" \
  "$DEVICE"

bash "$ROOT_DIR/pipeline/scripts/run/run_monai_supervised_cycle.sh" \
  "$CT_DIR" \
  "$LABEL_DIR" \
  "$SSL_OUT" \
  "$DEVICE" \
  "$SSL_DIR/ssl_best.pt" \
  "$CORRECTED_LABEL_DIR"

echo "Done. Results under: $WORK_DIR"
