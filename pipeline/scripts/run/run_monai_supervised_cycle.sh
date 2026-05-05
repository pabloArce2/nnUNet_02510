#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  run_monai_supervised_cycle.sh <ct_dir> <label_dir> <output_root> [device] [ssl_checkpoint] [corrected_label_dir]

Arguments:
  ct_dir               Folder with *_0000.nii.gz CT files
  label_dir            Folder with weak labels <case_id>.nii.gz (e.g. ilastik)
  output_root          Working directory for this run
  device               auto|cpu|cuda|mps (default: auto)
  ssl_checkpoint       Optional checkpoint from SSL pretraining
  corrected_label_dir  Optional corrected hard cases <case_id>.nii.gz

This runs:
  1) create_monai_splits.py with labels/corrections
  2) train_monai_seg.py (baseline or SSL-initialized)
  3) predict_monai_seg.py on test split
  4) evaluate_segmentation.py
EOF
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="$1"
LABEL_DIR="$2"
OUTPUT_ROOT="$3"
DEVICE="${4:-auto}"
SSL_CKPT="${5:-}"
CORRECTED_DIR="${6:-}"

SPLIT_JSON="$OUTPUT_ROOT/splits.json"
SEG_DIR="$OUTPUT_ROOT/models/seg"

mkdir -p "$OUTPUT_ROOT"

SPLIT_CMD=(
  python "$ROOT_DIR/pipeline/scripts/monai/create_monai_splits.py"
  --ct-dir "$CT_DIR"
  --label-dir "$LABEL_DIR"
  --output-json "$SPLIT_JSON"
)
if [[ -n "$CORRECTED_DIR" ]]; then
  SPLIT_CMD+=(--corrected-label-dir "$CORRECTED_DIR")
fi

echo "[1/4] Building split JSON"
"${SPLIT_CMD[@]}"

SEG_CMD=(
  python "$ROOT_DIR/pipeline/scripts/monai/train_monai_seg.py"
  --split-json "$SPLIT_JSON"
  --output-dir "$SEG_DIR"
  --device "$DEVICE"
)
if [[ -n "$SSL_CKPT" ]]; then
  SEG_CMD+=(--init-ssl-checkpoint "$SSL_CKPT")
fi

echo "[2/4] Training segmentation model"
"${SEG_CMD[@]}"

SEG_RUN_DIR="$SEG_DIR"
if [[ -f "$SEG_DIR/latest_run.txt" ]]; then
  SEG_RUN_DIR="$(cat "$SEG_DIR/latest_run.txt")"
fi
SEG_CKPT="$SEG_RUN_DIR/seg_best.pt"
if [[ ! -f "$SEG_CKPT" ]]; then
  echo "[error] Could not find trained checkpoint: $SEG_CKPT" >&2
  exit 2
fi
RUN_TAG="$(basename "$SEG_RUN_DIR")"
PRED_DIR="$OUTPUT_ROOT/predictions/test/$RUN_TAG"
METRICS_CSV="$OUTPUT_ROOT/metrics/${RUN_TAG}_test_metrics.csv"

echo "[3/4] Predicting test set"
python "$ROOT_DIR/pipeline/scripts/monai/predict_monai_seg.py" \
  --split-json "$SPLIT_JSON" \
  --checkpoint "$SEG_CKPT" \
  --output-dir "$PRED_DIR" \
  --split-name test \
  --save-probability \
  --device "$DEVICE"

echo "[4/4] Evaluating"
python "$ROOT_DIR/pipeline/scripts/analysis/evaluate_segmentation.py" \
  --ct-dir "$CT_DIR" \
  --pred-dir "$PRED_DIR/pred_masks" \
  --label-dir "$LABEL_DIR" \
  --output-csv "$METRICS_CSV"

echo "Done. Metrics at: $METRICS_CSV"
