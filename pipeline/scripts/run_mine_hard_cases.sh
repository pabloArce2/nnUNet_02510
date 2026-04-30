#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<'EOF'
Usage:
  run_mine_hard_cases.sh <ct_dir> <prediction_dir> <output_dir>

Arguments:
  ct_dir          Folder with *_0000.nii.gz CT files
  prediction_dir  Output folder from predict_monai_seg.py (contains pred_masks/ and optional pred_probs/)
  output_dir      Destination for ranked hard cases
EOF
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="$1"
PRED_ROOT="$2"
OUTPUT_DIR="$3"

CMD=(
  python "$ROOT_DIR/pipeline/scripts/mine_hard_cases.py"
  --ct-dir "$CT_DIR"
  --pred-dir "$PRED_ROOT/pred_masks"
  --output-dir "$OUTPUT_DIR"
)

if [[ -d "$PRED_ROOT/pred_probs" ]]; then
  CMD+=(--prob-dir "$PRED_ROOT/pred_probs")
fi

"${CMD[@]}"
