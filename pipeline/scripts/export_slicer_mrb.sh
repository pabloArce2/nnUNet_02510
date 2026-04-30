#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  export_slicer_mrb.sh <slicer_executable> [stage1_root] [overwrite]

Arguments:
  slicer_executable  Path/name of Slicer executable (e.g. Slicer)
  stage1_root        Stage-1 root folder (default: pipeline/work/stage1)
  overwrite          true|false (default: false)

What it does:
  Creates one .mrb per matched case from:
    <stage1_root>/ct_nifti/*_0000.nii.gz
    <stage1_root>/pseudolabels/liver_masks/*.nii.gz

Output:
  <stage1_root>/slicer_mrb/<case_id>.mrb
EOF
  exit 1
fi

SLICER_EXE="$1"
STAGE1_ROOT="${2:-pipeline/work/stage1}"
OVERWRITE="${3:-false}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CT_DIR="$ROOT_DIR/$STAGE1_ROOT/ct_nifti"
MASK_DIR="$ROOT_DIR/$STAGE1_ROOT/pseudolabels/liver_masks"
OUT_DIR="$ROOT_DIR/$STAGE1_ROOT/slicer_mrb"
SCRIPT_PATH="$ROOT_DIR/pipeline/scripts/export_slicer_mrb_batch.py"

if [[ ! -d "$CT_DIR" ]]; then
  echo "CT directory not found: $CT_DIR" >&2
  exit 1
fi
if [[ ! -d "$MASK_DIR" ]]; then
  echo "Mask directory not found: $MASK_DIR" >&2
  exit 1
fi

cmd=(
  "$SLICER_EXE"
  --no-main-window
  --python-script "$SCRIPT_PATH"
  --
  --ct-dir "$CT_DIR"
  --mask-dir "$MASK_DIR"
  --output-dir "$OUT_DIR"
  --skip-empty-masks
)

if [[ "$OVERWRITE" == "true" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"

echo "MRB scenes saved in: $OUT_DIR"
