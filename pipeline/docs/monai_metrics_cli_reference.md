# MONAI Pig Binary: Metrics + CLI Reference

This document summarizes:
- all metrics currently implemented in `train_monai_seg.py`
- all relevant CLI arguments for `train_monai_seg_pig_binary.py` (wrapper) and `train_monai_seg.py` (core trainer)
- what your latest shared run used and what each setting means

## 1) Implemented metrics

### Training patch-distribution diagnostics
- `train_fg_fraction`: fraction of foreground voxels in sampled training patches.
- `train_class_fraction_c0`, `train_class_fraction_c1`, ...: per-class voxel fractions in sampled patches.

### Validation metrics (`val_*`)
- `val_loss`: DiceCE loss on validation full volumes.
- `val_dice`: liver-class Dice (class 1).
- `val_iou`: liver-class IoU.
- `val_precision`: liver-class precision.
- `val_recall`: liver-class recall/sensitivity.
- `val_specificity`: background specificity.
- `val_accuracy`: voxel accuracy.
- `val_balanced_accuracy`: `(recall + specificity) / 2`.
- `val_dice_bg`: background-class Dice.
- `val_macro_dice`: `(dice_fg + dice_bg) / 2`.
- `val_tp`, `val_fp`, `val_fn`, `val_tn`: confusion-matrix counts.

### Optional train full-volume metrics (`train_vol_*`)
Only when `--train-full-volume-interval > 0`.
- `train_vol_loss`
- `train_vol_dice`, `train_vol_iou`, `train_vol_precision`
- `train_vol_recall`, `train_vol_specificity`, `train_vol_accuracy`
- `train_vol_balanced_accuracy`, `train_vol_dice_bg`, `train_vol_macro_dice`
- `train_vol_tp`, `train_vol_fp`, `train_vol_fn`, `train_vol_tn`

### Where metrics are saved
- Console logs each epoch.
- TensorBoard scalars (if not disabled).
- `seg_history.csv` and `seg_history.json` in run dir.

## 2) CLI arguments

## 2.1 Wrapper: `pipeline/scripts/monai/pig_binary/train_monai_seg_pig_binary.py`

### Wrapper-specific
- `--split-json` (required): input split file.
- `--output-dir` (required): run output root.
- `--preprocessed-split-json`: where wrapper writes prepared split (default `<output-dir>/split_preprocessed.json`).
- `--skip-image-preprocess`: skip CT canonicalization.
- `--run-mode {normal,loocv}`:
  - `normal`: one training run.
  - `loocv`: one run per labeled hold-out case.

### Behavior
- Rebases paths to current repo.
- Canonicalizes CT with `nib.as_closest_canonical` unless skipped.
- Converts labels to binary `(label > 0)`.
- Writes preprocessed split JSON.
- Forwards all extra args directly to `train_monai_seg.py`.

## 2.2 Core trainer: `pipeline/scripts/monai/train_monai_seg.py`

### I/O and run control
- `--split-json` (required): training split JSON.
- `--output-dir` (required): root for outputs.
- `--experiment-name`: tag in timestamped run folder.
- `--flat-output`: write directly into `--output-dir`.
- `--seed`: random seed.
- `--device {auto,cpu,cuda,mps}`.

### Optimization and schedule
- `--epochs`
- `--batch-size`
- `--num-workers`
- `--lr`
- `--weight-decay`

### Model/task
- `--num-classes`: includes background (binary = 2).
- `--ce-weights`: class weights for CE term inside DiceCE.
- `--init-ssl-checkpoint`: initialize matching encoder/model weights from SSL checkpoint.

### Patching/sampling
- `--patch-size X Y Z`
- `--use-pos-neg-crop`: switch from `RandSpatialCropd` to `RandCropByPosNegLabeld`.
- `--pos-neg-ratio`: positive:negative center ratio for label-aware crop.
- `--train-num-samples`: number of sampled patches per item when pos-neg crop is on.

### Intensity and augmentation
- `--a-min`, `--a-max`: CT clipping window for `ScaleIntensityRanged` to [0, 1].
- `--flip-prob`: per-axis random flip probability.
- `--gaussian-noise-prob`

### Geometry normalization
- `--target-spacing SX SY SZ`: optional `Spacingd` resampling for image+label.

### Validation/evaluation cadence
- `--val-interval`: validate every N epochs.
- `--train-full-volume-interval`: evaluate full-volume metrics on train split every N epochs (`0` disables).

### TensorBoard/debug outputs
- `--no-tensorboard`
- `--tensorboard-dir`
- `--tensorboard-port`
- `--save-random-patch-every-epoch` / `--no-save-random-patch-every-epoch`

### Label override / H5 preprocessing options
- `--preprocessed-label-dir`
- `--h5-label-dir`, `--preprocess-output-dir`
- `--h5-image-key`, `--h5-label-key`
- `--h5-label-threshold`, `--h5-liver-label`
- `--h5-axis-order {zyx,xyz}`
- `--h5-source-spacing SX SY SZ`
- `--h5-target-spacing TX TY TZ`
- `--h5-clean-min-island-voxels`
- `--h5-closing-radius`
- `--h5-opening-radius`
- `--h5-fill-holes`

## 3) Last run (from your shared command/logs)

The latest configuration you shared in chat corresponds to:

```bash
python .../train_monai_seg_pig_binary.py \
  --split-json .../splits_pig_binary_onecase_debug.json \
  --output-dir .../pipeline/runs/monai_pig_binary_debug_onecase \
  --device cuda \
  --epochs 120 \
  --batch-size 1 \
  --num-workers 2 \
  --patch-size 128 128 128 \
  --ce-weights 1.0 1.0 \
  --lr 5e-5 \
  --weight-decay 1e-5 \
  --val-interval 1 \
  --train-full-volume-interval 0 \
  --flip-prob 0.0 \
  --gaussian-noise-prob 0.0 \
  --use-pos-neg-crop \
  --pos-neg-ratio 1 \
  --train-num-samples 8
```

### Meaning of that setup
- One-case debug/overfit style run (`batch-size 1`, frequent validation).
- No augmentation noise/flips to simplify debugging.
- Balanced positive/negative sampling (`pos-neg-ratio 1`) with moderate patch multiplicity (`train-num-samples 8`).
- Equal CE class weights (`1.0, 1.0`).
- No periodic full-volume train eval (`train-full-volume-interval 0`).

## 4) Practical metric interpretation guide

- `val_dice` low + `val_specificity` very low: likely overpredicting liver.
- `train_fg_fraction` near 0: sampling mostly background patches.
- `train_fg_fraction` very high with tiny true FG volume: sampling too liver-heavy.
- Rising `val_macro_dice` + rising `val_specificity`: usually healthier class balance.

