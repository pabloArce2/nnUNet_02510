# Pig Binary MONAI Pipeline Overview

## Goal
Segment liver from pig CT volumes using a binary model (`background=0`, `liver=1`) with optional SSL pretraining.

## High-level flow
1. Build split metadata  
   Script: `pipeline/scripts/monai/pig_binary/create_pig_binary_split.py`
2. Wrapper preprocessing and training launch  
   Script: `pipeline/scripts/monai/pig_binary/train_monai_seg_pig_binary.py`
3. Core segmentation training and validation  
   Script: `pipeline/scripts/monai/train_monai_seg.py`
4. Optional SSL pretraining branch  
   Script: `pipeline/scripts/monai/train_monai_ssl.py`

## Inputs
- Labeled CT: `dataset/labeled/animal/`
- Labeled liver masks: `dataset/labeled/liver/`
- Unlabeled CT pool: `dataset/pig_nii_unlabeled/`
- Split JSON: `dataset/labeled/pig_binary/splits_pig_binary.json` (or similar)

## Wrapper preprocessing (`train_monai_seg_pig_binary.py`)
- Rebase image/label paths to current repo
- Canonicalize CT orientation (`nib.as_closest_canonical`)
- Convert labels to binary (`label > 0`)
- Write `split_preprocessed.json`
- Forward training args to `train_monai_seg.py`

## Core training (`train_monai_seg.py`)
- Label/image alignment to same grid (resample + axis-permutation fallback)
- Label preflight sanity checks (shape, affine, foreground)
- Train transforms:
  - Load + channel-first + intensity scaling
  - Optional spacing normalization (`--target-spacing`)
  - Patch sampling:
    - default `RandSpatialCropd`
    - or `RandCropByPosNegLabeld` (`--use-pos-neg-crop`)
  - Augmentations (flip, Gaussian noise)
- Model: MONAI 3D U-Net
- Loss: DiceCE
- Validation: full-volume sliding-window inference

## Key outputs
- Run folder: `pipeline/runs/<run_name>/<timestamp>_seg/`
- Checkpoints:
  - `seg_last.pt`
  - `seg_best.pt`
- Metrics/history:
  - `seg_history.csv`
  - `seg_history.json`
- TensorBoard logs: `<run_dir>/tensorboard`
- Debug patches (optional): `<run_dir>/debug_patches`

## Metrics tracked
- Liver-focused: `val_dice`, `val_iou`, `val_precision`, `val_recall`
- Background/global: `val_specificity`, `val_accuracy`, `val_balanced_accuracy`, `val_dice_bg`, `val_macro_dice`
- Confusion counts: `val_tp`, `val_fp`, `val_fn`, `val_tn`
- Sampling diagnostics: `train_fg_fraction`, `train_class_fraction_c*`

## Optional SSL branch
- Pretrain encoder with masked reconstruction on `splits.ssl_pretrain`
- Use checkpoint for segmentation init with `--init-ssl-checkpoint`

## References
- Metrics + CLI details: `pipeline/docs/monai_metrics_cli_reference.md`
- Diagram source: `pipeline/docs/pig_binary_pipeline_diagram.md`
