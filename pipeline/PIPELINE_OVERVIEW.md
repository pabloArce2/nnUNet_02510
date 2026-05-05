# Pipeline Overview

## What this pipeline is made for

This project is built to segment pig liver in CT volumes and to support a comparison between three training strategies:

1. Supervised 3D U-Net from scratch
2. Self-supervised (SSL) pretraining + supervised fine-tuning
3. SSL + targeted correction of hard cases

It is designed to still be useful when labels are limited or unavailable.

## High-level workflow

The main workflow is:

1. Standardize CT data to nnUNet-style NIfTI names (`*_0000.nii.gz`)
2. Build split metadata (`splits.json`)
3. Run SSL pretraining on unlabeled CTs
4. Train supervised segmentation (baseline and/or SSL-initialized)
5. Predict on test data
6. Evaluate metrics and mine hard cases for manual correction
7. Retrain with corrected hard-case labels

## Core pipeline pieces

### 1) Data preparation

- `pipeline/scripts/dataset/prepare_nifti_from_nii.py`: standardizes existing NIfTI CT files
- `pipeline/scripts/npy/prepare_nifti_from_npy.py`: converts `.npy` CT arrays to NIfTI and clips HU range
- `pipeline/scripts/monai/create_monai_splits.py`: creates case index and train/val/test split metadata

Output is a CT folder and a `splits.json` used by training and inference scripts.

### 2) SSL pretraining

- `pipeline/scripts/monai/train_monai_ssl.py`

Trains masked-reconstruction SSL on CT volumes without labels. This creates `ssl_best.pt`, which can initialize the supervised model.

### 3) Supervised segmentation cycle

- `pipeline/scripts/monai/train_monai_seg.py`
- `pipeline/scripts/monai/predict_monai_seg.py`
- `pipeline/scripts/analysis/evaluate_segmentation.py`
- Wrapper: `pipeline/scripts/run/run_monai_supervised_cycle.sh`

This cycle trains a liver segmentation model, predicts test masks, and writes per-case evaluation metrics (Dice, IoU, precision, volume errors, HU-based checks).

### 4) Hard-case mining and correction loop

- `pipeline/scripts/analysis/mine_hard_cases.py`
- Wrapper: `pipeline/scripts/run/run_mine_hard_cases.sh`

Ranks difficult cases, prepares review folders, and supports adding corrected labels. Those corrections can be merged into another supervised training run.

## Main run scripts and when to use them

- `pipeline/scripts/run/run_all.sh`: end-to-end wrapper; always runs SSL; runs supervised stages when `LABEL_DIR` is provided
- `pipeline/scripts/run/run_monai_ssl_pretrain.sh`: SSL-only run on unlabeled data
- `pipeline/scripts/run/run_monai_supervised_cycle.sh`: single supervised train/predict/evaluate cycle
- `pipeline/scripts/run/run_liver_stage1_only.sh`: stage-1 pseudo-label path using TotalSegmentator liver masks

## Probability postprocessing path (`.npy`)

For predicted class-probability tensors saved as `.npy`, the pipeline includes postprocessing to build cleaner label maps:

- `pipeline/scripts/npy/postprocess_probs_argmax.py`

What it does:

1. Loads 4D class probabilities and computes argmax labels
2. Fills holes (`morph` or `knn` strategy)
3. Reassigns low-confidence small components
4. Enforces one connected component for pig and liver classes
5. Exports refined NIfTI labels and JSON summaries

This path is useful for turning raw probability outputs into topology-consistent segmentation masks.

## Typical experiment progression

1. Run SSL on all available CTs
2. Train baseline supervised model (from scratch)
3. Train SSL-initialized supervised model
4. Compare test metrics
5. Mine and correct hard cases
6. Retrain with corrections and compare again

This gives a practical way to quantify whether SSL and targeted corrections improve liver segmentation quality under limited-label conditions.
