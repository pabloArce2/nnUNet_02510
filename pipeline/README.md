# MONAI Pipeline: SSL -> Segmentation -> Targeted Corrections

This pipeline implements your intended comparison:

1. `Model 1`: 3D U-Net from scratch
2. `Model 2`: SSL-pretrained 3D U-Net
3. `Model 3`: SSL-pretrained 3D U-Net + targeted corrected hard cases

It is designed to run even when you currently have zero labels.

## 1) Install

Install PyTorch first (matching your CUDA/CPU setup), then:

```bash
pip install -r pipeline/requirements.txt
```

## 2) Run self-supervision now (no labels needed)

```bash
bash pipeline/scripts/run_monai_ssl_pretrain.sh dataset/pig_nii_unlabeled
```

Default outputs:

- `pipeline/work/monai/ct_nifti/`
- `pipeline/work/monai/splits.json`
- `pipeline/work/monai/models/ssl/ssl_best.pt`
- `pipeline/work/monai/models/ssl/tensorboard/` (training metrics for TensorBoard)

Notes:

- All CTs are used for SSL via masked reconstruction.
- Input scanning is recursive, so files inside subfolders are included automatically.
- Supervised `train/val/test` in `splits.json` will be empty until labels exist.
- Split assignment defaults to `stable_hash`, so train/val/test stays consistent across reruns.

Monitor SSL training in TensorBoard (run in a second terminal while training is running):

```bash
tensorboard --logdir pipeline/work/monai/models/ssl/tensorboard --port 6006
```

Then open: `http://localhost:6006`

## 3) Add ilastik labels and train segmentation

Place labels as:

- `<label_dir>/<case_id>.nii.gz`

where `<case_id>` matches CT name without `_0000.nii.gz`.

### Baseline (from scratch)

```bash
bash pipeline/scripts/run_monai_supervised_cycle.sh \
  pipeline/work/monai/ct_nifti \
  path/to/ilastik_labels \
  pipeline/work/monai_baseline
```

### SSL-initialized

```bash
bash pipeline/scripts/run_monai_supervised_cycle.sh \
  pipeline/work/monai/ct_nifti \
  path/to/ilastik_labels \
  pipeline/work/monai_ssl_finetune \
  auto \
  pipeline/work/monai/models/ssl/ssl_best.pt
```

## 4) Mine hard cases for targeted correction

```bash
bash pipeline/scripts/run_mine_hard_cases.sh \
  pipeline/work/monai/ct_nifti \
  pipeline/work/monai_ssl_finetune/predictions/test \
  pipeline/work/monai_ssl_finetune/hard_case_review
```

You get:

- `hard_case_ranking.csv`
- `selected_cases.txt`
- `review_ct/` + `review_pred/`
- `corrected_labels/` (you fill this)

Correct selected hard cases in ilastik and save to `corrected_labels/<case_id>.nii.gz`.

## 5) Retrain with targeted corrections

```bash
bash pipeline/scripts/run_monai_supervised_cycle.sh \
  pipeline/work/monai/ct_nifti \
  path/to/ilastik_labels \
  pipeline/work/monai_ssl_plus_corrections \
  auto \
  pipeline/work/monai/models/ssl/ssl_best.pt \
  pipeline/work/monai_ssl_finetune/hard_case_review/corrected_labels
```

## 6) Core scripts

- `create_monai_splits.py`: builds case index + train/val/test split metadata
- `train_monai_ssl.py`: masked reconstruction pretraining on all CTs
- `train_monai_seg.py`: supervised liver segmentation, optionally SSL-initialized
- `predict_monai_seg.py`: inference for any split in split JSON
- `evaluate_segmentation.py`: Dice, IoU, precision, FP volume, volume error, HU difference
- `mine_hard_cases.py`: ranks likely failure cases for efficient correction

## 7) Expected metric outputs

`evaluate_segmentation.py` writes per-case CSV with:

- Dice
- IoU
- Precision
- False positive volume (ml)
- Predicted vs GT liver volume and volume error
- GT vs predicted mean liver HU and absolute HU difference

## 8) Keeping test set fixed

Use `create_monai_splits.py --fixed-test-cases path/to/test_case_ids.txt` once you decide your held-out set.
You can also set `--split-mode seeded_random` if you want the old random-with-seed behavior.

File format: one `case_id` per line.

## 9) Training/validation logs

`train_monai_seg.py` writes:

- `models/seg/seg_history.json`
- `models/seg/seg_history.csv`

Tracked per epoch:

- `train_loss`
- `val_loss` (on validation epochs)
- `val_dice`
- `val_iou`
- `val_precision`

`train_monai_seg.py` also writes TensorBoard logs under `<output_dir>/tensorboard`:

- `train/step_loss`
- `train/epoch_loss`
- `val/loss`
- `val/dice`
- `val/iou`
- `val/precision`

`train_monai_ssl.py` now also writes TensorBoard metrics:

- `train/step_ssl_loss`
- `train/step_mask_fraction`
- `train/epoch_ssl_loss`
- `train/epoch_mask_fraction`
- `val/epoch_ssl_loss` (when `splits.val` exists)
- `train/lr`
- `train/epoch_seconds`
- `train/steps_per_second`

SSL validation behavior:

- Training split key defaults to `splits.ssl_pretrain`
- Validation split key defaults to `splits.val`
- Overlapping case ids are removed from SSL training automatically to avoid leakage
- Configure with `--train-split-key`, `--val-split-key`, and `--val-interval`
