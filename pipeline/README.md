# MONAI Pipeline (Pig Binary)

Current flow:
1. Build split metadata (`create_pig_binary_split.py`)
2. Wrapper preprocessing + train (`train_monai_seg_pig_binary.py`)
3. Optional SSL pretraining (`train_monai_ssl.py`) and SSL initialization during segmentation training

## 1) Install

Install PyTorch first (matching your CUDA/CPU setup), then:

```bash
pip install -r pipeline/requirements.txt
```

## 2) Create pig-binary split JSON

```bash
python pipeline/scripts/monai/pig_binary/create_pig_binary_split.py \
  --output-json dataset/labeled/pig_binary/splits_pig_binary.json
```

Output:
- `dataset/labeled/pig_binary/splits_pig_binary.json`

## 3) Train pig-binary segmentation (wrapper)

```bash
python pipeline/scripts/monai/pig_binary/train_monai_seg_pig_binary.py \
  --split-json dataset/labeled/pig_binary/splits_pig_binary.json \
  --output-dir pipeline/runs/monai_pig_binary
```

Wrapper behavior:
- rebases paths to current repo
- canonicalizes CT orientation
- converts labels to binary liver masks (`label > 0`)
- writes `split_preprocessed.json`
- calls `train_monai_seg.py` with passthrough args

## 4) Optional SSL pretraining

```bash
python pipeline/scripts/monai/train_monai_ssl.py \
  --split-json dataset/labeled/pig_binary/splits_pig_binary.json \
  --output-dir pipeline/runs/monai_ssl
```

Use the resulting checkpoint with:
- `train_monai_seg.py --init-ssl-checkpoint <path-to-ssl-checkpoint>`

## 5) Core scripts

- `pipeline/scripts/monai/pig_binary/create_pig_binary_split.py`
- `pipeline/scripts/monai/pig_binary/train_monai_seg_pig_binary.py`
- `pipeline/scripts/monai/train_monai_seg.py`
- `pipeline/scripts/monai/train_monai_ssl.py`
- `pipeline/scripts/monai/predict_monai_seg.py`

## 6) Segmentation metrics currently tracked

Validation (`val_*`):
- `val_loss`
- `val_dice`, `val_iou`, `val_precision`
- `val_recall`, `val_specificity`, `val_accuracy`
- `val_balanced_accuracy`, `val_dice_bg`, `val_macro_dice`
- `val_tp`, `val_fp`, `val_fn`, `val_tn`

Patch distribution diagnostics:
- `train_fg_fraction`
- `train_class_fraction_c*`

Optional train full-volume metrics (`train_vol_*`) when `--train-full-volume-interval > 0`.

Outputs:
- `seg_history.csv`
- `seg_history.json`
- TensorBoard logs under `<run_dir>/tensorboard`

## 7) CLI reference

See:
- `pipeline/docs/monai_metrics_cli_reference.md`
