# Pig Liver Segmentation Pipeline (MONAI + SSL)

This repository now uses a MONAI 3D U-Net workflow with self-supervised pretraining and later supervised fine-tuning.

Runbook:
- [Pipeline runbook](pipeline/README.md)

## Current status fit (no labels yet)

You can already run the SSL stage on all unlabeled pig CT scans.

```bash
pip install -r pipeline/requirements.txt
bash pipeline/scripts/run_monai_ssl_pretrain.sh dataset/pig_nii_unlabeled
```

This produces:

- `pipeline/work/monai/ct_nifti/` (standardized `*_0000.nii.gz`)
- `pipeline/work/monai/splits.json` (stable train/val/test metadata; supervised splits empty until labels exist)
- `pipeline/work/monai/models/ssl/ssl_best.pt`

SSL training also writes TensorBoard logs at:

- `pipeline/work/monai/models/ssl/tensorboard`

View them with:

```bash
tensorboard --logdir pipeline/work/monai/models/ssl/tensorboard --port 6006
```

## When labels are available

Use ilastik masks (`<case_id>.nii.gz`) and run:

```bash
bash pipeline/scripts/run_monai_supervised_cycle.sh \
  pipeline/work/monai/ct_nifti \
  path/to/ilastik_labels \
  pipeline/work/monai_supervised \
  auto \
  pipeline/work/monai/models/ssl/ssl_best.pt
```

Then mine hard cases:

```bash
bash pipeline/scripts/run_mine_hard_cases.sh \
  pipeline/work/monai/ct_nifti \
  pipeline/work/monai_supervised/predictions/test \
  pipeline/work/monai_supervised/hard_case_review
```
