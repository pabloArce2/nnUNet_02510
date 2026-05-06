# Pig Liver Segmentation Pipeline (MONAI + SSL)

This repository contains a MONAI-based 3D liver segmentation pipeline for pig CT data, including:
- supervised binary liver segmentation
- optional SSL pretraining on unlabeled CT
- preprocessing, label/image alignment, and training diagnostics

Main runbook:
- [Pipeline runbook](pipeline/README.md)

Additional reference:
- [MONAI metrics + CLI reference](pipeline/docs/monai_metrics_cli_reference.md)

## Quick start

```bash
pip install -r pipeline/requirements.txt
```

Create pig-binary split JSON:

```bash
python pipeline/scripts/monai/pig_binary/create_pig_binary_split.py
```

Train pig-binary segmentation (wrapper does CT canonicalization + binary mask conversion):

```bash
python pipeline/scripts/monai/pig_binary/train_monai_seg_pig_binary.py \
  --split-json dataset/labeled/pig_binary/splits_pig_binary.json \
  --output-dir pipeline/runs/monai_pig_binary
```

Optional SSL pretraining:

```bash
python pipeline/scripts/monai/train_monai_ssl.py \
  --split-json dataset/labeled/pig_binary/splits_pig_binary.json \
  --output-dir pipeline/runs/monai_ssl
```
