# Labeled H5 Preprocessing Guide (ILASTIK -> MONAI SSL/SL)

This guide is for preparing your repository workflow end-to-end:

- SSL pretraining on CT volumes
- Supervised segmentation with labels converted from ILASTIK H5

No liver ROI cropping is used in preprocessing, so train/test stay consistent with full-volume inference.

## 1) Install dependencies

```bash
pip install -r pipeline/requirements.txt
pip install h5py scipy SimpleITK
```

## 2) Prepare CT NIfTI once (shared by SSL and supervised)

If your CT source is already NIfTI:

```bash
python pipeline/scripts/prepare_nifti_from_nii.py \
  --input-dir dataset/pig_nii_unlabeled \
  --output-dir pipeline/work/monai/ct_nifti
```

If your CT source is `.npy`:

```bash
python pipeline/scripts/prepare_nifti_from_npy.py \
  --input-dir dataset/pig_npy_unlabeled \
  --output-dir pipeline/work/monai/ct_nifti
```

## 3) Build split metadata

```bash
python pipeline/scripts/create_monai_splits.py \
  --ct-dir pipeline/work/monai/ct_nifti \
  --output-json pipeline/work/monai/splits.json
```

At this stage supervised `train/val/test` may be empty if no labels are attached yet, but SSL can run.

## 4) Run SSL pretraining

```bash
python pipeline/scripts/train_monai_ssl.py \
  --split-json pipeline/work/monai/splits.json \
  --output-dir pipeline/work/monai/models/ssl \
  --device auto
```

## 5) Debug one H5 sample before bulk preprocessing

```bash
python pipeline/scripts/convert_h5_probs_debug.py \
  --input dataset/pig_h5_labeled/<example>.h5 \
  --dataset-key /exported_data \
  --output-dir pipeline/work/h5_debug \
  --axis-order zyx \
  --class-axis -1 \
  --write-prob-channels
```

Use the generated `*_argmax_summary.json` and `*_prob_ch*.nii.gz` files to verify class mapping and axis convention before running folder preprocessing.

## 6) Preprocess labeled H5 folder to supervised labels

If H5 contains both image and label datasets:

```bash
python pipeline/scripts/preprocess_h5_labeled_folder.py \
  --input-dir dataset/pig_h5_labeled \
  --output-images-dir pipeline/work/h5_preprocessed/images \
  --output-labels-dir pipeline/work/h5_preprocessed/labels \
  --fill-holes \
  --axis-order zyx
```

If H5 is label-only, align labels to prepared CT NIfTI geometry:

```bash
python pipeline/scripts/preprocess_h5_labeled_folder.py \
  --input-dir dataset/pig_h5_labeled \
  --output-images-dir pipeline/work/h5_preprocessed/images_unused \
  --output-labels-dir pipeline/work/h5_preprocessed/labels \
  --label-only \
  --image-ref-dir pipeline/work/monai/ct_nifti \
  --fill-holes \
  --axis-order zyx
```

## 7) Rebuild splits with labels and run supervised segmentation

```bash
python pipeline/scripts/create_monai_splits.py \
  --ct-dir pipeline/work/monai/ct_nifti \
  --label-dir pipeline/work/h5_preprocessed/labels \
  --output-json pipeline/work/monai_supervised/splits.json
```

From scratch:

```bash
python pipeline/scripts/train_monai_seg.py \
  --split-json pipeline/work/monai_supervised/splits.json \
  --output-dir pipeline/work/monai_supervised/models/seg \
  --flat-output \
  --device auto
```

SSL-initialized:

```bash
python pipeline/scripts/train_monai_seg.py \
  --split-json pipeline/work/monai_supervised/splits.json \
  --output-dir pipeline/work/monai_supervised/models/seg_ssl_init \
  --flat-output \
  --init-ssl-checkpoint pipeline/work/monai/models/ssl/ssl_best.pt \
  --device auto
```

## 8) Predict + evaluate test split

```bash
python pipeline/scripts/predict_monai_seg.py \
  --split-json pipeline/work/monai_supervised/splits.json \
  --checkpoint pipeline/work/monai_supervised/models/seg_ssl_init/seg_best.pt \
  --output-dir pipeline/work/monai_supervised/predictions/test \
  --split-name test \
  --save-probability \
  --device auto
```

```bash
python pipeline/scripts/evaluate_segmentation.py \
  --ct-dir pipeline/work/monai/ct_nifti \
  --pred-dir pipeline/work/monai_supervised/predictions/test/pred_masks \
  --label-dir pipeline/work/h5_preprocessed/labels \
  --output-csv pipeline/work/monai_supervised/metrics/test_metrics.csv
```
