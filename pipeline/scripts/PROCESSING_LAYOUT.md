# Script Layout

Scripts are organized by purpose and file type.

## `h5/`
- `argmax_probs_to_nifti.py`: H5 probabilities -> argmax mask NIfTI (debug)
- `postprocess_probs_argmax.py`: refined mask from H5 probabilities
- `preprocess_labeled_folder.py`: bulk preprocessing for labeled H5 folder

## `npy/`
- `argmax_probs_to_nifti.py`: NPY probabilities -> argmax mask NIfTI (debug)
- `postprocess_probs_argmax.py`: refined mask from NPY probabilities
- `prepare_nifti_from_npy.py`: `.npy` CT -> `*_0000.nii.gz`

## `monai/`
- `create_monai_splits.py`
- `train_monai_ssl.py`
- `train_monai_seg.py`
- `predict_monai_seg.py`
- `evaluate_ssl_reconstruction.py`
- `monai_common.py`

## `dataset/`
- `prepare_nifti_from_nii.py`
- `build_liver_examples.py`
- `build_nnunet_liver_dataset.py`

## `analysis/`
- `evaluate_segmentation.py`
- `mine_hard_cases.py`
- `select_cases_for_manual_review.py`

## `export/`
- `export_slicer_mrb_batch.py`

## `run/`
- `run_all.sh`
- `run_liver_stage1_only.sh`
- `run_mine_hard_cases.sh`
- `run_monai_ssl_pretrain.sh`
- `run_monai_supervised_cycle.sh`
- `run_totalseg_liver_batch.sh`
- `run_totalseg_on_dataset.sh`
- `export_slicer_mrb.sh`
