#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate liver segmentation predictions")
    p.add_argument("--ct-dir", type=Path, required=True, help="Directory with *_0000.nii.gz CT volumes")
    p.add_argument("--pred-dir", type=Path, required=True, help="Directory with prediction masks <case_id>.nii.gz")
    p.add_argument("--label-dir", type=Path, required=True, help="Directory with ground truth masks <case_id>.nii.gz")
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def case_id_from_ct(path: Path) -> str:
    return path.name[: -len("_0000.nii.gz")]


def voxel_volume_ml(header) -> float:
    zoom = header.get_zooms()[:3]
    mm3 = float(zoom[0] * zoom[1] * zoom[2])
    return mm3 / 1000.0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def main() -> None:
    args = parse_args()

    rows = []
    for ct_path in sorted(args.ct_dir.glob("*_0000.nii.gz")):
        case_id = case_id_from_ct(ct_path)
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        gt_path = args.label_dir / f"{case_id}.nii.gz"

        if not pred_path.exists() or not gt_path.exists():
            continue

        ct_nii = nib.load(str(ct_path))
        pred_nii = nib.load(str(pred_path))
        gt_nii = nib.load(str(gt_path))

        ct = np.asarray(ct_nii.get_fdata(), dtype=np.float32)
        pred = np.asarray(pred_nii.get_fdata() > 0.5, dtype=np.uint8)
        gt = np.asarray(gt_nii.get_fdata() > 0.5, dtype=np.uint8)

        if pred.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {case_id}: pred={pred.shape} gt={gt.shape}")
        if ct.shape != gt.shape:
            raise ValueError(f"CT/label shape mismatch for {case_id}: ct={ct.shape} gt={gt.shape}")

        tp = int(np.logical_and(pred == 1, gt == 1).sum())
        fp = int(np.logical_and(pred == 1, gt == 0).sum())
        fn = int(np.logical_and(pred == 0, gt == 1).sum())

        dice = safe_div(2 * tp, 2 * tp + fp + fn)
        iou = safe_div(tp, tp + fp + fn)
        precision = safe_div(tp, tp + fp)

        vox_ml = voxel_volume_ml(gt_nii.header)
        pred_vol = float(pred.sum()) * vox_ml
        gt_vol = float(gt.sum()) * vox_ml
        fp_vol = float(fp) * vox_ml
        vol_abs_err = abs(pred_vol - gt_vol)
        vol_rel_err = safe_div(vol_abs_err, gt_vol)

        gt_mask = gt > 0
        pred_mask = pred > 0
        gt_hu = float(ct[gt_mask].mean()) if gt_mask.any() else 0.0
        pred_hu = float(ct[pred_mask].mean()) if pred_mask.any() else 0.0
        hu_diff = abs(pred_hu - gt_hu)

        rows.append(
            {
                "case_id": case_id,
                "dice": f"{dice:.6f}",
                "iou": f"{iou:.6f}",
                "precision": f"{precision:.6f}",
                "fp_volume_ml": f"{fp_vol:.3f}",
                "pred_volume_ml": f"{pred_vol:.3f}",
                "gt_volume_ml": f"{gt_vol:.3f}",
                "volume_abs_error_ml": f"{vol_abs_err:.3f}",
                "volume_rel_error": f"{vol_rel_err:.6f}",
                "gt_mean_hu": f"{gt_hu:.3f}",
                "pred_mean_hu": f"{pred_hu:.3f}",
                "mean_hu_abs_diff": f"{hu_diff:.3f}",
            }
        )

    if not rows:
        raise RuntimeError("No overlapping CT/pred/label cases found")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    means = {}
    numeric_keys = [
        "dice",
        "iou",
        "precision",
        "fp_volume_ml",
        "volume_abs_error_ml",
        "volume_rel_error",
        "mean_hu_abs_diff",
    ]
    for key in numeric_keys:
        means[key] = float(np.mean([float(r[key]) for r in rows]))

    print(f"Saved per-case metrics: {args.output_csv}")
    print("Mean metrics:")
    for k in numeric_keys:
        print(f"  {k}: {means[k]:.6f}")


if __name__ == "__main__":
    main()
