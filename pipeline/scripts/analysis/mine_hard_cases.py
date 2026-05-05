#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank likely hard cases from predicted masks")
    p.add_argument("--ct-dir", type=Path, required=True, help="Directory with *_0000.nii.gz CT volumes")
    p.add_argument("--pred-dir", type=Path, required=True, help="Directory with predicted masks <case_id>.nii.gz")
    p.add_argument(
        "--prob-dir",
        type=Path,
        default=None,
        help="Optional directory with predicted probability maps <case_id>.nii.gz",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def case_id_from_ct(path: Path) -> str:
    return path.name[: -len("_0000.nii.gz")]


def voxel_volume_ml(header) -> float:
    zoom = header.get_zooms()[:3]
    mm3 = float(zoom[0] * zoom[1] * zoom[2])
    return mm3 / 1000.0


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ct_path in sorted(args.ct_dir.glob("*_0000.nii.gz")):
        case_id = case_id_from_ct(ct_path)
        pred_path = args.pred_dir / f"{case_id}.nii.gz"
        if not pred_path.exists():
            continue

        pred_nii = nib.load(str(pred_path))
        pred = np.asarray(pred_nii.get_fdata() > 0.5, dtype=np.uint8)

        vox_ml = voxel_volume_ml(pred_nii.header)
        volume_ml = float(pred.sum()) * vox_ml
        labeled, ncomp = cc_label(pred)

        largest_comp = 0
        if ncomp > 0:
            counts = np.bincount(labeled.reshape(-1))
            largest_comp = int(counts[1:].max()) if len(counts) > 1 else 0
        disconnected_ratio = 0.0 if pred.sum() == 0 else 1.0 - (largest_comp / float(pred.sum()))

        confidence = None
        if args.prob_dir is not None:
            prob_path = args.prob_dir / f"{case_id}.nii.gz"
            if prob_path.exists():
                prob = np.asarray(nib.load(str(prob_path)).get_fdata(), dtype=np.float32)
                if pred.sum() > 0:
                    confidence = float(prob[pred > 0].mean())
                else:
                    confidence = float(prob.mean())

        score = disconnected_ratio * 2.0
        if volume_ml < 150 or volume_ml > 2200:
            score += 1.0
        if confidence is not None:
            score += max(0.0, 0.8 - confidence)

        rows.append(
            {
                "case_id": case_id,
                "volume_ml": f"{volume_ml:.3f}",
                "connected_components": str(int(ncomp)),
                "disconnected_ratio": f"{disconnected_ratio:.5f}",
                "mean_confidence": "" if confidence is None else f"{confidence:.5f}",
                "hardness_score": f"{score:.5f}",
                "ct_path": str(ct_path.resolve()),
                "pred_path": str(pred_path.resolve()),
            }
        )

    if not rows:
        raise RuntimeError("No overlapping CT and prediction files found")

    rows_sorted = sorted(rows, key=lambda r: float(r["hardness_score"]), reverse=True)

    csv_path = args.output_dir / "hard_case_ranking.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    selected = rows_sorted[: max(1, min(args.top_k, len(rows_sorted)))]
    review_ct = args.output_dir / "review_ct"
    review_pred = args.output_dir / "review_pred"
    review_corrected = args.output_dir / "corrected_labels"
    review_ct.mkdir(exist_ok=True)
    review_pred.mkdir(exist_ok=True)
    review_corrected.mkdir(exist_ok=True)

    selected_ids = []
    for row in selected:
        case_id = row["case_id"]
        selected_ids.append(case_id)
        shutil.copy2(row["ct_path"], review_ct / Path(row["ct_path"]).name)
        shutil.copy2(row["pred_path"], review_pred / f"{case_id}.nii.gz")

    (args.output_dir / "selected_cases.txt").write_text("\n".join(selected_ids) + "\n")
    (args.output_dir / "README.txt").write_text(
        "Hard-case review folder\n"
        "- Open review_ct + review_pred in ilastik/Slicer.\n"
        "- Save corrected liver labels to corrected_labels/<case_id>.nii.gz.\n"
        "- Rebuild splits using --corrected-label-dir and retrain segmentation.\n"
    )

    print(f"Saved ranking CSV: {csv_path}")
    print(f"Selected {len(selected_ids)} hard cases for correction in: {args.output_dir}")


if __name__ == "__main__":
    main()
