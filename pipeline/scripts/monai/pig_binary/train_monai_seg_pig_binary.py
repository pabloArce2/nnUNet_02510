#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Pig binary liver segmentation training wrapper. "
            "Preprocesses all CT examples to canonical orientation, then runs train_monai_seg.py "
            "with existing label-alignment + foreground preflight checks."
        )
    )
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--preprocessed-split-json", type=Path, default=None)
    p.add_argument("--skip-image-preprocess", action="store_true")
    p.add_argument(
        "--run-mode",
        type=str,
        choices=["normal", "loocv"],
        default="normal",
        help="normal: single training run using splits.train/splits.val; loocv: one run per labeled hold-out fold.",
    )
    return p.parse_args()


def preprocess_all_images(split_payload: dict, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = {}
    for row in split_payload.get("cases", []):
        cid = row["case_id"]
        img_path = Path(row["image"])
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image for case {cid}: {img_path}")
        nii = nib.load(str(img_path))
        canonical = nib.as_closest_canonical(nii)
        dst = out_dir / f"{cid}.nii.gz"
        nib.save(canonical, str(dst))
        updated[cid] = str(dst.resolve())
    return updated


def preprocess_binary_labels(split_payload: dict, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    updated = {}
    for row in split_payload.get("cases", []):
        cid = row["case_id"]
        label_path = row.get("label")
        if not label_path:
            continue
        src = Path(label_path)
        if not src.exists():
            raise FileNotFoundError(f"Missing label for case {cid}: {src}")
        nii = nib.load(str(src))
        arr = np.asanyarray(nii.dataobj)
        binary = (arr > 0).astype(np.uint8)
        dst = out_dir / f"{cid}.nii.gz"
        nib.save(nib.Nifti1Image(binary, nii.affine, nii.header), str(dst))
        updated[cid] = str(dst.resolve())
    return updated


def sanitize_case_id(cid: str) -> str:
    return cid.replace("/", "_").replace(" ", "_")


def build_loocv_splits(split_payload: dict) -> list[tuple[str, dict]]:
    labeled_ids = list(split_payload.get("splits", {}).get("labeled", []))
    if len(labeled_ids) < 2:
        raise RuntimeError(
            f"LOOCV requires at least 2 labeled cases, found {len(labeled_ids)} in splits.labeled"
        )

    folds: list[tuple[str, dict]] = []
    for val_case in labeled_ids:
        fold_payload = copy.deepcopy(split_payload)
        train_ids = [cid for cid in labeled_ids if cid != val_case]
        fold_payload["splits"]["train"] = train_ids
        fold_payload["splits"]["val"] = [val_case]
        if "counts" in fold_payload and isinstance(fold_payload["counts"], dict):
            fold_payload["counts"]["train"] = len(train_ids)
            fold_payload["counts"]["val"] = 1
        folds.append((val_case, fold_payload))
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preprocessed-split-json", type=Path, default=None)
    parser.add_argument("--skip-image-preprocess", action="store_true")
    parser.add_argument("--run-mode", type=str, choices=["normal", "loocv"], default="normal")
    args, passthrough = parser.parse_known_args()

    split_payload = json.loads(args.split_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_split = args.preprocessed_split_json or (args.output_dir / "split_preprocessed.json")

    if args.skip_image_preprocess:
        work_payload = split_payload
    else:
        mapped = preprocess_all_images(split_payload, args.output_dir / "preprocessed_images")
        work_payload = dict(split_payload)
        work_payload["cases"] = [dict(r) for r in split_payload.get("cases", [])]
        for row in work_payload["cases"]:
            row["image"] = mapped[row["case_id"]]
    label_mapped = preprocess_binary_labels(work_payload, args.output_dir / "preprocessed_labels_binary")
    for row in work_payload.get("cases", []):
        cid = row["case_id"]
        if cid in label_mapped:
            row["label"] = label_mapped[cid]
            row["label_source"] = "binary_liver_mask"

    base_script = Path(__file__).resolve().parent.parent / "train_monai_seg.py"
    if args.run_mode == "normal":
        preprocessed_split.write_text(json.dumps(work_payload, indent=2) + "\n")
        print(f"Prepared split for training: {preprocessed_split}")
        cmd = [
            sys.executable,
            str(base_script),
            "--split-json",
            str(preprocessed_split),
            "--output-dir",
            str(args.output_dir),
        ]
        cmd.extend(passthrough)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return

    folds = build_loocv_splits(work_payload)
    loocv_split_dir = args.output_dir / "loocv_splits"
    loocv_split_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prepared {len(folds)} LOOCV folds from splits.labeled")
    for fold_idx, (val_case, fold_payload) in enumerate(folds, start=1):
        safe_case = sanitize_case_id(val_case)
        fold_name = f"fold{fold_idx:02d}_{safe_case}"
        fold_split = loocv_split_dir / f"{fold_name}.json"
        fold_output_dir = args.output_dir / fold_name
        fold_split.write_text(json.dumps(fold_payload, indent=2) + "\n")
        print(
            f"Fold {fold_idx}/{len(folds)}: val={val_case}, "
            f"train={fold_payload['splits']['train']} -> split={fold_split}"
        )
        cmd = [
            sys.executable,
            str(base_script),
            "--split-json",
            str(fold_split),
            "--output-dir",
            str(fold_output_dir),
        ]
        cmd.extend(passthrough)
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
