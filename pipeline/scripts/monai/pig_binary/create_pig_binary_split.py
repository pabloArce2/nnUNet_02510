#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def normalize_case_id(stem: str) -> str:
    case = stem.strip().replace(" ", "_")
    case = re.sub(r"_+", "_", case)
    return case


def parse_label_case_id(path: Path) -> str:
    stem = nifti_stem(path)
    if stem.endswith("_liver"):
        stem = stem[: -len("_liver")]
    return normalize_case_id(stem)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create pig binary split JSON from labeled + unlabeled NIfTI folders")
    p.add_argument(
        "--labeled-animal-dir",
        type=Path,
        default=Path("/home/arian-sumak/code/DTU/nnUNet_02510/dataset/labeled/animal"),
    )
    p.add_argument(
        "--labeled-liver-dir",
        type=Path,
        default=Path("/home/arian-sumak/code/DTU/nnUNet_02510/dataset/labeled/liver"),
    )
    p.add_argument(
        "--all-ct-dir",
        type=Path,
        default=Path("/home/arian-sumak/code/DTU/nnUNet_02510/dataset/pig_nii_unlabeled"),
    )
    p.add_argument(
        "--val-female-case",
        type=str,
        default="Animal_223883_female_12months",
        help="Unlabeled female case id for validation split",
    )
    p.add_argument(
        "--test-female-case",
        type=str,
        default="Animal_323426_female_26months",
        help="Unlabeled female case id for test split",
    )
    p.add_argument(
        "--test-male-case",
        type=str,
        default="Animal_223944_male_12months",
        help="Unlabeled male case id for test split",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=Path("/home/arian-sumak/code/DTU/nnUNet_02510/dataset/labeled/splits_pig_binary.json"),
    )
    p.add_argument(
        "--labeled-val-case",
        type=str,
        default=None,
        help="Optional labeled case id to place in labeled validation split",
    )
    p.add_argument(
        "--labeled-val-count",
        type=int,
        default=1,
        help="Number of labeled cases held out for validation when --labeled-val-case is not set",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labeled_images = {}
    for p in sorted(args.labeled_animal_dir.glob("*.nii")) + sorted(args.labeled_animal_dir.glob("*.nii.gz")):
        labeled_images[normalize_case_id(nifti_stem(p))] = str(p.resolve())

    labeled_masks = {}
    for p in sorted(args.labeled_liver_dir.glob("*.nii")) + sorted(args.labeled_liver_dir.glob("*.nii.gz")):
        labeled_masks[parse_label_case_id(p)] = str(p.resolve())

    missing_masks = sorted([cid for cid in labeled_images if cid not in labeled_masks])
    missing_images = sorted([cid for cid in labeled_masks if cid not in labeled_images])
    if missing_masks or missing_images:
        raise RuntimeError(
            "Labeled animal/liver files must be fully paired by case_id.\n"
            f"Missing liver masks for: {missing_masks}\n"
            f"Missing animal images for: {missing_images}"
        )

    labeled_ids = sorted(labeled_images.keys())
    if not labeled_ids:
        raise RuntimeError(
            f"No labeled pairs found in {args.labeled_animal_dir} and {args.labeled_liver_dir}"
        )
    if args.labeled_val_count < 0:
        raise RuntimeError(f"--labeled-val-count must be >= 0, got {args.labeled_val_count}")
    if args.labeled_val_count >= len(labeled_ids):
        raise RuntimeError(
            f"--labeled-val-count ({args.labeled_val_count}) must be smaller than "
            f"number of labeled cases ({len(labeled_ids)})"
        )

    if args.labeled_val_case is not None:
        val_labeled = normalize_case_id(args.labeled_val_case)
        if val_labeled not in labeled_ids:
            raise RuntimeError(
                f"--labeled-val-case {val_labeled} not found in labeled pairs: {labeled_ids}"
            )
        val_ids = [val_labeled]
    else:
        val_ids = labeled_ids[-args.labeled_val_count :] if args.labeled_val_count > 0 else []
    train_ids = [cid for cid in labeled_ids if cid not in set(val_ids)]

    all_ct_cases = {}
    for p in sorted(args.all_ct_dir.rglob("*.nii")) + sorted(args.all_ct_dir.rglob("*.nii.gz")):
        all_ct_cases[normalize_case_id(nifti_stem(p))] = str(p.resolve())

    for cid in labeled_ids:
        all_ct_cases[cid] = labeled_images[cid]

    val_case = normalize_case_id(args.val_female_case)
    test_f = normalize_case_id(args.test_female_case)
    test_m = normalize_case_id(args.test_male_case)

    required = [val_case, test_f, test_m]
    missing = [cid for cid in required if cid not in all_ct_cases]
    if missing:
        raise RuntimeError(f"Requested val/test cases not found in CT pool: {missing}")

    val_unlabeled = [val_case]
    test_unlabeled = [test_f, test_m]

    case_rows = []
    for cid in sorted(all_ct_cases.keys()):
        case_rows.append(
            {
                "case_id": cid,
                "image": all_ct_cases[cid],
                "label": labeled_masks.get(cid),
                "label_source": "manual_liver" if cid in labeled_masks else None,
            }
        )

    payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "description": "Pig liver binary segmentation split. Supervised train uses all labeled animal/liver pairs.",
        "splits": {
            "ssl_pretrain": sorted(all_ct_cases.keys()),
            "labeled": labeled_ids,
            "unlabeled_only": sorted([cid for cid in all_ct_cases if cid not in set(labeled_ids)]),
            "train": train_ids,
            "val": val_ids,
            "test": [],
            "val_unlabeled": val_unlabeled,
            "test_unlabeled": test_unlabeled,
        },
        "cases": case_rows,
        "counts": {
            "all_cases": len(all_ct_cases),
            "labeled_cases": len(labeled_ids),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": 0,
            "val_unlabeled": len(val_unlabeled),
            "test_unlabeled": len(test_unlabeled),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote split file: {args.output_json}")
    print(f"Labeled total: {labeled_ids}")
    print(f"Train labeled: {train_ids}")
    print(f"Val labeled: {val_ids}")
    print(f"Val unlabeled: {val_unlabeled}")
    print(f"Test unlabeled: {test_unlabeled}")


if __name__ == "__main__":
    main()
