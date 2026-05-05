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
    return p.parse_args()


def main() -> None:
    args = parse_args()

    labeled_images = {}
    for p in sorted(args.labeled_animal_dir.glob("*.nii")) + sorted(args.labeled_animal_dir.glob("*.nii.gz")):
        labeled_images[normalize_case_id(nifti_stem(p))] = str(p.resolve())

    labeled_masks = {}
    for p in sorted(args.labeled_liver_dir.glob("*.nii")) + sorted(args.labeled_liver_dir.glob("*.nii.gz")):
        labeled_masks[parse_label_case_id(p)] = str(p.resolve())

    train_ids = sorted([cid for cid in labeled_images if cid in labeled_masks])
    if len(train_ids) != 3:
        raise RuntimeError(f"Expected 3 labeled train cases, found {len(train_ids)}: {train_ids}")

    all_ct_cases = {}
    for p in sorted(args.all_ct_dir.rglob("*.nii")) + sorted(args.all_ct_dir.rglob("*.nii.gz")):
        all_ct_cases[normalize_case_id(nifti_stem(p))] = str(p.resolve())

    for cid in train_ids:
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
        "description": "Pig liver binary segmentation split. Supervised train uses 3 labeled female cases.",
        "splits": {
            "ssl_pretrain": sorted(all_ct_cases.keys()),
            "labeled": train_ids,
            "unlabeled_only": sorted([cid for cid in all_ct_cases if cid not in set(train_ids)]),
            "train": train_ids,
            "val": [],
            "test": [],
            "val_unlabeled": val_unlabeled,
            "test_unlabeled": test_unlabeled,
        },
        "cases": case_rows,
        "counts": {
            "all_cases": len(all_ct_cases),
            "labeled_cases": len(train_ids),
            "train": len(train_ids),
            "val": 0,
            "test": 0,
            "val_unlabeled": len(val_unlabeled),
            "test_unlabeled": len(test_unlabeled),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote split file: {args.output_json}")
    print(f"Train labeled: {train_ids}")
    print(f"Val unlabeled: {val_unlabeled}")
    print(f"Test unlabeled: {test_unlabeled}")


if __name__ == "__main__":
    main()
