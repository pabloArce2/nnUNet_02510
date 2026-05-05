#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build nnU-Net dataset from CT + pseudo/corrected liver masks")
    p.add_argument("--ct-nifti-dir", type=Path, required=True, help="Folder with *_0000.nii.gz CT files")
    p.add_argument("--pseudo-liver-dir", type=Path, required=True, help="Folder with pseudo liver masks: <case>.nii.gz")
    p.add_argument("--corrected-liver-dir", type=Path, default=None, help="Optional corrected masks: <case>.nii.gz")
    p.add_argument("--nnunet-raw", type=Path, required=True, help="Path to nnUNet_raw")
    p.add_argument("--dataset-id", type=int, default=301, help="nnU-Net dataset id (default 301)")
    p.add_argument("--dataset-name", type=str, default="LiverPseudoAnimal", help="Dataset name suffix")
    p.add_argument(
        "--test-cases",
        type=Path,
        default=None,
        help="Optional text file with held-out case ids (one per line)",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Used only if --test-cases is not provided",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split size within training pool for splits_final.json",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--corrected-dup-factor",
        type=int,
        default=1,
        help=(
            "Duplicate corrected cases this many times in training set to up-weight manual labels. "
            "1 means no duplication."
        ),
    )
    p.add_argument("--overwrite", action="store_true", help="Delete existing dataset folder before writing")
    return p.parse_args()


def case_id_from_mask_filename(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def load_case_ids_from_ct_dir(ct_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for p in sorted(ct_dir.glob("*_0000.nii.gz")):
        case_id = p.name[: -len("_0000.nii.gz")]
        mapping[case_id] = p
    return mapping


def read_case_list(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text().splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            out.add(item)
    return out


def ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def split_train_val(cases: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    case_list = list(cases)
    if not case_list:
        return [], []
    rnd = random.Random(seed)
    rnd.shuffle(case_list)

    if len(case_list) == 1:
        return case_list, []

    val_n = max(1, int(round(len(case_list) * val_fraction)))
    val_n = min(val_n, len(case_list) - 1)
    val = sorted(case_list[:val_n])
    train = sorted(case_list[val_n:])
    return train, val


def write_dataset_json(dataset_dir: Path, dataset_name: str, num_training_cases: int) -> None:
    payload = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1},
        "numTraining": int(num_training_cases),
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "description": "Liver segmentation with pseudo-labels from TotalSegmentator and optional manual corrections",
        "reference": "Generated locally",
        "licence": "unknown",
        "release": "0.0",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()

    if args.corrected_dup_factor < 1:
        raise ValueError("--corrected-dup-factor must be >= 1")

    ct_cases = load_case_ids_from_ct_dir(args.ct_nifti_dir)
    if not ct_cases:
        raise FileNotFoundError(f"No *_0000.nii.gz files found in {args.ct_nifti_dir}")

    pseudo_cases = {case_id_from_mask_filename(p.name): p for p in args.pseudo_liver_dir.glob("*.nii.gz")}
    corrected_cases = {}
    if args.corrected_liver_dir is not None and args.corrected_liver_dir.exists():
        corrected_cases = {case_id_from_mask_filename(p.name): p for p in args.corrected_liver_dir.glob("*.nii.gz")}

    available = sorted([cid for cid in ct_cases if cid in pseudo_cases])
    missing_pseudo = sorted([cid for cid in ct_cases if cid not in pseudo_cases])
    if missing_pseudo:
        print("Warning: missing pseudo masks for:", ", ".join(missing_pseudo))

    if not available:
        raise RuntimeError("No cases have both CT and pseudo liver mask")

    rnd = random.Random(args.seed)
    if args.test_cases is not None:
        test_cases = read_case_list(args.test_cases)
        unknown = sorted(test_cases - set(available))
        if unknown:
            raise ValueError(f"Test list contains unknown/unavailable case ids: {unknown}")
    else:
        n_test = max(1, int(round(len(available) * args.test_fraction)))
        n_test = min(n_test, len(available) - 1) if len(available) > 1 else 0
        shuffled = available[:]
        rnd.shuffle(shuffled)
        test_cases = set(shuffled[:n_test])

    train_pool = sorted([c for c in available if c not in test_cases])
    if not train_pool:
        raise RuntimeError("No training cases left after test split")

    train_cases, val_cases = split_train_val(train_pool, args.val_fraction, args.seed)
    corrected_in_train = sorted([c for c in train_pool if c in corrected_cases])

    dataset_folder_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = args.nnunet_raw / dataset_folder_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    ensure_empty_dir(dataset_dir, overwrite=args.overwrite)
    images_tr.mkdir(exist_ok=True)
    labels_tr.mkdir(exist_ok=True)
    images_ts.mkdir(exist_ok=True)
    labels_ts.mkdir(exist_ok=True)

    training_ids: List[str] = []
    train_split_ids: List[str] = []

    for case_id in train_pool:
        src_img = ct_cases[case_id]
        src_lbl = corrected_cases.get(case_id, pseudo_cases[case_id])

        dst_img = images_tr / f"{case_id}_0000.nii.gz"
        dst_lbl = labels_tr / f"{case_id}.nii.gz"

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)
        training_ids.append(case_id)
        if case_id in train_cases:
            train_split_ids.append(case_id)

        if case_id in corrected_cases and args.corrected_dup_factor > 1:
            for i in range(1, args.corrected_dup_factor):
                dup_id = f"{case_id}_corrdup{i:02d}"
                shutil.copy2(src_img, images_tr / f"{dup_id}_0000.nii.gz")
                shutil.copy2(src_lbl, labels_tr / f"{dup_id}.nii.gz")
                training_ids.append(dup_id)
                if case_id in train_cases:
                    train_split_ids.append(dup_id)

    for case_id in sorted(test_cases):
        shutil.copy2(ct_cases[case_id], images_ts / f"{case_id}_0000.nii.gz")
        src_lbl = corrected_cases.get(case_id, pseudo_cases[case_id])
        shutil.copy2(src_lbl, labels_ts / f"{case_id}.nii.gz")

    write_dataset_json(dataset_dir, args.dataset_name, len(training_ids))

    split_payload = [
        {
            "train": sorted(train_split_ids),
            "val": sorted(val_cases),
        }
    ]
    (dataset_dir / "splits_final.json").write_text(json.dumps(split_payload, indent=2) + "\n")

    manifest = {
        "dataset_folder": str(dataset_dir),
        "num_ct_total": len(ct_cases),
        "num_available_with_pseudo": len(available),
        "num_training_cases_effective": len(training_ids),
        "num_training_unique": len(train_pool),
        "num_validation_unique": len(val_cases),
        "num_test_unique": len(test_cases),
        "corrected_cases_available": sorted(corrected_cases.keys()),
        "corrected_cases_used_in_train": corrected_in_train,
        "train_unique": train_pool,
        "val_unique": val_cases,
        "test_unique": sorted(test_cases),
        "missing_pseudo": missing_pseudo,
    }
    (dataset_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Created {dataset_dir}")
    print(f"Training cases (effective): {len(training_ids)}")
    print(f"Train/val unique: {len(train_cases)}/{len(val_cases)}")
    print(f"Test unique: {len(test_cases)}")


if __name__ == "__main__":
    main()
