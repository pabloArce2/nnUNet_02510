#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def repo_root() -> Path:
    expected_root = Path("/home/arian-sumak/code/DTU/ADLCV-Visual-Debugger").resolve()

    if (expected_root / "dataset").is_dir() and (expected_root / "pipeline").is_dir():
        return expected_root

    for parent in Path(__file__).resolve().parents:
        if (parent / "dataset").is_dir() and (parent / "pipeline").is_dir():
            return parent

    return expected_root


REPO_ROOT = repo_root()


def resolve_repo_path(path: Path | str) -> Path:
    candidate = Path(path)

    if candidate.exists():
        return candidate

    if candidate.is_absolute():
        parts = candidate.parts
        if REPO_ROOT.name in parts:
            repo_index = len(parts) - 1 - list(reversed(parts)).index(REPO_ROOT.name)
            return REPO_ROOT.joinpath(*parts[repo_index + 1:])
        return candidate

    rebased = REPO_ROOT / candidate
    if rebased.exists():
        return rebased

    return candidate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute liver voxel and physical volume from label NIfTI files listed in a split JSON."
    )
    p.add_argument(
        "--split-json",
        type=Path,
        default=REPO_ROOT / "dataset" / "labeled" / "splits_pig_binary.json",
        help="Path to split JSON containing a 'cases' array with 'label' paths.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path for computed per-case metrics.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for computed per-case metrics.",
    )
    return p.parse_args()


def compute_case_metrics(case_id: str, label_path: Path) -> dict:
    nii = nib.load(str(label_path))
    mask = np.asanyarray(nii.dataobj) > 0
    sx, sy, sz = [float(v) for v in nii.header.get_zooms()[:3]]

    voxel_volume_mm3 = sx * sy * sz
    liver_voxels = int(mask.sum())
    liver_volume_mm3 = liver_voxels * voxel_volume_mm3
    liver_volume_cm3 = liver_volume_mm3 / 1000.0

    return {
        "case_id": case_id,
        "label_path": str(label_path.resolve()),
        "spacing_mm": [sx, sy, sz],
        "voxel_volume_mm3": voxel_volume_mm3,
        "liver_voxels": liver_voxels,
        "liver_volume_mm3": liver_volume_mm3,
        "liver_volume_cm3": liver_volume_cm3,
    }


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "case_id",
        "label_path",
        "spacing_x_mm",
        "spacing_y_mm",
        "spacing_z_mm",
        "voxel_volume_mm3",
        "liver_voxels",
        "liver_volume_mm3",
        "liver_volume_cm3",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in rows:
            writer.writerow(
                {
                    "case_id": r["case_id"],
                    "label_path": r["label_path"],
                    "spacing_x_mm": r["spacing_mm"][0],
                    "spacing_y_mm": r["spacing_mm"][1],
                    "spacing_z_mm": r["spacing_mm"][2],
                    "voxel_volume_mm3": r["voxel_volume_mm3"],
                    "liver_voxels": r["liver_voxels"],
                    "liver_volume_mm3": r["liver_volume_mm3"],
                    "liver_volume_cm3": r["liver_volume_cm3"],
                }
            )


def main() -> None:
    args = parse_args()

    split_json = resolve_repo_path(args.split_json)
    split_payload = json.loads(split_json.read_text())

    rows = []

    for case in split_payload.get("cases", []):
        case_id = case["case_id"]
        label_str = case.get("label")

        if not label_str:
            continue

        label_path = resolve_repo_path(label_str)

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for case {case_id}: {label_path}")

        rows.append(compute_case_metrics(case_id, label_path))

    if not rows:
        raise RuntimeError(
            "No labeled cases found in split JSON ('label' missing or null for all cases)."
        )

    print("case_id,spacing_mm,voxel_volume_mm3,liver_voxels,liver_volume_mm3,liver_volume_cm3")

    for r in rows:
        print(
            f"{r['case_id']},{tuple(r['spacing_mm'])},{r['voxel_volume_mm3']},"
            f"{r['liver_voxels']},{r['liver_volume_mm3']},{r['liver_volume_cm3']}"
        )

    if args.output_json is not None:
        output_json = resolve_repo_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps({"cases": rows}, indent=2) + "\n")
        print(f"Wrote JSON: {output_json}")

    if args.output_csv is not None:
        output_csv = resolve_repo_path(args.output_csv)
        write_csv(rows, output_csv)
        print(f"Wrote CSV: {output_csv}")


if __name__ == "__main__":
    main()