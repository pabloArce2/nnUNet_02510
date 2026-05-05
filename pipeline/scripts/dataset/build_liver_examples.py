#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Create a minimal liver segmentation example dataset from CT and liver masks. "
            "Masks are binarized to 0=background, 1=liver."
        )
    )
    p.add_argument("--ct-nifti-dir", type=Path, required=True, help="Folder with *_0000.nii.gz CT files")
    p.add_argument("--liver-mask-dir", type=Path, required=True, help="Folder with <case_id>.nii.gz liver masks")
    p.add_argument("--output-dir", type=Path, required=True, help="Destination directory")
    p.add_argument("--overwrite", action="store_true", help="Recreate output directory if it exists")
    return p.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Use --overwrite to recreate it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=False)


def case_id_from_ct_name(name: str) -> str:
    return name[: -len("_0000.nii.gz")]


def case_id_from_mask_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def main() -> None:
    args = parse_args()

    images_dir = args.output_dir / "images"
    labels_dir = args.output_dir / "labels"

    ensure_output_dir(args.output_dir, args.overwrite)
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir.mkdir(parents=True, exist_ok=False)

    ct_map: Dict[str, Path] = {
        case_id_from_ct_name(p.name): p
        for p in sorted(args.ct_nifti_dir.glob("*_0000.nii.gz"))
    }
    mask_map: Dict[str, Path] = {
        case_id_from_mask_name(p.name): p
        for p in sorted(args.liver_mask_dir.glob("*.nii.gz"))
    }

    common_case_ids = sorted(set(ct_map.keys()) & set(mask_map.keys()))
    if not common_case_ids:
        raise RuntimeError("No matching case IDs between CT and liver masks")

    rows: List[Dict[str, str]] = []

    for case_id in common_case_ids:
        ct_src = ct_map[case_id]
        mask_src = mask_map[case_id]

        ct_dst = images_dir / f"{case_id}_0000.nii.gz"
        mask_dst = labels_dir / f"{case_id}.nii.gz"

        shutil.copy2(ct_src, ct_dst)

        mask_img = sitk.ReadImage(str(mask_src))
        mask_bin = sitk.Cast(mask_img > 0, sitk.sitkUInt8)
        mask_bin.CopyInformation(mask_img)
        sitk.WriteImage(mask_bin, str(mask_dst), useCompression=True)

        rows.append(
            {
                "case_id": case_id,
                "ct_image": str(ct_dst.resolve()),
                "liver_mask": str(mask_dst.resolve()),
            }
        )
        print(f"Prepared example: {case_id}")

    missing_masks = sorted(set(ct_map.keys()) - set(mask_map.keys()))
    if missing_masks:
        print("Warning: missing liver masks for:", ", ".join(missing_masks))

    manifest_path = args.output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "ct_image", "liver_mask"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} examples to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
