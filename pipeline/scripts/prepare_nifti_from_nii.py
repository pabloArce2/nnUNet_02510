#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List

import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare NIfTI CT files for TotalSegmentator by renaming to *_0000.nii.gz"
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Folder with .nii or .nii.gz files")
    p.add_argument("--output-dir", type=Path, required=True, help="Output folder for *_0000.nii.gz files")
    p.add_argument("--prefix", type=str, default="ct", help="Case id prefix")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV path to save source/target mapping",
    )
    return p.parse_args()


def sanitize_case_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return clean.lower()


def unique_case_ids(names: Iterable[str], prefix: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    used = set()
    for name in names:
        base = f"{prefix}_{sanitize_case_name(name)}"
        cid = base
        counter = 1
        while cid in used:
            cid = f"{base}_{counter:02d}"
            counter += 1
        used.add(cid)
        out[name] = cid
    return out


def source_stem(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    return path.stem


def find_nifti_files(input_dir: Path) -> List[Path]:
    nii_gz = sorted(input_dir.rglob("*.nii.gz"))
    nii = sorted(input_dir.rglob("*.nii"))

    # Avoid duplicate handling in unusual cases where both glob patterns could map to same path.
    all_files = []
    seen = set()
    for p in nii_gz + nii:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            all_files.append(p)
    return all_files


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nifti_files = find_nifti_files(args.input_dir)
    if not nifti_files:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in {args.input_dir}")

    stems = [source_stem(p) for p in nifti_files]
    case_id_map = unique_case_ids(stems, args.prefix)

    manifest_path = args.manifest or (args.output_dir / "case_manifest.csv")
    rows: List[Dict[str, str]] = []

    for src in nifti_files:
        stem = source_stem(src)
        case_id = case_id_map[stem]
        dst = args.output_dir / f"{case_id}_0000.nii.gz"

        # Always read+write to guarantee consistent gzip output naming.
        img = sitk.ReadImage(str(src))
        sitk.WriteImage(img, str(dst), useCompression=True)

        rows.append(
            {
                "source_nifti": str(src.resolve()),
                "source_stem": stem,
                "case_id": case_id,
                "output_nii": str(dst.resolve()),
            }
        )
        print(f"Wrote {dst.name} from {src.name}")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
