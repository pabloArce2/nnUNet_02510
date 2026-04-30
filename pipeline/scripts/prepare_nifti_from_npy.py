#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import SimpleITK as sitk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert 3D CT .npy files (Z,Y,X) to NIfTI for TotalSegmentator/nnU-Net."
    )
    p.add_argument("--input-dir", type=Path, required=True, help="Folder with .npy CT volumes")
    p.add_argument("--output-dir", type=Path, required=True, help="Destination folder for .nii.gz volumes")
    p.add_argument(
        "--default-spacing",
        type=float,
        nargs=3,
        metavar=("SX", "SY", "SZ"),
        default=[1.0, 1.0, 1.0],
        help="Voxel spacing in mm as x y z when per-case spacing is not provided",
    )
    p.add_argument(
        "--spacing-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON mapping case_id -> [sx, sy, sz]. "
            "Use when you know per-case spacing from scanner metadata."
        ),
    )
    p.add_argument(
        "--clip",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=[-1024.0, 2000.0],
        help="Clip intensity range before writing NIfTI",
    )
    p.add_argument("--prefix", type=str, default="ct", help="Case id prefix used in output file names")
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV path to save source/target case mapping",
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


def load_spacing_map(path: Path | None) -> Dict[str, Tuple[float, float, float]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    out: Dict[str, Tuple[float, float, float]] = {}
    for k, v in data.items():
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError(f"Invalid spacing entry for {k}: {v}")
        out[k] = (float(v[0]), float(v[1]), float(v[2]))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.output_dir / "case_manifest.csv")

    npy_files = sorted(args.input_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {args.input_dir}")

    source_names = [f.stem for f in npy_files]
    case_id_map = unique_case_ids(source_names, args.prefix)
    spacing_map = load_spacing_map(args.spacing_json)

    rows: List[Dict[str, str]] = []
    low, high = float(args.clip[0]), float(args.clip[1])

    for npy_path in npy_files:
        source_name = npy_path.stem
        case_id = case_id_map[source_name]
        output_nii = args.output_dir / f"{case_id}_0000.nii.gz"

        vol = np.load(npy_path)
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume in {npy_path}, got shape {vol.shape}")

        vol = vol.astype(np.float32, copy=False)
        vol = np.clip(vol, low, high)

        img = sitk.GetImageFromArray(vol)
        spacing = spacing_map.get(case_id, tuple(args.default_spacing))
        img.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))

        sitk.WriteImage(img, str(output_nii), useCompression=True)

        rows.append(
            {
                "source_npy": str(npy_path.resolve()),
                "source_stem": source_name,
                "case_id": case_id,
                "spacing_x": str(spacing[0]),
                "spacing_y": str(spacing[1]),
                "spacing_z": str(spacing[2]),
                "output_nii": str(output_nii.resolve()),
            }
        )
        print(f"Wrote {output_nii.name} from {npy_path.name}")

    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
