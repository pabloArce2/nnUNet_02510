#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'SimpleITK'. Install with: pip install SimpleITK") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert NPY probability tensors to argmax NIfTI masks")
    p.add_argument("--input", type=Path, required=True, help="Input .npy file or directory with .npy probability tensors")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--case-id", type=str, default=None, help="Case id override for single-file mode")
    p.add_argument("--axis-order", choices=["zyx", "xyz"], default="zyx", help="Spatial order for resulting 3D labels")
    p.add_argument("--class-axis", type=int, default=-1, help="Class axis in probability tensor (default: -1)")
    p.add_argument("--write-prob-channels", action="store_true", help="Also write each probability channel to NIfTI")
    p.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"))
    return p.parse_args()


def sanitize_case_id(token: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    clean = re.sub(r"_+", "_", clean)
    clean = re.sub(r"_probabilities?$", "", clean)
    return clean


def to_zyx(arr: np.ndarray, axis_order: str) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")
    if axis_order == "xyz":
        return np.transpose(arr, (2, 1, 0))
    return arr


def write_nifti(arr_zyx: np.ndarray, out_path: Path, spacing_xyz: tuple[float, float, float], is_label: bool) -> None:
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    img.SetSpacing(spacing_xyz)
    if is_label:
        img = sitk.Cast(img, sitk.sitkUInt8)
    else:
        img = sitk.Cast(img, sitk.sitkFloat32)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    files = sorted([p for p in path.glob("*.npy") if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No .npy files found in directory: {path}")
    return files


def process_one(npy_path: Path, args: argparse.Namespace, case_override: str | None = None) -> dict:
    case_id = sanitize_case_id(case_override or npy_path.stem)
    probs = np.asarray(np.load(npy_path, mmap_mode="r"), dtype=np.float32)

    if probs.ndim != 4:
        raise ValueError(f"Expected 4D probability tensor, got shape {probs.shape}")

    class_axis = int(args.class_axis)
    if class_axis < 0:
        class_axis += probs.ndim
    if class_axis < 0 or class_axis >= probs.ndim:
        raise ValueError(f"Invalid class axis {args.class_axis} for shape {probs.shape}")

    labels = np.argmax(probs, axis=class_axis).astype(np.uint8)
    labels_zyx = to_zyx(labels, args.axis_order)

    out_label = args.output_dir / f"{case_id}_argmax_labels.nii.gz"
    spacing = (float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2]))
    write_nifti(labels_zyx, out_label, spacing, is_label=True)

    uniq, counts = np.unique(labels, return_counts=True)
    summary = {
        "source_npy": str(npy_path.resolve()),
        "shape_probs": [int(v) for v in probs.shape],
        "class_axis": class_axis,
        "shape_argmax": [int(v) for v in labels.shape],
        "shape_argmax_zyx": [int(v) for v in labels_zyx.shape],
        "class_histogram": {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())},
        "argmax_output": str(out_label.resolve()),
    }

    if args.write_prob_channels:
        probs_moved = np.moveaxis(probs, class_axis, -1)
        ch_paths: list[str] = []
        for c in range(probs_moved.shape[-1]):
            ch = to_zyx(probs_moved[..., c], args.axis_order)
            out_ch = args.output_dir / f"{case_id}_prob_ch{c}.nii.gz"
            write_nifti(ch, out_ch, spacing, is_label=False)
            ch_paths.append(str(out_ch.resolve()))
        summary["probability_outputs"] = ch_paths

    summary_path = args.output_dir / f"{case_id}_argmax_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args.input)
    for i, npy_path in enumerate(inputs):
        case_override = args.case_id if len(inputs) == 1 and i == 0 else None
        summary = process_one(npy_path, args, case_override=case_override)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
