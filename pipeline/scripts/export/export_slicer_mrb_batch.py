#!/usr/bin/env python3
"""Batch-export one Slicer .mrb scene per CT+liver-mask case.

Run with Slicer, for example:
  Slicer --no-main-window --python-script pipeline/scripts/export/export_slicer_mrb_batch.py -- \
    --ct-dir pipeline/work/stage1/ct_nifti \
    --mask-dir pipeline/work/stage1/pseudolabels/liver_masks \
    --output-dir pipeline/work/stage1/slicer_mrb
"""

from __future__ import annotations

import argparse
from pathlib import Path


# This module is only available inside Slicer.
try:
    import slicer  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime guard for regular Python
    raise RuntimeError(
        "This script must be run via 3D Slicer, for example:\n"
        "  Slicer --no-main-window --python-script pipeline/scripts/export/export_slicer_mrb_batch.py -- ...\n"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export one .mrb scene per matched CT + liver mask case")
    p.add_argument("--ct-dir", type=Path, required=True, help="Directory containing *_0000.nii.gz CT files")
    p.add_argument("--mask-dir", type=Path, required=True, help="Directory containing <case_id>.nii.gz masks")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory where .mrb files are saved")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .mrb files")
    p.add_argument(
        "--skip-empty-masks",
        action="store_true",
        help="Skip cases where loaded segmentation contains zero segments",
    )
    return p.parse_args()


def case_id_from_ct_name(name: str) -> str:
    if not name.endswith("_0000.nii.gz"):
        raise ValueError(f"Invalid CT filename (expected *_0000.nii.gz): {name}")
    return name[: -len("_0000.nii.gz")]


def load_volume(path: Path):
    ok, node = slicer.util.loadVolume(str(path), returnNode=True)
    if not ok or node is None:
        raise RuntimeError(f"Failed to load volume: {path}")
    return node


def load_segmentation(path: Path):
    ok, node = slicer.util.loadSegmentation(str(path), returnNode=True)
    if not ok or node is None:
        raise RuntimeError(f"Failed to load segmentation: {path}")
    return node


def segmentation_is_empty(segmentation_node) -> bool:
    return segmentation_node.GetSegmentation().GetNumberOfSegments() == 0


def clear_scene() -> None:
    slicer.mrmlScene.Clear(0)


def main() -> None:
    args = parse_args()

    ct_map = {
        case_id_from_ct_name(p.name): p
        for p in sorted(args.ct_dir.glob("*_0000.nii.gz"))
    }
    mask_map = {
        p.name.replace(".nii.gz", ""): p
        for p in sorted(args.mask_dir.glob("*.nii.gz"))
    }

    common_case_ids = sorted(set(ct_map.keys()) & set(mask_map.keys()))
    if not common_case_ids:
        raise RuntimeError("No matching case IDs found between CT and mask directories")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_exists = 0
    skipped_empty = 0

    for case_id in common_case_ids:
        out_file = args.output_dir / f"{case_id}.mrb"

        if out_file.exists() and not args.overwrite:
            print(f"[skip] {case_id}: {out_file.name} exists (use --overwrite)")
            skipped_exists += 1
            continue

        clear_scene()
        ct_node = load_volume(ct_map[case_id])
        seg_node = load_segmentation(mask_map[case_id])

        if args.skip_empty_masks and segmentation_is_empty(seg_node):
            print(f"[skip] {case_id}: segmentation has zero segments")
            skipped_empty += 1
            continue

        slicer.util.setSliceViewerLayers(background=ct_node, fit=True)
        seg_node.GetDisplayNode().SetVisibility2DFill(True)
        seg_node.GetDisplayNode().SetVisibility2DOutline(True)

        if not slicer.util.saveScene(str(out_file)):
            raise RuntimeError(f"Failed to save scene: {out_file}")

        saved += 1
        print(f"[ok] {out_file}")

    print(
        "Done. "
        f"saved={saved}, skipped_existing={skipped_exists}, skipped_empty={skipped_empty}, total_matched={len(common_case_ids)}"
    )


if __name__ == "__main__":
    main()
