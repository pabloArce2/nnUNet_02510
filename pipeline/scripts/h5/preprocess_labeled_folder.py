#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'h5py'. Install with: pip install h5py") from exc

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'SimpleITK'. Install with: pip install SimpleITK") from exc


@dataclass
class PreprocessConfig:
    image_key: str | None
    label_key: str | None
    label_only: bool
    image_ref_dir: Path | None
    label_threshold: float
    liver_label: int
    axis_order: str
    source_spacing: tuple[float, float, float]
    target_spacing: tuple[float, float, float] | None
    clean_min_island_voxels: int
    closing_radius: int
    opening_radius: int
    fill_holes: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess a folder of labeled H5 volumes to NIfTI image/label pairs")
    p.add_argument("--input-dir", type=Path, required=True, help="Folder with .h5 files")
    p.add_argument("--output-images-dir", type=Path, required=True, help="Output folder for <case>_0000.nii.gz")
    p.add_argument("--output-labels-dir", type=Path, required=True, help="Output folder for <case>.nii.gz")
    p.add_argument("--image-key", type=str, default=None)
    p.add_argument("--label-key", type=str, default=None)
    p.add_argument("--label-only", action="store_true", help="Process label/probability dataset even if H5 has no image dataset")
    p.add_argument(
        "--image-ref-dir",
        type=Path,
        default=None,
        help="Optional folder with reference CT NIfTI files named <case>_0000.nii.gz; used in --label-only mode",
    )
    p.add_argument("--label-threshold", type=float, default=0.5)
    p.add_argument("--liver-label", type=int, default=1)
    p.add_argument("--axis-order", choices=["zyx", "xyz"], default="zyx", help="How to interpret raw arrays")
    p.add_argument("--source-spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"))
    p.add_argument("--target-spacing", type=float, nargs=3, default=None, metavar=("TX", "TY", "TZ"))
    p.add_argument("--clean-min-island-voxels", type=int, default=200)
    p.add_argument("--closing-radius", type=int, default=1)
    p.add_argument("--opening-radius", type=int, default=1)
    p.add_argument("--fill-holes", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--summary-json", type=Path, default=None)
    return p.parse_args()


def sanitize_token(token: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    return re.sub(r"_+", "_", clean)


def normalize_case_key(token: str) -> str:
    t = sanitize_token(token)
    t = re.sub(r"^(ct_)", "", t)
    t = re.sub(r"(_probabilities|_probability|_labels|_label|_mask|_segmentation|_seg)$", "", t)
    return t


def pick_case_id(stem: str, case_map: dict[str, str] | None) -> str:
    raw = sanitize_token(stem)
    nkey = normalize_case_key(raw)
    if case_map is None:
        return nkey
    return case_map.get(nkey, raw)


def resolve_reference_image(image_ref_dir: Path, case_id: str) -> Path | None:
    direct = image_ref_dir / f"{case_id}_0000.nii.gz"
    if direct.exists():
        return direct

    target = normalize_case_key(case_id)
    for p in image_ref_dir.glob("*_0000.nii.gz"):
        stem = p.name[: -len("_0000.nii.gz")]
        if normalize_case_key(stem) == target:
            return p
    return None


def dataset_paths(h5f: h5py.File) -> list[str]:
    out: list[str] = []

    def collect(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            out.append("/" + name)

    h5f.visititems(collect)
    return sorted(out)


def choose_keys(paths: Sequence[str], image_key: str | None, label_key: str | None) -> tuple[str, str]:
    if image_key and label_key:
        return image_key, label_key

    low = [p.lower() for p in paths]
    label_tokens = ("label", "mask", "seg", "prob", "pred", "class", "ilastik", "exported_data")
    image_tokens = ("raw", "image", "volume", "input", "scan", "ct", "mri")

    def find_first(candidates: Sequence[str]) -> str | None:
        for needle in candidates:
            for i, p in enumerate(low):
                if needle in p:
                    return paths[i]
        return None

    def find_image_key() -> str | None:
        # Prefer explicit image-like names that are not also label-like.
        for i, p in enumerate(low):
            if any(tok in p for tok in image_tokens) and not any(tok in p for tok in label_tokens):
                return paths[i]
        # Fallback: best-effort by common image aliases.
        return find_first(["raw", "image", "volume", "input", "scan", "ct", "mri"])

    guessed_image = image_key or find_image_key()
    guessed_label = label_key or find_first(
        [
            "label",
            "mask",
            "seg",
            "probabilities",
            "probability",
            "prob",
            "prediction",
            "pred",
            "class",
            "ilastik",
            "exported_data",
        ]
    )

    # Avoid resolving both keys to the same dataset when auto-detecting.
    if guessed_image is not None and guessed_label == guessed_image and label_key is None:
        for i, p in enumerate(low):
            if paths[i] != guessed_image and any(k in p for k in label_tokens):
                guessed_label = paths[i]
                break

    if guessed_image is None and guessed_label is not None:
        raise ValueError(
            "Detected a label/probability dataset but no image dataset. "
            f"Available datasets: {', '.join(paths)}. "
            "This usually means the H5 is a label-only ILASTIK export (e.g. /exported_data). "
            "Use this script on H5 files that contain both raw image and label, or provide raw images from another source."
        )

    if guessed_image is None or guessed_label is None:
        missing = []
        if guessed_image is None:
            missing.append("image")
        if guessed_label is None:
            missing.append("label")
        raise ValueError(
            f"Could not auto-detect {', '.join(missing)} key(s). "
            f"Use --image-key/--label-key. Available datasets: {', '.join(paths)}"
        )
    return guessed_image, guessed_label


def choose_label_key(paths: Sequence[str], label_key: str | None) -> str:
    if label_key:
        return label_key
    low = [p.lower() for p in paths]
    candidates = ["label", "mask", "seg", "probabilities", "probability", "prob", "prediction", "pred", "class", "ilastik", "exported_data"]
    for needle in candidates:
        for i, p in enumerate(low):
            if needle in p:
                return paths[i]
    if len(paths) == 1:
        return paths[0]
    raise ValueError(f"Could not auto-detect label key. Use --label-key. Available datasets: {', '.join(paths)}")


def to_zyx(arr: np.ndarray, axis_order: str) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        # Handle channel-last image volumes (e.g., RGB/RGBA) by collapsing to grayscale.
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D array, got shape {arr.shape}. "
            "If this is a multi-class probability tensor, it is likely the label dataset was selected as image; "
            "set --image-key and --label-key explicitly."
        )
    if axis_order == "xyz":
        return np.transpose(arr, (2, 1, 0))
    return arr


def to_discrete_labels(label_arr: np.ndarray, threshold: float) -> np.ndarray:
    if np.issubdtype(label_arr.dtype, np.integer):
        return label_arr.astype(np.uint8, copy=False)

    if label_arr.ndim == 4:
        # Support both channel-first (C,Z,Y,X) and channel-last (Z,Y,X,C) probabilities.
        if label_arr.shape[0] == 1:
            return (label_arr[0] >= threshold).astype(np.uint8)
        if label_arr.shape[-1] == 1:
            return (label_arr[..., 0] >= threshold).astype(np.uint8)

        first_is_channel = label_arr.shape[0] <= 8 and label_arr.shape[-1] > 8
        last_is_channel = label_arr.shape[-1] <= 8 and label_arr.shape[0] > 8

        if first_is_channel and not last_is_channel:
            return np.argmax(label_arr, axis=0).astype(np.uint8)
        if last_is_channel and not first_is_channel:
            return np.argmax(label_arr, axis=-1).astype(np.uint8)

        # Ambiguous case: choose the smaller edge axis as channel axis.
        axis = 0 if label_arr.shape[0] <= label_arr.shape[-1] else -1
        return np.argmax(label_arr, axis=axis).astype(np.uint8)

    return (label_arr >= threshold).astype(np.uint8)


def binary_structure(radius: int) -> np.ndarray:
    base = ndimage.generate_binary_structure(3, 1)
    if radius <= 1:
        return base
    return ndimage.iterate_structure(base, radius)


def keep_large_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    labeled, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    for i, s in enumerate(sizes, start=1):
        if int(s) >= min_voxels:
            keep[i] = True
    if not keep[1:].any():
        keep[np.argmax(sizes) + 1] = True
    return keep[labeled]


def clean_liver_mask(mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    out = keep_large_components(mask, cfg.clean_min_island_voxels)
    if cfg.fill_holes:
        out = ndimage.binary_fill_holes(out)
    if cfg.closing_radius > 0:
        out = ndimage.binary_closing(out, structure=binary_structure(cfg.closing_radius))
    if cfg.opening_radius > 0:
        out = ndimage.binary_opening(out, structure=binary_structure(cfg.opening_radius))
    out = keep_large_components(out, cfg.clean_min_island_voxels)
    return out.astype(bool, copy=False)


def as_sitk(arr_zyx: np.ndarray, spacing_xyz: tuple[float, float, float], is_label: bool) -> sitk.Image:
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    img.SetSpacing(spacing_xyz)
    if is_label:
        img = sitk.Cast(img, sitk.sitkUInt8)
    else:
        img = sitk.Cast(img, sitk.sitkFloat32)
    return img


def resample_pair(image_arr: np.ndarray, label_arr: np.ndarray, src_spacing: tuple[float, float, float], tgt_spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    image_sitk = as_sitk(image_arr, src_spacing, is_label=False)
    label_sitk = as_sitk(label_arr, src_spacing, is_label=True)

    orig_size = np.array(image_sitk.GetSize(), dtype=np.int32)
    orig_spacing = np.array(image_sitk.GetSpacing(), dtype=np.float64)
    tgt_spacing_np = np.array(tgt_spacing, dtype=np.float64)
    new_size = np.maximum(1, np.round(orig_size * (orig_spacing / tgt_spacing_np)).astype(np.int32))

    def _resample(inp: sitk.Image, interp: int) -> sitk.Image:
        rs = sitk.ResampleImageFilter()
        rs.SetInterpolator(interp)
        rs.SetOutputSpacing(tuple(float(v) for v in tgt_spacing_np))
        rs.SetSize([int(v) for v in new_size.tolist()])
        rs.SetOutputDirection(inp.GetDirection())
        rs.SetOutputOrigin(inp.GetOrigin())
        rs.SetDefaultPixelValue(0)
        return rs.Execute(inp)

    image_out = _resample(image_sitk, sitk.sitkBSpline)
    label_out = _resample(label_sitk, sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(image_out), sitk.GetArrayFromImage(label_out)


def preprocess_one(h5_path: Path, cfg: PreprocessConfig, out_images: Path, out_labels: Path, case_map: dict[str, str] | None = None, dry_run: bool = False) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        paths = dataset_paths(f)
        if cfg.label_only:
            label_key = choose_label_key(paths, cfg.label_key)
            image_key = None
            image_raw = None
            label_raw = np.asarray(f[label_key][...])
        else:
            image_key, label_key = choose_keys(paths, cfg.image_key, cfg.label_key)
            image_raw = np.asarray(f[image_key][...])
            label_raw = np.asarray(f[label_key][...])

    if cfg.label_only:
        image = None
    else:
        assert image_raw is not None
        image = to_zyx(image_raw, cfg.axis_order).astype(np.float32, copy=False)
    label = to_zyx(to_discrete_labels(label_raw, cfg.label_threshold), cfg.axis_order).astype(np.uint8, copy=False)

    liver_mask = label == int(cfg.liver_label)
    liver_mask = clean_liver_mask(liver_mask, cfg)

    cleaned_label = np.where(liver_mask, int(cfg.liver_label), 0).astype(np.uint8, copy=False)

    crop_box = {"z0": 0, "z1": int(label.shape[0]), "y0": 0, "y1": int(label.shape[1]), "x0": 0, "x1": int(label.shape[2])}

    spacing = cfg.source_spacing
    if cfg.target_spacing is not None and image is not None:
        image, cleaned_label = resample_pair(image, cleaned_label, cfg.source_spacing, cfg.target_spacing)
        spacing = cfg.target_spacing

    case_id = pick_case_id(h5_path.stem, case_map)
    out_img = out_images / f"{case_id}_0000.nii.gz"
    out_lab = out_labels / f"{case_id}.nii.gz"

    meta = {
        "source_h5": str(h5_path.resolve()),
        "case_id": case_id,
        "image_key": image_key,
        "label_key": label_key,
        "shape_out_zyx": [int(v) for v in (image.shape if image is not None else cleaned_label.shape)],
        "spacing_xyz": [float(v) for v in spacing],
        "crop_box_zyx": crop_box,
        "image_output": str(out_img.resolve()) if image is not None else None,
        "label_output": str(out_lab.resolve()),
    }

    if not dry_run:
        out_labels.mkdir(parents=True, exist_ok=True)
        if image is not None:
            out_images.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(as_sitk(image, spacing, is_label=False), str(out_img), useCompression=True)
            sitk.WriteImage(as_sitk(cleaned_label, spacing, is_label=True), str(out_lab), useCompression=True)
        else:
            ref_path = resolve_reference_image(cfg.image_ref_dir, case_id) if cfg.image_ref_dir is not None else None
            if ref_path is not None:
                ref_img = sitk.ReadImage(str(ref_path))
                lbl_img = sitk.GetImageFromArray(cleaned_label.astype(np.uint8, copy=False))
                if tuple(int(v) for v in lbl_img.GetSize()) != tuple(int(v) for v in ref_img.GetSize()):
                    rs = sitk.ResampleImageFilter()
                    rs.SetReferenceImage(ref_img)
                    rs.SetInterpolator(sitk.sitkNearestNeighbor)
                    rs.SetDefaultPixelValue(0)
                    lbl_img = rs.Execute(lbl_img)
                lbl_img.SetOrigin(ref_img.GetOrigin())
                lbl_img.SetDirection(ref_img.GetDirection())
                lbl_img.SetSpacing(ref_img.GetSpacing())
                sitk.WriteImage(lbl_img, str(out_lab), useCompression=True)
            else:
                sitk.WriteImage(as_sitk(cleaned_label, spacing, is_label=True), str(out_lab), useCompression=True)

    return meta


def preprocess_folder(input_dir: Path, out_images: Path, out_labels: Path, cfg: PreprocessConfig, case_ids: Sequence[str] | None = None, dry_run: bool = False) -> list[dict[str, Any]]:
    case_map = None
    if case_ids:
        tmp: dict[str, str] = {}
        for cid in case_ids:
            key = normalize_case_key(cid)
            if key in tmp and tmp[key] != cid:
                continue
            tmp[key] = cid
        case_map = tmp

    files = sorted(input_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    reports: list[dict[str, Any]] = []
    for fpath in files:
        reports.append(preprocess_one(fpath, cfg, out_images, out_labels, case_map=case_map, dry_run=dry_run))
    return reports


def main() -> None:
    args = parse_args()
    cfg = PreprocessConfig(
        image_key=args.image_key,
        label_key=args.label_key,
        label_only=bool(args.label_only),
        image_ref_dir=args.image_ref_dir,
        label_threshold=float(args.label_threshold),
        liver_label=int(args.liver_label),
        axis_order=args.axis_order,
        source_spacing=(float(args.source_spacing[0]), float(args.source_spacing[1]), float(args.source_spacing[2])),
        target_spacing=(float(args.target_spacing[0]), float(args.target_spacing[1]), float(args.target_spacing[2])) if args.target_spacing else None,
        clean_min_island_voxels=int(args.clean_min_island_voxels),
        closing_radius=int(args.closing_radius),
        opening_radius=int(args.opening_radius),
        fill_holes=bool(args.fill_holes),
    )

    reports = preprocess_folder(
        input_dir=args.input_dir,
        out_images=args.output_images_dir,
        out_labels=args.output_labels_dir,
        cfg=cfg,
        dry_run=bool(args.dry_run),
    )

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "output_images_dir": str(args.output_images_dir.resolve()),
        "output_labels_dir": str(args.output_labels_dir.resolve()),
        "dry_run": bool(args.dry_run),
        "count": len(reports),
        "cases": reports,
    }
    text = json.dumps(summary, indent=2)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
