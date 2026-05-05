#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'SimpleITK'. Install with: pip install SimpleITK") from exc


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
    p = argparse.ArgumentParser(description="Postprocess NPY probabilities into refined argmax labels")
    p.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "dataset" / "pig_npy_labeled",
        help="NPY file or directory containing NPY probability files",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--case-id", type=str, default=None, help="Used only when processing a single file")
    p.add_argument("--axis-order", choices=["zyx", "xyz"], default="zyx")
    p.add_argument("--class-axis", type=int, default=-1)
    p.add_argument("--pig-class", type=int, default=1)
    p.add_argument("--liver-class", type=int, default=0)
    p.add_argument("--bg-class", type=int, default=2)
    p.add_argument("--k-neighbors", type=int, default=30)
    p.add_argument("--hole-fill-method", choices=["knn", "morph"], default="morph")
    p.add_argument("--max-hole-size-voxels", type=int, default=15000)
    p.add_argument("--morph-connectivity", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--morph-opening-iterations", type=int, default=2)
    p.add_argument("--morph-closing-iterations", type=int, default=1)
    p.add_argument("--max-small-component-voxels", type=int, default=6000)
    p.add_argument("--low-confidence-threshold", type=float, default=0.7)
    p.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"))
    p.add_argument("--image-ref", type=Path, default=None, help="Optional reference CT for geometry")
    return p.parse_args()


def sanitize_case_id(token: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", token).strip("_").lower()
    clean = re.sub(r"_+", "_", clean)
    clean = re.sub(r"_probabilities?$", "", clean)
    return clean


def to_zyx_3d(arr: np.ndarray, axis_order: str) -> np.ndarray:
    if axis_order == "xyz":
        return np.transpose(arr, (2, 1, 0))
    return arr


def to_zyx_4d(arr: np.ndarray, axis_order: str) -> np.ndarray:
    if axis_order == "xyz":
        return np.transpose(arr, (2, 1, 0, 3))
    return arr


def write_label_nifti(
    arr_zyx: np.ndarray,
    out_path: Path,
    spacing_xyz: tuple[float, float, float],
    ref_img: sitk.Image | None,
) -> None:
    img = sitk.GetImageFromArray(arr_zyx.astype(np.uint8, copy=False))

    if ref_img is not None:
        img.SetOrigin(ref_img.GetOrigin())
        img.SetDirection(ref_img.GetDirection())
        img.SetSpacing(ref_img.GetSpacing())
    else:
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        img.SetSpacing(spacing_xyz)

    sitk.WriteImage(img, str(out_path), useCompression=True)


def fill_holes_knn(
    labels: np.ndarray,
    probs: np.ndarray,
    pig_class: int,
    liver_class: int,
    bg_class: int,
    k: int,
    max_hole_size: int,
) -> tuple[np.ndarray, int]:
    out = labels.copy()
    fg = (out == pig_class) | (out == liver_class)
    holes = ndimage.binary_fill_holes(fg) & (out == bg_class)

    if not np.any(holes):
        return out, 0

    cc, _ = ndimage.label(holes, structure=ndimage.generate_binary_structure(3, 1))
    objs = ndimage.find_objects(cc)
    changed = 0

    for comp_id, slc in enumerate(objs, start=1):
        if slc is None:
            continue

        comp = cc[slc] == comp_id
        comp_size = int(comp.sum())

        if comp_size == 0 or comp_size > max_hole_size:
            continue

        ext = []
        for s, dim in zip(slc, out.shape):
            lo = max(0, int(s.start) - 20)
            hi = min(dim, int(s.stop) + 20)
            ext.append(slice(lo, hi))
        ext = tuple(ext)

        ext_lbl = out[ext]
        ext_prb = probs[ext]
        ext_fg = (ext_lbl == pig_class) | (ext_lbl == liver_class)
        coords = np.argwhere(ext_fg)

        if coords.shape[0] == 0:
            continue

        classes = ext_lbl[ext_fg]
        class_prob = ext_prb[ext_fg, classes]
        k_eff = max(1, min(int(k), int(coords.shape[0])))
        tree = cKDTree(coords)

        local_coords = np.argwhere(comp)
        offset = np.array(
            [
                int(slc[0].start) - int(ext[0].start),
                int(slc[1].start) - int(ext[1].start),
                int(slc[2].start) - int(ext[2].start),
            ]
        )
        query_pts = local_coords + offset[None, :]

        _, idx = tree.query(query_pts, k=k_eff)

        if k_eff == 1:
            idx = idx[:, None]

        nbr_cls = classes[idx]
        nbr_prb = class_prob[idx]
        pig_votes = np.sum((nbr_cls == pig_class) * nbr_prb, axis=1)
        liver_votes = np.sum((nbr_cls == liver_class) * nbr_prb, axis=1)
        assign = np.where(liver_votes > pig_votes, liver_class, pig_class).astype(np.uint8)

        local = out[slc].copy()
        local[comp] = assign
        out[slc] = local
        changed += comp_size

    return out, changed


def fill_holes_morph(
    labels: np.ndarray,
    probs: np.ndarray,
    pig_class: int,
    liver_class: int,
    bg_class: int,
    connectivity: int,
    opening_iterations: int,
    closing_iterations: int,
) -> tuple[np.ndarray, int]:
    out = labels.copy()
    structure = ndimage.generate_binary_structure(3, int(connectivity))
    fg = (out == pig_class) | (out == liver_class)

    if int(closing_iterations) > 0:
        fg = ndimage.binary_closing(fg, structure=structure, iterations=int(closing_iterations))

    fg = ndimage.binary_fill_holes(fg)

    if int(opening_iterations) > 0:
        fg = ndimage.binary_opening(fg, structure=structure, iterations=int(opening_iterations))

    add_fg = fg & (out == bg_class)
    rem_fg = (~fg) & ((out == pig_class) | (out == liver_class))

    if np.any(add_fg):
        pig_p = probs[..., pig_class][add_fg]
        liver_p = probs[..., liver_class][add_fg]
        out[add_fg] = np.where(liver_p > pig_p, liver_class, pig_class).astype(np.uint8)

    if np.any(rem_fg):
        out[rem_fg] = np.uint8(bg_class)

    changed = int(np.count_nonzero(add_fg) + np.count_nonzero(rem_fg))
    return out, changed


def relabel_low_conf_small_components(
    labels: np.ndarray,
    probs: np.ndarray,
    pig_class: int,
    liver_class: int,
    bg_class: int,
    max_size: int,
    conf_thr: float,
) -> tuple[np.ndarray, int]:
    out = labels.copy()
    changed = 0
    structure = ndimage.generate_binary_structure(3, 1)

    for src in (pig_class, liver_class):
        cc, _ = ndimage.label(out == src, structure=structure)
        objs = ndimage.find_objects(cc)

        for comp_id, slc in enumerate(objs, start=1):
            if slc is None:
                continue

            comp = cc[slc] == comp_id
            comp_size = int(comp.sum())

            if comp_size == 0 or comp_size > max_size:
                continue

            local_probs = probs[slc]
            mean_conf = float(local_probs[comp, src].mean())

            if mean_conf >= conf_thr:
                continue

            local_labels = out[slc].copy()

            if src == pig_class:
                liver_p = local_probs[..., liver_class][comp]
                bg_p = local_probs[..., bg_class][comp]
                new_vals = np.where(liver_p > bg_p, liver_class, bg_class).astype(np.uint8)
            else:
                new_vals = np.full(comp_size, pig_class, dtype=np.uint8)

            local_labels[comp] = new_vals
            out[slc] = local_labels
            changed += comp_size

    return out, changed


def enforce_single_component_per_label(
    labels: np.ndarray,
    pig_class: int,
    liver_class: int,
    bg_class: int,
    k_neighbors: int,
) -> tuple[np.ndarray, int, int]:
    out = labels.copy()
    structure = ndimage.generate_binary_structure(3, 1)
    pig_reassigned = 0
    liver_reassigned = 0

    # Liver: keep largest component, reassign all others to pig.
    liver_cc, n_liver = ndimage.label(out == liver_class, structure=structure)
    if n_liver > 1:
        sizes = np.bincount(liver_cc.ravel())
        sizes[0] = 0
        keep_id = int(np.argmax(sizes))
        rem_mask = (liver_cc > 0) & (liver_cc != keep_id)
        liver_reassigned = int(np.count_nonzero(rem_mask))

        if liver_reassigned > 0:
            out[rem_mask] = np.uint8(pig_class)

    # Pig: keep largest component, reassign all others by kNN rule.
    pig_cc, n_pig = ndimage.label(out == pig_class, structure=structure)
    if n_pig > 1:
        sizes = np.bincount(pig_cc.ravel())
        sizes[0] = 0
        keep_id = int(np.argmax(sizes))
        rem_mask = (pig_cc > 0) & (pig_cc != keep_id)
        pig_reassigned = int(np.count_nonzero(rem_mask))

        if pig_reassigned > 0:
            non_pig_mask = ~rem_mask & (out != pig_class)
            non_pig_coords = np.argwhere(non_pig_mask)
            rem_coords = np.argwhere(rem_mask)

            if non_pig_coords.shape[0] == 0:
                out[rem_mask] = np.uint8(bg_class)
            else:
                non_pig_labels = out[non_pig_mask]
                k_eff = max(1, min(int(k_neighbors), int(non_pig_coords.shape[0])))
                tree = cKDTree(non_pig_coords)

                _, idx = tree.query(rem_coords, k=k_eff)

                if k_eff == 1:
                    idx = idx[:, None]

                nbr_labels = non_pig_labels[idx]
                all_liver = np.all(nbr_labels == liver_class, axis=1)
                assign = np.where(all_liver, liver_class, bg_class).astype(np.uint8)
                out[rem_mask] = assign

    return out, pig_reassigned, liver_reassigned


def collect_inputs(path: Path) -> list[Path]:
    path = resolve_repo_path(path)

    if path.is_file():
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")

    files = sorted([p for p in path.glob("*.npy") if p.is_file()])

    if not files:
        raise FileNotFoundError(f"No .npy files found in directory: {path}")

    return files


def process_one(
    npy_path: Path,
    args: argparse.Namespace,
    ref_img: sitk.Image | None,
    case_override: str | None = None,
) -> dict:
    probs_raw = np.asarray(np.load(npy_path, mmap_mode="r"), dtype=np.float32)

    if probs_raw.ndim != 4:
        raise ValueError(f"Expected 4D probability tensor, got shape {probs_raw.shape} for {npy_path}")

    class_axis = int(args.class_axis)

    if class_axis < 0:
        class_axis += probs_raw.ndim

    probs = np.moveaxis(probs_raw, class_axis, -1)
    labels = np.argmax(probs, axis=-1).astype(np.uint8)

    labels_zyx = to_zyx_3d(labels, args.axis_order)
    probs_zyx = to_zyx_4d(probs, args.axis_order)

    if args.hole_fill_method == "knn":
        after_holes, filled = fill_holes_knn(
            labels_zyx,
            probs_zyx,
            pig_class=int(args.pig_class),
            liver_class=int(args.liver_class),
            bg_class=int(args.bg_class),
            k=int(args.k_neighbors),
            max_hole_size=int(args.max_hole_size_voxels),
        )
    else:
        after_holes, filled = fill_holes_morph(
            labels_zyx,
            probs_zyx,
            pig_class=int(args.pig_class),
            liver_class=int(args.liver_class),
            bg_class=int(args.bg_class),
            connectivity=int(args.morph_connectivity),
            opening_iterations=int(args.morph_opening_iterations),
            closing_iterations=int(args.morph_closing_iterations),
        )

    refined, reassigned = relabel_low_conf_small_components(
        after_holes,
        probs_zyx,
        pig_class=int(args.pig_class),
        liver_class=int(args.liver_class),
        bg_class=int(args.bg_class),
        max_size=int(args.max_small_component_voxels),
        conf_thr=float(args.low_confidence_threshold),
    )

    refined, pig_single_reassigned, liver_single_reassigned = enforce_single_component_per_label(
        refined,
        pig_class=int(args.pig_class),
        liver_class=int(args.liver_class),
        bg_class=int(args.bg_class),
        k_neighbors=int(args.k_neighbors),
    )

    case_id = sanitize_case_id(case_override or npy_path.stem)
    out_label = args.output_dir / f"{case_id}_refined_argmax_labels.nii.gz"

    write_label_nifti(
        refined,
        out_label,
        spacing_xyz=(float(args.spacing[0]), float(args.spacing[1]), float(args.spacing[2])),
        ref_img=ref_img,
    )

    u0, c0 = np.unique(labels_zyx, return_counts=True)
    u1, c1 = np.unique(refined, return_counts=True)

    summary = {
        "source_npy": str(npy_path.resolve()),
        "shape_probs": [int(v) for v in probs_raw.shape],
        "class_axis": int(class_axis),
        "class_hist_before": {int(k): int(v) for k, v in zip(u0.tolist(), c0.tolist())},
        "class_hist_after": {int(k): int(v) for k, v in zip(u1.tolist(), c1.tolist())},
        "pig_class": int(args.pig_class),
        "liver_class": int(args.liver_class),
        "bg_class": int(args.bg_class),
        "hole_fill_method": str(args.hole_fill_method),
        "k_neighbors": int(args.k_neighbors),
        "max_hole_size_voxels": int(args.max_hole_size_voxels),
        "morph_connectivity": int(args.morph_connectivity),
        "morph_opening_iterations": int(args.morph_opening_iterations),
        "morph_closing_iterations": int(args.morph_closing_iterations),
        "max_small_component_voxels": int(args.max_small_component_voxels),
        "low_confidence_threshold": float(args.low_confidence_threshold),
        "filled_hole_voxels": int(filled),
        "reassigned_low_conf_voxels": int(reassigned),
        "reassigned_pig_for_single_component": int(pig_single_reassigned),
        "reassigned_liver_for_single_component": int(liver_single_reassigned),
        "output_label": str(out_label.resolve()),
    }

    out_summary = args.output_dir / f"{case_id}_refined_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")

    return summary


def main() -> None:
    args = parse_args()

    args.input = resolve_repo_path(args.input)
    args.output_dir = resolve_repo_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_ref is not None:
        args.image_ref = resolve_repo_path(args.image_ref)

    inputs = collect_inputs(args.input)
    ref_img = sitk.ReadImage(str(args.image_ref)) if args.image_ref is not None else None

    all_summaries: list[dict] = []

    for i, npy_path in enumerate(inputs):
        case_override = args.case_id if len(inputs) == 1 and i == 0 else None
        summary = process_one(npy_path, args, ref_img=ref_img, case_override=case_override)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2))

    if len(all_summaries) > 1:
        batch_summary = args.output_dir / "batch_refined_summary.json"
        batch_summary.write_text(json.dumps(all_summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()