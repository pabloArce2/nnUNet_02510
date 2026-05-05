#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandSpatialCropd,
    ScaleIntensityRanged,
)
from tqdm import tqdm

from monai_common import build_unet, datalist_from_case_ids, load_split, pick_device, set_determinism

try:
    from nibabel.processing import resample_from_to
except ImportError:
    resample_from_to = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MONAI 3D U-Net for segmentation")
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--experiment-name",
        type=str,
        default="seg",
        help="Short name added to timestamped run folder (default: seg)",
    )
    p.add_argument(
        "--flat-output",
        action="store_true",
        help="Write outputs directly into --output-dir instead of a timestamped subfolder",
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96], metavar=("X", "Y", "Z"))
    p.add_argument("--val-interval", type=int, default=5)
    p.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes (including background)")
    p.add_argument("--a-min", type=float, default=-200.0)
    p.add_argument("--a-max", type=float, default=300.0)
    p.add_argument(
        "--gaussian-noise-prob",
        type=float,
        default=0.15,
        help="Probability for RandGaussianNoised on training images (set 0 to disable for debugging)",
    )
    p.add_argument(
        "--flip-prob",
        type=float,
        default=0.5,
        help="Probability for each random flip axis in training augmentation (set 0 to disable flips)",
    )
    p.add_argument(
        "--use-pos-neg-crop",
        action="store_true",
        help="Use RandCropByPosNegLabeld to bias training patches toward liver voxels",
    )
    p.add_argument(
        "--pos-neg-ratio",
        type=float,
        default=2.0,
        help="Positive-to-negative ratio for label-aware crop sampling when --use-pos-neg-crop is set",
    )
    p.add_argument(
        "--train-num-samples",
        type=int,
        default=1,
        help=(
            "Number of random training patches sampled per volume item each step. "
            "Used with --use-pos-neg-crop; effective patches/step ~= batch_size * train_num_samples."
        ),
    )
    p.add_argument(
        "--ce-weights",
        type=float,
        nargs="+",
        default=[1.0, 3.0],
        metavar="W",
        help="Class weights for CE term in DiceCELoss; must match --num-classes",
    )
    p.add_argument(
        "--init-ssl-checkpoint",
        type=Path,
        default=None,
        help="Optional SSL checkpoint from train_monai_ssl.py for encoder initialization",
    )
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    p.add_argument(
        "--save-random-patch-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save 3 random training patches (image + label) at each validation step to <run_dir>/debug_patches",
    )
    p.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory (default: <output-dir>/tensorboard)",
    )
    p.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="Port shown in the TensorBoard launch hint (default: 6006)",
    )
    p.add_argument(
        "--preprocessed-label-dir",
        type=Path,
        default=None,
        help="Optional folder with preprocessed labels named <case_id>.nii.gz",
    )
    p.add_argument(
        "--h5-label-dir",
        type=Path,
        default=None,
        help="Optional folder with labeled .h5 files; if set, run preprocessing before training",
    )
    p.add_argument(
        "--preprocess-output-dir",
        type=Path,
        default=None,
        help="Output directory for auto-preprocessed labels (default: <run_dir>/preprocessed_labels)",
    )
    p.add_argument("--h5-image-key", type=str, default=None, help="Optional H5 dataset key for image volume")
    p.add_argument("--h5-label-key", type=str, default=None, help="Optional H5 dataset key for label/probability")
    p.add_argument("--h5-label-threshold", type=float, default=0.5)
    p.add_argument("--h5-liver-label", type=int, default=1)
    p.add_argument("--h5-axis-order", type=str, choices=["zyx", "xyz"], default="zyx")
    p.add_argument(
        "--h5-source-spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
    )
    p.add_argument(
        "--h5-target-spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("TX", "TY", "TZ"),
    )
    p.add_argument("--h5-clean-min-island-voxels", type=int, default=200)
    p.add_argument("--h5-closing-radius", type=int, default=1)
    p.add_argument("--h5-opening-radius", type=int, default=1)
    p.add_argument("--h5-fill-holes", action="store_true")
    return p.parse_args()


def sanitize_name(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip())
    clean = re.sub(r"-+", "-", clean).strip("-_")
    return clean or "run"


def make_run_dir(output_root: Path, experiment_name: str) -> Path:
    tag = datetime.now().strftime("%y%m%d-%H%M%S")
    stem = f"{tag}_{sanitize_name(experiment_name)}"
    run_dir = output_root / stem
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_latest_pointers(output_root: Path, run_dir: Path, artifact_names: list[str]) -> None:
    (output_root / "latest_run.txt").write_text(str(run_dir.resolve()) + "\n")
    for name in artifact_names:
        src = run_dir / name
        if not src.exists():
            continue
        dst = output_root / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def maybe_load_ssl_weights(model: torch.nn.Module, ckpt_path: Path) -> int:
    # Our own checkpoints include metadata (for example pathlib.Path), so explicitly
    # disable weights_only for compatibility with PyTorch 2.6+ default behavior.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    current = model.state_dict()
    matched = {}
    for key, tensor in state.items():
        if key in current and tuple(current[key].shape) == tuple(tensor.shape):
            matched[key] = tensor
    current.update(matched)
    model.load_state_dict(current)
    return len(matched)


def apply_label_override(split_payload: dict, label_dir: Path) -> int:
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")
    overrides = {p.stem.replace(".nii", ""): str(p.resolve()) for p in label_dir.glob("*.nii.gz")}
    updated = 0
    for row in split_payload.get("cases", []):
        cid = row.get("case_id")
        if cid in overrides:
            row["label"] = overrides[cid]
            row["label_source"] = "preprocessed"
            updated += 1
    return updated


def align_labels_to_images(split_payload: dict, output_dir: Path, case_ids: list[str]) -> int:
    if resample_from_to is None:
        raise RuntimeError(
            "Label/image alignment requires nibabel.processing.resample_from_to. "
            "Install SciPy if missing: pip install scipy"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned = 0
    selected = set(case_ids)

    def fg_stats(arr: np.ndarray) -> tuple[int, float]:
        fg_mask = arr > 0.5
        return int(fg_mask.sum()), float(fg_mask.mean())

    for row in split_payload.get("cases", []):
        cid = row.get("case_id")
        if cid not in selected:
            continue
        image_path = row.get("image")
        label_path = row.get("label")
        if not image_path or not label_path:
            continue
        img = nib.load(image_path)
        lbl = nib.load(label_path)
        same_shape = tuple(lbl.shape) == tuple(img.shape)
        same_affine = (lbl.affine == img.affine).all()
        if same_shape and same_affine:
            continue

        original_np = np.asanyarray(lbl.dataobj)
        orig_fg_vox, orig_fg_frac = fg_stats(original_np)
        fixed = resample_from_to(lbl, (img.shape, img.affine), order=0)
        fixed_np = np.asanyarray(fixed.dataobj)
        fixed_fg_vox, fixed_fg_frac = fg_stats(fixed_np)

        # Fallback for common xyz/zyx confusion: try axis permutations when
        # the original label shape is a permutation of image shape and direct
        # alignment collapses almost all foreground.
        if (
            tuple(sorted(lbl.shape)) == tuple(sorted(img.shape))
            and orig_fg_vox > 0
            and fixed_fg_vox < max(128, int(0.001 * orig_fg_vox))
        ):
            best_img = fixed
            best_fg_vox = fixed_fg_vox
            best_fg_frac = fixed_fg_frac
            best_perm: tuple[int, int, int] | None = None
            lbl_data = np.asanyarray(lbl.dataobj)
            for perm in itertools.permutations((0, 1, 2)):
                permuted = np.transpose(lbl_data, perm)
                if tuple(permuted.shape) != tuple(img.shape):
                    continue
                candidate = nib.Nifti1Image(permuted.astype(lbl_data.dtype, copy=False), img.affine, img.header)
                cand_np = np.asanyarray(candidate.dataobj)
                cand_fg_vox, cand_fg_frac = fg_stats(cand_np)
                if cand_fg_vox > best_fg_vox:
                    best_img = candidate
                    best_fg_vox = cand_fg_vox
                    best_fg_frac = cand_fg_frac
                    best_perm = perm
            if best_perm is not None:
                fixed = best_img
                print(
                    "Auto-reoriented label via axis permutation "
                    f"for {cid}: perm={best_perm}, fg_voxels {fixed_fg_vox} -> {best_fg_vox}, "
                    f"fg_fraction {fixed_fg_frac:.8f} -> {best_fg_frac:.8f}"
                )

        dst = output_dir / f"{cid}.nii.gz"
        nib.save(fixed, str(dst))
        row["label"] = str(dst.resolve())
        row["label_source"] = "aligned_to_image"
        aligned += 1
    return aligned


def validate_training_labels(
    split_payload: dict,
    train_ids: list[str],
    min_fg_fraction: float = 1e-4,
    min_fg_voxels: int = 1000,
) -> None:
    lookup = {row["case_id"]: row for row in split_payload.get("cases", [])}
    issues: list[str] = []
    summaries: list[tuple[str, float, int, tuple[int, ...], tuple[int, ...]]] = []

    for cid in train_ids:
        row = lookup.get(cid)
        if row is None:
            issues.append(f"{cid}: missing from split payload")
            continue
        image_path = row.get("image")
        label_path = row.get("label")
        if not image_path or not label_path:
            issues.append(f"{cid}: missing image or label path")
            continue

        if not Path(image_path).exists():
            issues.append(f"{cid}: image file does not exist: {image_path}")
            continue
        if not Path(label_path).exists():
            issues.append(f"{cid}: label file does not exist: {label_path}")
            continue

        img = nib.load(image_path)
        lbl = nib.load(label_path)
        if tuple(img.shape) != tuple(lbl.shape):
            issues.append(f"{cid}: shape mismatch image={tuple(img.shape)} label={tuple(lbl.shape)}")
            continue
        if not (lbl.affine == img.affine).all():
            issues.append(f"{cid}: affine mismatch between image and label")
            continue

        label_np = lbl.get_fdata(dtype="float32")
        fg_mask = label_np > 0.5
        fg_vox = int(fg_mask.sum())
        fg_frac = float(fg_mask.mean())
        summaries.append((cid, fg_frac, fg_vox, tuple(img.shape), tuple(lbl.shape)))

        if fg_vox < min_fg_voxels or fg_frac < min_fg_fraction:
            issues.append(
                f"{cid}: foreground too small after alignment (fg_voxels={fg_vox}, fg_fraction={fg_frac:.8f})"
            )

    if issues:
        joined = "\n".join(f" - {msg}" for msg in issues)
        raise RuntimeError(
            "Training label preflight failed. Refusing to train with likely broken labels.\n"
            f"{joined}\n"
            "Common cause: axis order/orientation mismatch in preprocessing (xyz vs zyx)."
        )

    fg_values = [s[1] for s in summaries]
    if fg_values:
        print(
            "Label preflight OK: "
            f"n_train={len(summaries)}, fg_fraction min/mean/max="
            f"{min(fg_values):.6f}/{(sum(fg_values)/len(fg_values)):.6f}/{max(fg_values):.6f}"
        )


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = pick_device(args.device)

    split_payload = load_split(args.split_json)
    if args.num_classes < 2:
        raise ValueError("--num-classes must be >= 2")
    if not (0.0 <= args.flip_prob <= 1.0):
        raise ValueError("--flip-prob must be in [0, 1]")
    if not (0.0 <= args.gaussian_noise_prob <= 1.0):
        raise ValueError("--gaussian-noise-prob must be in [0, 1]")
    if args.train_num_samples < 1:
        raise ValueError("--train-num-samples must be >= 1")
    if len(args.ce_weights) != args.num_classes:
        raise ValueError(
            f"--ce-weights expects {args.num_classes} values for --num-classes={args.num_classes}, "
            f"got {len(args.ce_weights)}"
        )

    if args.h5_label_dir is not None:
        try:
            from h5.preprocess_labeled_folder import PreprocessConfig, preprocess_folder
        except ImportError as exc:
            raise RuntimeError(
                "H5 preprocessing requested but dependencies are missing. Install with: pip install h5py scipy SimpleITK"
            ) from exc

        preprocess_output_dir = args.preprocess_output_dir or (args.output_dir / "preprocessed_labels")
        preprocess_output_dir.mkdir(parents=True, exist_ok=True)
        cfg = PreprocessConfig(
            image_key=args.h5_image_key,
            label_key=args.h5_label_key,
            label_threshold=float(args.h5_label_threshold),
            liver_label=int(args.h5_liver_label),
            axis_order=args.h5_axis_order,
            source_spacing=tuple(float(v) for v in args.h5_source_spacing),
            target_spacing=tuple(float(v) for v in args.h5_target_spacing) if args.h5_target_spacing else None,
            clean_min_island_voxels=int(args.h5_clean_min_island_voxels),
            closing_radius=int(args.h5_closing_radius),
            opening_radius=int(args.h5_opening_radius),
            fill_holes=bool(args.h5_fill_holes),
        )
        case_ids = [c["case_id"] for c in split_payload.get("cases", [])]
        reports = preprocess_folder(
            input_dir=args.h5_label_dir,
            out_images=preprocess_output_dir / "images_unused",
            out_labels=preprocess_output_dir,
            cfg=cfg,
            case_ids=case_ids,
            dry_run=False,
        )
        summary_path = preprocess_output_dir / "preprocess_summary.json"
        summary_path.write_text(json.dumps({"count": len(reports), "cases": reports}, indent=2) + "\n")
        matched = apply_label_override(split_payload, preprocess_output_dir)
        print(f"H5 preprocessing complete: {len(reports)} file(s) processed, labels matched to split cases: {matched}")
        print(f"Preprocessed labels dir: {preprocess_output_dir}")
    elif args.preprocessed_label_dir is not None:
        matched = apply_label_override(split_payload, args.preprocessed_label_dir)
        print(f"Using preprocessed labels from {args.preprocessed_label_dir} (matched cases: {matched})")
    train_ids = split_payload["splits"].get("train", [])
    val_ids = split_payload["splits"].get("val", [])

    if not train_ids:
        raise RuntimeError(
            "No labeled training cases in split JSON. Add labels first, rerun create_monai_splits.py, then train."
        )

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root if args.flat_output else make_run_dir(output_root, args.experiment_name)
    print(f"Run output directory: {run_dir}")

    all_labeled_ids = list(dict.fromkeys(train_ids + val_ids))
    aligned_count = align_labels_to_images(split_payload, run_dir / "aligned_labels", all_labeled_ids)
    if aligned_count > 0:
        print(f"Aligned {aligned_count} label(s) to image voxel grids in {run_dir / 'aligned_labels'}")

    train_data = datalist_from_case_ids(split_payload, train_ids, require_label=True)
    val_data = datalist_from_case_ids(split_payload, val_ids, require_label=True) if val_ids else []
    validate_training_labels(split_payload, train_ids)

    crop_transform = (
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=tuple(args.patch_size),
            pos=float(args.pos_neg_ratio),
            neg=1.0,
            num_samples=int(args.train_num_samples),
            image_key="image",
            image_threshold=0.0,
        )
        if args.use_pos_neg_crop
        else RandSpatialCropd(keys=["image", "label"], roi_size=tuple(args.patch_size), random_size=False)
    )

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
            crop_transform,
            RandFlipd(keys=["image", "label"], prob=float(args.flip_prob), spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=float(args.flip_prob), spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=float(args.flip_prob), spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=float(args.gaussian_noise_prob), mean=0.0, std=0.01),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_ds = Dataset(data=train_data, transform=train_transforms)
    val_ds = Dataset(data=val_data, transform=val_transforms) if val_data else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2),
            pin_memory=(device.type == "cuda"),
        )
        if val_ds is not None
        else None
    )

    model = build_unet(out_channels=args.num_classes).to(device)
    if args.init_ssl_checkpoint:
        matched = maybe_load_ssl_weights(model, args.init_ssl_checkpoint)
        print(f"Loaded {matched} matching parameters from SSL checkpoint")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_weights = torch.tensor(args.ce_weights, dtype=torch.float32, device=device)
    try:
        criterion = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=ce_weights)
    except TypeError:
        criterion = DiceCELoss(to_onehot_y=True, softmax=True, weight=ce_weights)

    best_dice = -1.0
    history = []
    global_step = 0

    use_tensorboard = not args.no_tensorboard
    if use_tensorboard and SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard logging requested but not available. Install it with: pip install tensorboard"
        )

    tb_writer = None
    tb_dir = args.tensorboard_dir or (run_dir / "tensorboard")
    if use_tensorboard:
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        tb_writer.add_text("run/device", str(device))
        tb_writer.add_text("run/split_json", str(args.split_json))
        tb_writer.add_scalar("config/lr", float(args.lr), 0)
        tb_writer.add_scalar("config/weight_decay", float(args.weight_decay), 0)
        print(f"TensorBoard logs: {tb_dir}")
        print(
            f"Open TensorBoard: tensorboard --logdir '{tb_dir}' --port {args.tensorboard_port} "
            f"(then open http://localhost:{args.tensorboard_port})"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running = 0.0
        running_fg_fraction = 0.0
        running_class_fraction = torch.zeros(args.num_classes, dtype=torch.float64)
        sampled_patches: list[tuple[np.ndarray, np.ndarray]] = []
        seen_patches = 0
        steps = 0
        pbar = tqdm(train_loader, desc=f"SEG epoch {epoch}/{args.epochs}", leave=False)

        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_item = float(loss.detach().cpu().item())
            y_int = y[:, 0].long()
            class_fracs = []
            for cls_idx in range(args.num_classes):
                frac = float((y_int == cls_idx).float().mean().detach().cpu().item())
                class_fracs.append(frac)
                running_class_fraction[cls_idx] += frac
            fg_fraction_item = 1.0 - class_fracs[0]
            running += loss_item
            running_fg_fraction += fg_fraction_item
            steps += 1
            global_step += 1

            if args.save_random_patch_every_epoch:
                patch_idx = int(torch.randint(low=0, high=x.shape[0], size=(1,)).item())
                candidate_image = x[patch_idx, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                candidate_label = y[patch_idx, 0].detach().cpu().numpy().astype(np.uint8, copy=False)
                seen_patches += 1
                # Reservoir sampling: keep 3 uniformly random patches across all seen patches in this epoch.
                if len(sampled_patches) < 3:
                    sampled_patches.append((candidate_image, candidate_label))
                else:
                    replace_at = int(torch.randint(low=0, high=seen_patches, size=(1,)).item())
                    if replace_at < 3:
                        sampled_patches[replace_at] = (candidate_image, candidate_label)

            pbar.set_postfix(
                loss=f"{loss_item:.5f}",
                fg=f"{fg_fraction_item:.4f}",
                c0=f"{class_fracs[0]:.4f}",
                c1=f"{class_fracs[1]:.4f}" if args.num_classes > 1 else "n/a",
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/step_loss", loss_item, global_step)
                tb_writer.add_scalar("train/step_fg_fraction", fg_fraction_item, global_step)
                for cls_idx, frac in enumerate(class_fracs):
                    tb_writer.add_scalar(f"train/step_class_fraction_c{cls_idx}", frac, global_step)
                tb_writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

        train_loss = running / max(steps, 1)
        train_fg_fraction = running_fg_fraction / max(steps, 1)
        train_class_fractions = (running_class_fraction / max(steps, 1)).tolist()
        epoch_seconds = time.time() - epoch_start
        steps_per_sec = steps / max(epoch_seconds, 1e-8)
        row = {"epoch": epoch, "train_loss": train_loss, "train_fg_fraction": float(train_fg_fraction)}
        for cls_idx, frac in enumerate(train_class_fractions):
            row[f"train_class_fraction_c{cls_idx}"] = float(frac)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch_loss", train_loss, epoch)
            tb_writer.add_scalar("train/epoch_fg_fraction", float(train_fg_fraction), epoch)
            for cls_idx, frac in enumerate(train_class_fractions):
                tb_writer.add_scalar(f"train/epoch_class_fraction_c{cls_idx}", float(frac), epoch)
            tb_writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch)
            tb_writer.add_scalar("train/steps_per_second", steps_per_sec, epoch)

        if val_loader is not None and (epoch % args.val_interval == 0 or epoch == args.epochs):
            model.eval()
            val_running = 0.0
            val_steps = 0
            tp = 0
            fp = 0
            fn = 0
            with torch.no_grad():
                for vbatch in val_loader:
                    vx = vbatch["image"].to(device)
                    vy = vbatch["label"].to(device)
                    logits = sliding_window_inference(vx, tuple(args.patch_size), sw_batch_size=1, predictor=model)
                    vloss = criterion(logits, vy)
                    val_running += float(vloss.detach().cpu().item())
                    val_steps += 1

                    pred_lbl = torch.argmax(logits, dim=1)
                    gt_lbl = (vy[:, 0] > 0.5).long()
                    tp += int(torch.logical_and(pred_lbl == 1, gt_lbl == 1).sum().item())
                    fp += int(torch.logical_and(pred_lbl == 1, gt_lbl == 0).sum().item())
                    fn += int(torch.logical_and(pred_lbl == 0, gt_lbl == 1).sum().item())

            val_loss = val_running / max(val_steps, 1)
            val_dice = (2.0 * tp) / max((2 * tp + fp + fn), 1)
            val_iou = tp / max((tp + fp + fn), 1)
            val_precision = tp / max((tp + fp), 1)
            row["val_loss"] = float(val_loss)
            row["val_dice"] = val_dice
            row["val_iou"] = float(val_iou)
            row["val_precision"] = float(val_precision)
            if tb_writer is not None:
                tb_writer.add_scalar("val/loss", float(val_loss), epoch)
                tb_writer.add_scalar("val/dice", val_dice, epoch)
                tb_writer.add_scalar("val/iou", float(val_iou), epoch)
                tb_writer.add_scalar("val/precision", float(val_precision), epoch)
            print(
                "Epoch "
                f"{epoch:04d}: train_loss={train_loss:.6f} "
                f"train_fg={train_fg_fraction:.4f} "
                f"train_cls=[{', '.join(f'{v:.4f}' for v in train_class_fractions)}] "
                f"val_loss={val_loss:.6f} val_dice={val_dice:.5f} "
                f"val_iou={val_iou:.5f} val_precision={val_precision:.5f}"
            )
        else:
            val_dice = None
            print(
                f"Epoch {epoch:04d}: train_loss={train_loss:.6f} "
                f"train_fg={train_fg_fraction:.4f} "
                f"train_cls=[{', '.join(f'{v:.4f}' for v in train_class_fractions)}]"
            )

        should_validate = val_loader is not None and (epoch % args.val_interval == 0 or epoch == args.epochs)
        if args.save_random_patch_every_epoch and should_validate and sampled_patches:
            patch_dir = run_dir / "debug_patches"
            patch_dir.mkdir(parents=True, exist_ok=True)
            affine = np.eye(4, dtype=np.float32)
            for i, (patch_image, patch_label) in enumerate(sampled_patches, start=1):
                img_path = patch_dir / f"epoch_{epoch:04d}_patch{i}_image.nii.gz"
                lbl_path = patch_dir / f"epoch_{epoch:04d}_patch{i}_label.nii.gz"
                nib.save(nib.Nifti1Image(patch_image, affine), str(img_path))
                nib.save(nib.Nifti1Image(patch_label, affine), str(lbl_path))

        history.append(row)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "run_dir": str(run_dir),
            "history": row,
        }
        torch.save(ckpt, run_dir / "seg_last.pt")

        if val_dice is not None:
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(ckpt, run_dir / "seg_best.pt")
                write_latest_pointers(output_root, run_dir, ["seg_best.pt", "seg_last.pt"])
        elif epoch == args.epochs:
            torch.save(ckpt, run_dir / "seg_best.pt")
            write_latest_pointers(output_root, run_dir, ["seg_best.pt", "seg_last.pt"])

    (run_dir / "seg_history.json").write_text(json.dumps(history, indent=2) + "\n")
    csv_path = run_dir / "seg_history.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["epoch", "train_loss", "train_fg_fraction"]
        fieldnames += [f"train_class_fraction_c{i}" for i in range(args.num_classes)]
        fieldnames += ["val_loss", "val_dice", "val_iou", "val_precision"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    write_latest_pointers(output_root, run_dir, ["seg_best.pt", "seg_last.pt", "seg_history.json", "seg_history.csv"])
    print(f"Saved checkpoints to {run_dir}")


if __name__ == "__main__":
    main()
