#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
)
from tqdm import tqdm

from monai_common import build_unet, datalist_from_case_ids, load_split, pick_device, set_determinism

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-supervised masked reconstruction pretraining for MONAI 3D U-Net")
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--experiment-name",
        type=str,
        default="ssl",
        help="Short name added to timestamped run folder (default: ssl)",
    )
    p.add_argument(
        "--flat-output",
        action="store_true",
        help="Write outputs directly into --output-dir instead of a timestamped subfolder",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96], metavar=("X", "Y", "Z"))
    p.add_argument("--a-min", type=float, default=-200.0)
    p.add_argument("--a-max", type=float, default=300.0)
    p.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Fraction of voxels to hide/corrupt during reconstruction (default: 0.5)",
    )
    p.add_argument(
        "--masking-mode",
        type=str,
        default="block",
        choices=["voxel", "block"],
        help="voxel=random independent voxels, block=contiguous block masking (default: block)",
    )
    p.add_argument(
        "--block-size",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        metavar=("BX", "BY", "BZ"),
        help="Block size for block masking (default: 16 16 16)",
    )
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument(
        "--train-split-key",
        type=str,
        default="ssl_pretrain",
        help="Split key used for SSL training cases (default: ssl_pretrain)",
    )
    p.add_argument(
        "--val-split-key",
        type=str,
        default="val",
        help="Split key used for SSL validation cases (default: val)",
    )
    p.add_argument("--val-interval", type=int, default=1, help="Run SSL validation every N epochs (default: 1)")
    p.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
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


def global_ssim_torch(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return num / torch.clamp(den, min=1e-8)


def build_mask(x: torch.Tensor, mask_ratio: float, masking_mode: str, block_size: tuple[int, int, int]) -> torch.Tensor:
    if masking_mode == "voxel":
        return (torch.rand_like(x) < mask_ratio).float()

    b, c, sx, sy, sz = x.shape
    bx, by, bz = block_size
    bx = max(1, min(bx, sx))
    by = max(1, min(by, sy))
    bz = max(1, min(bz, sz))
    mask = torch.zeros_like(x)
    target_vox = int(math.ceil(mask_ratio * sx * sy * sz))
    block_vox = bx * by * bz
    blocks_per_sample = max(1, int(math.ceil(target_vox / max(block_vox, 1))))
    for bi in range(b):
        for _ in range(blocks_per_sample):
            x0 = int(torch.randint(0, sx - bx + 1, (1,), device=x.device).item())
            y0 = int(torch.randint(0, sy - by + 1, (1,), device=x.device).item())
            z0 = int(torch.randint(0, sz - bz + 1, (1,), device=x.device).item())
            mask[bi, :, x0 : x0 + bx, y0 : y0 + by, z0 : z0 + bz] = 1.0
    return mask


def main() -> None:
    args = parse_args()
    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError("--mask-ratio must be between 0 and 1")
    if args.val_interval < 1:
        raise ValueError("--val-interval must be >= 1")
    if any(v < 1 for v in args.block_size):
        raise ValueError("--block-size values must be >= 1")

    set_determinism(args.seed)
    device = pick_device(args.device)
    split_payload = load_split(args.split_json)

    case_ids = split_payload["splits"].get(args.train_split_key, [])
    if not case_ids:
        raise RuntimeError(f"Split file has no cases under splits.{args.train_split_key}")

    val_ids = split_payload["splits"].get(args.val_split_key, [])
    if val_ids:
        train_id_set = set(case_ids)
        val_id_set = set(val_ids)
        overlap = train_id_set.intersection(val_id_set)
        if overlap:
            case_ids = sorted([cid for cid in case_ids if cid not in val_id_set])
            print(
                f"Removed {len(overlap)} overlapping case(s) from SSL train split to avoid leakage "
                f"with validation split '{args.val_split_key}'."
            )
        if not case_ids:
            raise RuntimeError("No SSL training cases left after removing overlap with validation split")

    train_datalist = datalist_from_case_ids(split_payload, case_ids, require_label=False)
    val_datalist = datalist_from_case_ids(split_payload, val_ids, require_label=False) if val_ids else []

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(args.a_min),
                a_max=float(args.a_max),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            RandSpatialCropd(keys=["image"], roi_size=tuple(args.patch_size), random_size=False),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.15, sigma_x=(0.25, 1.0), sigma_y=(0.25, 1.0), sigma_z=(0.25, 1.0)),
            EnsureTyped(keys=["image"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=float(args.a_min),
                a_max=float(args.a_max),
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CenterSpatialCropd(keys=["image"], roi_size=tuple(args.patch_size)),
            EnsureTyped(keys=["image"]),
        ]
    )

    train_ds = Dataset(data=train_datalist, transform=train_transforms)
    val_ds = Dataset(data=val_datalist, transform=val_transforms) if val_datalist else None

    loader = DataLoader(
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

    model = build_unet(out_channels=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    block_size = tuple(int(v) for v in args.block_size)

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root if args.flat_output else make_run_dir(output_root, args.experiment_name)
    print(f"Run output directory: {run_dir}")

    history = []
    best_loss = float("inf")
    best_path = run_dir / "ssl_best.pt"
    last_path = run_dir / "ssl_last.pt"
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
        tb_writer.add_text("run/ssl_train_split", args.train_split_key)
        tb_writer.add_text("run/ssl_val_split", args.val_split_key)
        tb_writer.add_scalar("config/mask_ratio", float(args.mask_ratio), 0)
        tb_writer.add_text("config/masking_mode", args.masking_mode)
        tb_writer.add_text("config/block_size", str(block_size))
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
        epoch_loss = 0.0
        epoch_mask_ratio = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"SSL epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["image"].to(device)
            noise = torch.randn_like(x)
            mask = build_mask(x, args.mask_ratio, args.masking_mode, block_size)
            x_corrupt = x * (1.0 - mask) + noise * mask

            pred = model(x_corrupt)
            denom = torch.clamp(mask.sum(), min=1.0)
            loss = torch.sum(((pred - x) ** 2) * mask) / denom

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_item = float(loss.detach().cpu().item())
            mask_fraction = float(mask.detach().mean().cpu().item())
            epoch_loss += loss_item
            epoch_mask_ratio += mask_fraction
            steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss_item:.5f}")

            if tb_writer is not None:
                tb_writer.add_scalar("train/step_ssl_loss", loss_item, global_step)
                tb_writer.add_scalar("train/step_mask_fraction", mask_fraction, global_step)
                tb_writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

        mean_loss = epoch_loss / max(steps, 1)
        mean_mask_ratio = epoch_mask_ratio / max(steps, 1)
        epoch_seconds = time.time() - epoch_start
        steps_per_sec = steps / max(epoch_seconds, 1e-8)
        row = {"epoch": epoch, "loss": mean_loss}

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch_ssl_loss", mean_loss, epoch)
            tb_writer.add_scalar("train/epoch_mask_fraction", mean_mask_ratio, epoch)
            tb_writer.add_scalar("train/epoch_seconds", epoch_seconds, epoch)
            tb_writer.add_scalar("train/steps_per_second", steps_per_sec, epoch)

        val_loss = None
        if val_loader is not None and (epoch % args.val_interval == 0 or epoch == args.epochs):
            model.eval()
            val_running = 0.0
            val_full_mse_running = 0.0
            val_full_mae_running = 0.0
            val_psnr_running = 0.0
            val_ssim_running = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["image"].to(device)
                    noise = torch.randn_like(x)
                    mask = build_mask(x, args.mask_ratio, args.masking_mode, block_size)
                    x_corrupt = x * (1.0 - mask) + noise * mask

                    pred = model(x_corrupt)
                    denom = torch.clamp(mask.sum(), min=1.0)
                    loss = torch.sum(((pred - x) ** 2) * mask) / denom
                    full_mse = ((pred - x) ** 2).mean()
                    full_mae = torch.abs(pred - x).mean()
                    psnr = 10.0 * torch.log10(torch.tensor(1.0, device=device) / torch.clamp(full_mse, min=1e-8))
                    ssim = global_ssim_torch(pred, x, data_range=1.0)

                    val_running += float(loss.detach().cpu().item())
                    val_full_mse_running += float(full_mse.detach().cpu().item())
                    val_full_mae_running += float(full_mae.detach().cpu().item())
                    val_psnr_running += float(psnr.detach().cpu().item())
                    val_ssim_running += float(ssim.detach().cpu().item())
                    val_steps += 1

            val_loss = val_running / max(val_steps, 1)
            val_full_mse = val_full_mse_running / max(val_steps, 1)
            val_full_mae = val_full_mae_running / max(val_steps, 1)
            val_psnr = val_psnr_running / max(val_steps, 1)
            val_ssim = val_ssim_running / max(val_steps, 1)
            row["val_loss"] = val_loss
            row["val_full_mse"] = val_full_mse
            row["val_full_mae"] = val_full_mae
            row["val_psnr_db"] = val_psnr
            row["val_global_ssim"] = val_ssim

            if tb_writer is not None:
                tb_writer.add_scalar("val/epoch_ssl_loss", val_loss, epoch)
                tb_writer.add_scalar("val/full_mse", val_full_mse, epoch)
                tb_writer.add_scalar("val/full_mae", val_full_mae, epoch)
                tb_writer.add_scalar("val/psnr_db", val_psnr, epoch)
                tb_writer.add_scalar("val/global_ssim", val_ssim, epoch)

            print(
                f"Epoch {epoch:04d}: ssl_loss={mean_loss:.6f} val_ssl_loss={val_loss:.6f} "
                f"val_full_mse={val_full_mse:.6f} val_full_mae={val_full_mae:.6f} "
                f"val_psnr={val_psnr:.3f} val_ssim={val_ssim:.4f}"
            )
        else:
            print(f"Epoch {epoch:04d}: ssl_loss={mean_loss:.6f}")

        history.append(row)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "run_dir": str(run_dir),
            "mean_loss": mean_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, last_path)

        score_for_best = val_loss if val_loss is not None else mean_loss
        if score_for_best < best_loss:
            best_loss = score_for_best
            torch.save(ckpt, best_path)
            write_latest_pointers(output_root, run_dir, ["ssl_best.pt", "ssl_last.pt"])

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(ckpt, run_dir / f"ssl_epoch_{epoch:04d}.pt")

    (run_dir / "ssl_history.json").write_text(json.dumps(history, indent=2) + "\n")
    write_latest_pointers(output_root, run_dir, ["ssl_best.pt", "ssl_last.pt", "ssl_history.json"])
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    print(f"Saved best SSL checkpoint: {best_path}")


if __name__ == "__main__":
    main()
