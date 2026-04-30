#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandGaussianNoised,
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
    p = argparse.ArgumentParser(description="Train MONAI 3D U-Net for liver segmentation")
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
    p.add_argument("--a-min", type=float, default=-200.0)
    p.add_argument("--a-max", type=float, default=300.0)
    p.add_argument(
        "--init-ssl-checkpoint",
        type=Path,
        default=None,
        help="Optional SSL checkpoint from train_monai_ssl.py for encoder initialization",
    )
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


def maybe_load_ssl_weights(model: torch.nn.Module, ckpt_path: Path) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    current = model.state_dict()
    matched = {}
    for key, tensor in state.items():
        if key in current and tuple(current[key].shape) == tuple(tensor.shape):
            matched[key] = tensor
    current.update(matched)
    model.load_state_dict(current)
    return len(matched)


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = pick_device(args.device)

    split_payload = load_split(args.split_json)
    train_ids = split_payload["splits"].get("train", [])
    val_ids = split_payload["splits"].get("val", [])

    if not train_ids:
        raise RuntimeError(
            "No labeled training cases in split JSON. Add labels first, rerun create_monai_splits.py, then train."
        )

    train_data = datalist_from_case_ids(split_payload, train_ids, require_label=True)
    val_data = datalist_from_case_ids(split_payload, val_ids, require_label=True) if val_ids else []

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
            RandSpatialCropd(keys=["image", "label"], roi_size=tuple(args.patch_size), random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
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

    model = build_unet(out_channels=2).to(device)
    if args.init_ssl_checkpoint:
        matched = maybe_load_ssl_weights(model, args.init_ssl_checkpoint)
        print(f"Loaded {matched} matching parameters from SSL checkpoint")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root if args.flat_output else make_run_dir(output_root, args.experiment_name)
    print(f"Run output directory: {run_dir}")

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
            running += loss_item
            steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss_item:.5f}")

            if tb_writer is not None:
                tb_writer.add_scalar("train/step_loss", loss_item, global_step)
                tb_writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)

        train_loss = running / max(steps, 1)
        epoch_seconds = time.time() - epoch_start
        steps_per_sec = steps / max(epoch_seconds, 1e-8)
        row = {"epoch": epoch, "train_loss": train_loss}

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch_loss", train_loss, epoch)
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
                f"val_loss={val_loss:.6f} val_dice={val_dice:.5f} "
                f"val_iou={val_iou:.5f} val_precision={val_precision:.5f}"
            )
        else:
            val_dice = None
            print(f"Epoch {epoch:04d}: train_loss={train_loss:.6f}")

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
        fieldnames = ["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_precision"]
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
