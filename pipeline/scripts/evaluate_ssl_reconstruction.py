#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, ScaleIntensityRanged
from tqdm import tqdm

from monai_common import build_unet, datalist_from_case_ids, load_split, pick_device, set_determinism


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SSL reconstruction quality on a split")
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="SSL checkpoint (ssl_best.pt or ssl_last.pt)")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--summary-json", type=Path, default=None, help="Optional summary metrics JSON path")
    p.add_argument("--split-name", type=str, default="test", help="Split key to evaluate (default: test)")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96], metavar=("X", "Y", "Z"))
    p.add_argument("--sw-overlap", type=float, default=0.25, help="Sliding-window overlap (default: 0.25)")
    p.add_argument("--mask-ratio", type=float, default=0.35, help="Mask ratio for corruption (default: 0.35)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--a-min", type=float, default=-200.0)
    p.add_argument("--a-max", type=float, default=300.0)
    p.add_argument("--save-dir", type=Path, default=None, help="Optional folder to save recon/corrupt/error NIfTI files")
    return p.parse_args()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((data_range * data_range) / mse)


def global_ssim(x: np.ndarray, y: np.ndarray, data_range: float = 1.0) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(x.mean())
    mu_y = float(y.mean())
    var_x = float(((x - mu_x) ** 2).mean())
    var_y = float(((y - mu_y) ** 2).mean())
    cov_xy = float(((x - mu_x) * (y - mu_y)).mean())
    num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return safe_div(num, den)


def main() -> None:
    args = parse_args()
    if not (0.0 < args.mask_ratio < 1.0):
        raise ValueError("--mask-ratio must be between 0 and 1")
    if not (0.0 <= args.sw_overlap < 1.0):
        raise ValueError("--sw-overlap must be in [0, 1)")

    set_determinism(args.seed)
    device = pick_device(args.device)
    payload = load_split(args.split_json)
    case_ids = payload["splits"].get(args.split_name, [])
    if not case_ids:
        raise RuntimeError(f"No case ids in split '{args.split_name}'")

    eval_data = datalist_from_case_ids(payload, case_ids, require_label=False)
    transforms = Compose(
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
            EnsureTyped(keys=["image"]),
        ]
    )
    ds = Dataset(data=eval_data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    image_path_by_case = {row["case_id"]: Path(row["image"]) for row in eval_data}

    model = build_unet(out_channels=1).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    save_root = args.save_dir
    if save_root is not None:
        for sub in ("input", "corrupted", "recon", "error", "mask"):
            (save_root / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"SSL eval ({args.split_name})", leave=False):
            case_id = batch["case_id"][0]
            image_path = image_path_by_case[case_id]
            ref = nib.load(str(image_path))

            x = batch["image"].to(device)
            noise = torch.randn_like(x)
            mask = (torch.rand_like(x) < args.mask_ratio).float()
            x_corrupt = x * (1.0 - mask) + noise * mask

            pred = sliding_window_inference(
                x_corrupt,
                roi_size=tuple(args.patch_size),
                sw_batch_size=1,
                predictor=model,
                overlap=float(args.sw_overlap),
            )

            sq_err = (pred - x) ** 2
            abs_err = torch.abs(pred - x)
            mask_sum = float(mask.sum().detach().cpu().item())
            full_voxels = float(torch.numel(x))
            masked_mse = float((sq_err * mask).sum().detach().cpu().item() / max(mask_sum, 1.0))
            masked_mae = float((abs_err * mask).sum().detach().cpu().item() / max(mask_sum, 1.0))
            full_mse = float(sq_err.mean().detach().cpu().item())
            full_mae = float(abs_err.mean().detach().cpu().item())
            psnr = psnr_from_mse(full_mse, data_range=1.0)

            x_np = x[0, 0].detach().cpu().numpy()
            pred_np = pred[0, 0].detach().cpu().numpy()
            gssim = global_ssim(x_np, pred_np, data_range=1.0)

            rows.append(
                {
                    "case_id": case_id,
                    "split": args.split_name,
                    "mask_ratio_target": f"{args.mask_ratio:.6f}",
                    "mask_ratio_realized": f"{(mask_sum / max(full_voxels, 1.0)):.6f}",
                    "masked_mse": f"{masked_mse:.8f}",
                    "masked_mae": f"{masked_mae:.8f}",
                    "full_mse": f"{full_mse:.8f}",
                    "full_mae": f"{full_mae:.8f}",
                    "psnr_db": f"{psnr:.6f}",
                    "global_ssim": f"{gssim:.6f}",
                }
            )

            if save_root is not None:
                corrupt_np = x_corrupt[0, 0].detach().cpu().numpy().astype(np.float32)
                err_np = np.abs(pred_np - x_np).astype(np.float32)
                mask_np = mask[0, 0].detach().cpu().numpy().astype(np.float32)
                nib.save(nib.Nifti1Image(x_np.astype(np.float32), ref.affine, ref.header), str(save_root / "input" / f"{case_id}.nii.gz"))
                nib.save(
                    nib.Nifti1Image(corrupt_np, ref.affine, ref.header),
                    str(save_root / "corrupted" / f"{case_id}.nii.gz"),
                )
                nib.save(nib.Nifti1Image(pred_np.astype(np.float32), ref.affine, ref.header), str(save_root / "recon" / f"{case_id}.nii.gz"))
                nib.save(nib.Nifti1Image(err_np, ref.affine, ref.header), str(save_root / "error" / f"{case_id}.nii.gz"))
                nib.save(nib.Nifti1Image(mask_np, ref.affine, ref.header), str(save_root / "mask" / f"{case_id}.nii.gz"))

    if not rows:
        raise RuntimeError("No cases processed")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_keys = ["masked_mse", "masked_mae", "full_mse", "full_mae", "psnr_db", "global_ssim"]
    means = {k: float(np.mean([float(r[k]) for r in rows])) for k in metric_keys}
    summary = {
        "split_name": args.split_name,
        "num_cases": len(rows),
        "checkpoint": str(args.checkpoint),
        "mask_ratio": float(args.mask_ratio),
        "means": means,
    }

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Saved per-case metrics: {args.output_csv}")
    if args.summary_json is not None:
        print(f"Saved summary metrics: {args.summary_json}")
    print("Mean metrics:")
    for key in metric_keys:
        print(f"  {key}: {means[key]:.8f}")


if __name__ == "__main__":
    main()
