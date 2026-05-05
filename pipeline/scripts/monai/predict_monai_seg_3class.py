#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    ScaleIntensityRanged,
)
from tqdm import tqdm

from monai_common import build_unet, datalist_from_case_ids, load_split, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MONAI 3-class segmentation inference")
    p.add_argument("--split-json", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="seg_best.pt or seg_last.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--split-name",
        type=str,
        default="test",
        choices=["train", "val", "test", "ssl_pretrain", "labeled", "unlabeled_only"],
    )
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96], metavar=("X", "Y", "Z"))
    p.add_argument("--a-min", type=float, default=-200.0)
    p.add_argument("--a-max", type=float, default=300.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-probability", action="store_true", help="Also save per-class probability maps")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    payload = load_split(args.split_json)

    case_ids = payload["splits"].get(args.split_name, [])
    if not case_ids:
        raise RuntimeError(f"No case ids in split '{args.split_name}'")

    infer_data = datalist_from_case_ids(payload, case_ids, require_label=False)

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=0.0, b_max=1.0, clip=True),
            EnsureTyped(keys=["image"]),
        ]
    )

    ds = Dataset(data=infer_data, transform=transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    model = build_unet(out_channels=3).to(device)
    # Checkpoints saved by this pipeline may contain metadata objects beyond tensors.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pred_dir = args.output_dir / "pred_masks"
    prob_dir = args.output_dir / "pred_probs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if args.save_probability:
        prob_dir.mkdir(parents=True, exist_ok=True)

    # Network class index -> saved label id mapping:
    # class 0 -> liver label id 0
    # class 1 -> pig label id 1
    # class 2 -> background label id 2
    output_label_ids = np.array([0, 1, 2], dtype=np.uint8)

    image_path_by_case = {row["case_id"]: Path(row["image"]) for row in infer_data}

    for batch in tqdm(loader, desc="Predict", leave=False):
        case_id = batch["case_id"][0]
        image_path = image_path_by_case[case_id]
        ref = nib.load(str(image_path))

        x = batch["image"].to(device)
        with torch.no_grad():
            logits = sliding_window_inference(x, tuple(args.patch_size), sw_batch_size=1, predictor=model)
            pred_class = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)

        pred_cls_np = pred_class[0].detach().cpu().numpy().astype(np.uint8)
        pred_np = output_label_ids[pred_cls_np]
        nib.save(nib.Nifti1Image(pred_np, ref.affine, ref.header), str(pred_dir / f"{case_id}.nii.gz"))

        if args.save_probability:
            prob_liver = probs[0, 0].detach().cpu().numpy().astype(np.float32)
            prob_pig = probs[0, 1].detach().cpu().numpy().astype(np.float32)
            prob_background = probs[0, 2].detach().cpu().numpy().astype(np.float32)
            nib.save(nib.Nifti1Image(prob_liver, ref.affine, ref.header), str(prob_dir / f"{case_id}_class0_liver.nii.gz"))
            nib.save(nib.Nifti1Image(prob_pig, ref.affine, ref.header), str(prob_dir / f"{case_id}_class1_pig.nii.gz"))
            nib.save(
                nib.Nifti1Image(prob_background, ref.affine, ref.header),
                str(prob_dir / f"{case_id}_class2_background.nii.gz"),
            )

    print(f"Saved predictions to: {pred_dir}")
    if args.save_probability:
        print(f"Saved probabilities to: {prob_dir}")


if __name__ == "__main__":
    main()
