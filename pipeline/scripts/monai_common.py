#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


def case_id_from_ct_name(name: str) -> str:
    if not name.endswith("_0000.nii.gz"):
        raise ValueError(f"Unexpected CT filename format: {name}")
    return name[: -len("_0000.nii.gz")]


def case_id_from_mask_name(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return Path(name).stem


def scan_ct_cases(ct_dir: Path) -> Dict[str, str]:
    cases: Dict[str, str] = {}
    for path in sorted(ct_dir.rglob("*_0000.nii.gz")):
        case_id = case_id_from_ct_name(path.name)
        if case_id in cases:
            raise ValueError(f"Duplicate CT case id '{case_id}' found: {cases[case_id]} and {path.resolve()}")
        cases[case_id] = str(path.resolve())
    return cases


def scan_label_cases(label_dir: Optional[Path]) -> Dict[str, str]:
    if label_dir is None or not label_dir.exists():
        return {}
    cases: Dict[str, str] = {}
    for path in sorted(label_dir.rglob("*.nii.gz")):
        case_id = case_id_from_mask_name(path.name)
        if case_id in cases:
            raise ValueError(f"Duplicate label case id '{case_id}' found: {cases[case_id]} and {path.resolve()}")
        cases[case_id] = str(path.resolve())
    return cases


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_unet(out_channels: int) -> torch.nn.Module:
    from monai.networks.nets import UNet

    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def pick_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False")
    if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is False")
    return torch.device(req)


def load_split(split_json: Path) -> dict:
    payload = read_json(split_json)
    if "splits" not in payload:
        raise ValueError(f"Invalid split JSON: missing 'splits' in {split_json}")
    return payload


def make_case_lookup(split_payload: dict) -> Dict[str, dict]:
    lookup = {}
    for item in split_payload.get("cases", []):
        lookup[item["case_id"]] = item
    return lookup


def datalist_from_case_ids(split_payload: dict, case_ids: List[str], require_label: bool) -> List[dict]:
    lookup = make_case_lookup(split_payload)
    items: List[dict] = []
    for case_id in case_ids:
        case = lookup.get(case_id)
        if case is None:
            raise KeyError(f"Case id {case_id} missing from split JSON")
        row = {"case_id": case_id, "image": case["image"]}
        label = case.get("label")
        if require_label:
            if not label:
                raise ValueError(f"Case {case_id} does not have a label")
            row["label"] = label
        elif label:
            row["label"] = label
        items.append(row)
    return items
