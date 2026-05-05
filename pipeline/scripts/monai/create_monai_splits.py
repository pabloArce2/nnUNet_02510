#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import random
from pathlib import Path
from typing import List, Set

from monai_common import scan_ct_cases, scan_label_cases, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build split metadata for MONAI SSL + segmentation experiments. "
            "Works even when no labels exist yet."
        )
    )
    p.add_argument("--ct-dir", type=Path, required=True, help="Directory with *_0000.nii.gz CT volumes")
    p.add_argument("--label-dir", type=Path, default=None, help="Optional weak label dir (<case_id>.nii.gz)")
    p.add_argument(
        "--corrected-label-dir",
        type=Path,
        default=None,
        help="Optional corrected label dir; overrides weak labels when case ids overlap",
    )
    p.add_argument(
        "--fixed-test-cases",
        type=Path,
        default=None,
        help="Optional text file with one held-out case_id per line",
    )
    p.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction over labeled non-test cases")
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Test fraction over labeled cases when --fixed-test-cases is not provided",
    )
    p.add_argument(
        "--split-mode",
        type=str,
        default="stable_hash",
        choices=["stable_hash", "seeded_random"],
        help=(
            "stable_hash keeps case assignment stable when new cases are added. "
            "seeded_random reproduces the previous behavior."
        ),
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def read_case_list(path: Path) -> Set[str]:
    out: Set[str] = set()
    for line in path.read_text().splitlines():
        case = line.strip()
        if case and not case.startswith("#"):
            out.add(case)
    return out


def split_train_val(cases: List[str], val_fraction: float, seed: int) -> tuple[List[str], List[str]]:
    if not cases:
        return [], []
    if len(cases) == 1:
        return cases[:], []

    rnd = random.Random(seed)
    shuffled = cases[:]
    rnd.shuffle(shuffled)

    val_n = max(1, int(round(len(shuffled) * val_fraction)))
    val_n = min(val_n, len(shuffled) - 1)

    val = sorted(shuffled[:val_n])
    train = sorted(shuffled[val_n:])
    return train, val


def stable_score(case_id: str, seed: int, tag: str) -> float:
    text = f"{seed}:{tag}:{case_id}"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    value = int(digest, 16)
    return value / float(2**160 - 1)


def pick_by_fraction_stable(cases: List[str], fraction: float, seed: int, tag: str) -> List[str]:
    if not cases:
        return []
    if len(cases) == 1 or fraction <= 0.0:
        return []
    scored = sorted((stable_score(c, seed, tag), c) for c in cases)
    selected = [c for s, c in scored if s < fraction]
    if not selected:
        selected = [scored[0][1]]
    if len(selected) >= len(cases):
        selected = [c for _, c in scored[:-1]]
    return sorted(selected)


def main() -> None:
    args = parse_args()

    ct_cases = scan_ct_cases(args.ct_dir)
    if not ct_cases:
        raise FileNotFoundError(f"No *_0000.nii.gz files found in {args.ct_dir}")

    weak = scan_label_cases(args.label_dir)
    corrected = scan_label_cases(args.corrected_label_dir)

    all_case_ids = sorted(ct_cases.keys())
    labeled_case_ids = sorted([cid for cid in all_case_ids if cid in weak or cid in corrected])

    fixed_test: Set[str] = set()
    if args.fixed_test_cases is not None:
        fixed_test = read_case_list(args.fixed_test_cases)
        unknown = sorted(fixed_test - set(all_case_ids))
        if unknown:
            raise ValueError(f"Unknown case ids in --fixed-test-cases: {unknown}")

    if fixed_test:
        test_ids = sorted([cid for cid in labeled_case_ids if cid in fixed_test])
    else:
        if args.split_mode == "stable_hash":
            test_ids = pick_by_fraction_stable(
                labeled_case_ids,
                fraction=args.test_fraction,
                seed=args.seed,
                tag="test",
            )
        else:
            if len(labeled_case_ids) <= 1:
                test_ids = []
            else:
                rnd = random.Random(args.seed)
                shuffled = labeled_case_ids[:]
                rnd.shuffle(shuffled)
                n_test = max(1, int(round(len(shuffled) * args.test_fraction)))
                n_test = min(n_test, len(shuffled) - 1)
                test_ids = sorted(shuffled[:n_test])

    train_pool = sorted([cid for cid in labeled_case_ids if cid not in set(test_ids)])
    if args.split_mode == "stable_hash":
        val_ids = pick_by_fraction_stable(
            train_pool,
            fraction=args.val_fraction,
            seed=args.seed,
            tag="val",
        )
        train_ids = sorted([cid for cid in train_pool if cid not in set(val_ids)])
    else:
        train_ids, val_ids = split_train_val(train_pool, args.val_fraction, args.seed)

    unlabeled_only = sorted([cid for cid in all_case_ids if cid not in set(labeled_case_ids)])

    case_rows = []
    for cid in all_case_ids:
        label_path = None
        label_source = None
        if cid in corrected:
            label_path = corrected[cid]
            label_source = "corrected"
        elif cid in weak:
            label_path = weak[cid]
            label_source = "weak"

        case_rows.append(
            {
                "case_id": cid,
                "image": ct_cases[cid],
                "label": label_path,
                "label_source": label_source,
            }
        )

    payload = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ct_dir": str(args.ct_dir.resolve()),
        "label_dir": str(args.label_dir.resolve()) if args.label_dir else None,
        "corrected_label_dir": str(args.corrected_label_dir.resolve()) if args.corrected_label_dir else None,
        "splits": {
            "ssl_pretrain": all_case_ids,
            "labeled": labeled_case_ids,
            "unlabeled_only": unlabeled_only,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "fixed_test_all": sorted(fixed_test),
            "split_mode": args.split_mode,
        },
        "cases": case_rows,
        "counts": {
            "all_cases": len(all_case_ids),
            "labeled_cases": len(labeled_case_ids),
            "unlabeled_only": len(unlabeled_only),
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
    }

    write_json(args.output_json, payload)

    print(f"Wrote split file: {args.output_json}")
    print(f"Cases total/labeled/unlabeled: {len(all_case_ids)}/{len(labeled_case_ids)}/{len(unlabeled_only)}")
    print(f"Supervised split train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")


if __name__ == "__main__":
    main()
