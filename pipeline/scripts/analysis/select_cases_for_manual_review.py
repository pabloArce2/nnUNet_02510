#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select pseudo-labeled cases for manual correction")
    p.add_argument("--ct-nifti-dir", type=Path, required=True)
    p.add_argument("--pseudo-liver-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-cases", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cases = []
    for ct in sorted(args.ct_nifti_dir.glob("*_0000.nii.gz")):
        case_id = ct.name[: -len("_0000.nii.gz")]
        pseudo = args.pseudo_liver_dir / f"{case_id}.nii.gz"
        if pseudo.exists():
            cases.append((case_id, ct, pseudo))

    if not cases:
        raise RuntimeError("No overlapping CT+pseudo cases found")

    n = min(max(args.num_cases, 1), len(cases))
    rnd = random.Random(args.seed)
    selected = rnd.sample(cases, n)

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_dir} exists. Use --overwrite.")
        shutil.rmtree(args.output_dir)

    ct_out = args.output_dir / "ct"
    pseudo_out = args.output_dir / "pseudo_liver"
    corrected_out = args.output_dir / "corrected_liver"
    ct_out.mkdir(parents=True)
    pseudo_out.mkdir(parents=True)
    corrected_out.mkdir(parents=True)

    list_txt = []
    for case_id, ct, pseudo in sorted(selected, key=lambda x: x[0]):
        shutil.copy2(ct, ct_out / ct.name)
        shutil.copy2(pseudo, pseudo_out / pseudo.name)
        list_txt.append(case_id)

    (args.output_dir / "selected_cases.txt").write_text("\n".join(list_txt) + "\n")

    readme = (
        "Manual correction folder\n"
        "- Edit pseudo masks from pseudo_liver/ in your segmentation tool.\n"
        "- Save corrected masks into corrected_liver/ using exactly <case_id>.nii.gz file names.\n"
        "- Use selected_cases.txt when tracking reviewed cases.\n"
    )
    (args.output_dir / "README.txt").write_text(readme)

    print(f"Selected {len(selected)} cases into {args.output_dir}")


if __name__ == "__main__":
    main()
