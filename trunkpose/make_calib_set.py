"""Sample a small labelled subset of dataset_seg for INT8 / post-training calibration on export.

Usage:
    python make_calib_set.py                 # 50 images from val -> dataset_calib/
    yolo export model=runs/trunk_seg/weights/best.pt format=tflite int8=True data=dataset_calib/data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(ROOT / "dataset_seg"))
    parser.add_argument("--dst", default=str(ROOT / "dataset_calib"))
    parser.add_argument("--split", default="val", help="source split to sample from")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    src_cfg = yaml.safe_load((src / "data.yaml").read_text())

    img_dir, lbl_dir = src / "images" / args.split, src / "labels" / args.split
    # only keep images that have a non-empty label file
    candidates = sorted(
        p
        for p in img_dir.iterdir()
        if (lbl_dir / f"{p.stem}.txt").is_file()
        and (lbl_dir / f"{p.stem}.txt").stat().st_size > 0
    )
    if len(candidates) < args.n:
        raise SystemExit(
            f"only {len(candidates)} labelled images in {img_dir}, need {args.n}"
        )
    picked = random.Random(args.seed).sample(candidates, args.n)

    if dst.exists():
        shutil.rmtree(dst)
    out_img, out_lbl = dst / "images" / "calib", dst / "labels" / "calib"
    out_img.mkdir(parents=True)
    out_lbl.mkdir(parents=True)
    for p in picked:
        shutil.copy2(p, out_img / p.name)
        shutil.copy2(lbl_dir / f"{p.stem}.txt", out_lbl / f"{p.stem}.txt")

    # exporters read the val split for calibration; point train too so either works
    cfg = {
        "path": str(dst.resolve()),
        "train": "images/calib",
        "val": "images/calib",
        "names": src_cfg["names"],
    }
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {len(picked)} images + labels to {dst}")


if __name__ == "__main__":
    main()
