"""Train a YOLO-seg model on the chest-segmentation dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--fraction", type=float, default=1.0, help="fraction of train set to use (0-1]")
    parser.add_argument("--seed", type=int, default=9)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(ROOT / "dataset_seg" / "data.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        fraction=args.fraction,
        seed=args.seed,
        project=str(ROOT / "runs"),
        name="trunk_seg",
    )


if __name__ == "__main__":
    main()
