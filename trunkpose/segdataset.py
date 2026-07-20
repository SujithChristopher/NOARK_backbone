"""Build a YOLO-seg dataset: grayscale images + torso polygons from DensePose.

Torso = DensePose dp_masks part index 1. NOTE: dp_masks is the 14-part COARSE
body segmentation (1=Torso, 2=RHand, 3=LHand, ... 14=Head), which has a single
combined torso mask -- front and back are NOT separable here (that split only
exists in the 24-part fine dp_I point labels, which are sparse points, not masks).
So "chest" here means the whole trunk region.

Per person instance we emit one polygon (largest contour of the torso mask) in
YOLO-seg format:
  0 x1 y1 x2 y2 ...   (class 0 = chest, vertices normalized to [0,1])

Mirror of build_pose_dataset.py, but outputs segmentation polygons instead of
5 trunk keypoints, into a separate dataset_seg/ dir (pose dataset untouched).
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"
ANNOT_PATH = DATA_DIR / "densepose_coco_2014_minival.json"

DATASET_DIR = ROOT / "dataset_seg"
MIN_CHEST_AREA_PX = 400  # skip degenerate/tiny masks
VAL_FRACTION = 0.2
CHEST_PARTS = {1}  # dp_masks 14-part scheme: index 1 = Torso (index 2 = Right Hand!)
POLY_EPS_FRAC = 0.005  # approxPolyDP epsilon as fraction of contour arc length


def decode_chest_mask(dp_masks_entry, bbox, img_w, img_h) -> np.ndarray:
    """Decode the torso (part 1) RLE grid into a full-image-size binary mask."""
    x, y, w, h = bbox
    grid = np.zeros((256, 256), dtype=np.uint8)
    for part_idx, rle in enumerate(dp_masks_entry, start=1):
        if rle and part_idx in CHEST_PARTS:
            grid[mask_utils.decode(rle) > 0] = 1

    resized = cv2.resize(grid, (max(int(w), 1), max(int(h), 1)), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((img_h, img_w), dtype=np.uint8)
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x) + resized.shape[1], img_w), min(int(y) + resized.shape[0], img_h)
    full[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return full


def chest_polygon(chest_mask: np.ndarray):
    """Return the largest chest contour as an (N, 2) pixel polygon, or None if too small."""
    if chest_mask.sum() < MIN_CHEST_AREA_PX:
        return None

    # Bridge small gaps (e.g. an arm crossing the chest splitting the mask)
    # so occluded chests still yield one contour covering the full region.
    ys, xs = np.nonzero(chest_mask)
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    kernel_size = max(3, int(span * 0.1)) | 1  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(chest_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CHEST_AREA_PX:
        return None

    eps = POLY_EPS_FRAC * cv2.arcLength(contour, closed=True)
    poly = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
    if len(poly) < 3:
        return None
    return poly.astype(np.float32)


def to_yolo_seg_line(poly: np.ndarray, img_w: int, img_h: int) -> str:
    """class + flattened normalized polygon vertices."""
    norm = poly.copy()
    norm[:, 0] = np.clip(norm[:, 0] / img_w, 0, 1)
    norm[:, 1] = np.clip(norm[:, 1] / img_h, 0, 1)
    parts = ["0"] + [f"{v:.6f}" for v in norm.ravel()]
    return " ".join(parts)


def build():
    coco = COCO(str(ANNOT_PATH))
    img_ids = sorted({int(p.stem.split("_")[-1]) for p in IMG_DIR.glob("*.jpg")})
    random.seed(0)
    random.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * VAL_FRACTION))
    splits = {"val": set(img_ids[:n_val]), "train": set(img_ids[n_val:])}

    for split in splits:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    for img_id in img_ids:
        split = "val" if img_id in splits["val"] else "train"
        info = coco.loadImgs(img_id)[0]
        img_path = IMG_DIR / info["file_name"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = [a for a in coco.loadAnns(ann_ids) if "dp_masks" in a]

        lines = []
        for a in anns:
            chest_mask = decode_chest_mask(a["dp_masks"], a["bbox"], w, h)
            poly = chest_polygon(chest_mask)
            if poly is None:
                continue
            lines.append(to_yolo_seg_line(poly, w, h))

        if not lines:
            skipped += 1
            continue

        out_img = DATASET_DIR / "images" / split / info["file_name"]
        out_lbl = DATASET_DIR / "labels" / split / (Path(info["file_name"]).stem + ".txt")
        cv2.imwrite(str(out_img), gray_bgr)
        out_lbl.write_text("\n".join(lines))
        kept += 1

    print(f"Built seg dataset: {kept} images with chest polygons, {skipped} skipped (no valid mask)")

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(f"""\
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val
names:
  0: chest
""")
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    build()
