"""Build a YOLO-seg dataset: grayscale images + multi-class body-part polygons
from DensePose.

Classes (DensePose dp_masks 14-part COARSE scheme; front/back not separable --
that split only exists in the sparse 24-part fine dp_I point labels, not masks):
  0 torso  <- part 1
  1 arm    <- parts 2,3 (hands) + 10,11 (upper arm L/R) + 12,13 (lower arm L/R)
  2 head   <- part 14
Legs/feet (parts 4-9) are dropped: this rig is a SEATED person with hips
occluded below a desk, so legs are never in frame -- training on them would
just be empty classes.

`arm` exists so 06_trunk_axis.py can subtract it from the torso mask: an arm
resting on / crossing the chest used to bleed into the single undifferentiated
"chest" mask and contaminate the torso point cloud (see THINGS_DONE.md, arm
contamination bouts at ~9.5s/14s). Merging both sides into one class doubles
the per-class training signal, since side doesn't matter for exclusion.

Per person instance, emit one polygon per class present (largest contour of
that class's mask) in YOLO-seg format:
  cls x1 y1 x2 y2 ...   (vertices normalized to [0,1])

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
VAL_FRACTION = 0.2
POLY_EPS_FRAC = 0.005  # approxPolyDP epsilon as fraction of contour arc length

CLASS_NAMES = {0: "torso", 1: "arm", 2: "head"}
CLASS_PARTS = {0: {1}, 1: {2, 3, 10, 11, 12, 13}, 2: {14}}
MIN_AREA_PX = {0: 400, 1: 150, 2: 150}  # skip degenerate/tiny masks, per class


def decode_part_masks(dp_masks_entry, bbox, img_w, img_h) -> dict:
    """Decode dp_masks RLEs into one full-image binary mask per class present."""
    x, y, w, h = bbox
    grids = {cid: np.zeros((256, 256), dtype=np.uint8) for cid in CLASS_PARTS}
    for part_idx, rle in enumerate(dp_masks_entry, start=1):
        if not rle:
            continue
        for cid, parts in CLASS_PARTS.items():
            if part_idx in parts:
                grids[cid][mask_utils.decode(rle) > 0] = 1

    out = {}
    for cid, grid in grids.items():
        if not grid.any():
            continue
        resized = cv2.resize(grid, (max(int(w), 1), max(int(h), 1)), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((img_h, img_w), dtype=np.uint8)
        x0, y0 = max(int(x), 0), max(int(y), 0)
        x1, y1 = min(int(x) + resized.shape[1], img_w), min(int(y) + resized.shape[0], img_h)
        full[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
        out[cid] = full
    return out


def part_polygon(mask: np.ndarray, min_area: int):
    """Return the largest contour of `mask` as an (N, 2) pixel polygon, or None."""
    if mask.sum() < min_area:
        return None

    # Bridge small gaps (e.g. an arm crossing the chest splitting the torso mask)
    # so occluded regions still yield one contour covering the full part.
    ys, xs = np.nonzero(mask)
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    kernel_size = max(3, int(span * 0.1)) | 1  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    eps = POLY_EPS_FRAC * cv2.arcLength(contour, closed=True)
    poly = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
    if len(poly) < 3:
        return None
    return poly.astype(np.float32)


def to_yolo_seg_line(poly: np.ndarray, cls: int, img_w: int, img_h: int) -> str:
    """class + flattened normalized polygon vertices."""
    norm = poly.copy()
    norm[:, 0] = np.clip(norm[:, 0] / img_w, 0, 1)
    norm[:, 1] = np.clip(norm[:, 1] / img_h, 0, 1)
    parts = [str(cls)] + [f"{v:.6f}" for v in norm.ravel()]
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
    class_counts = {cid: 0 for cid in CLASS_NAMES}
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
            part_masks = decode_part_masks(a["dp_masks"], a["bbox"], w, h)
            for cid, mask in part_masks.items():
                poly = part_polygon(mask, MIN_AREA_PX[cid])
                if poly is None:
                    continue
                lines.append(to_yolo_seg_line(poly, cid, w, h))
                class_counts[cid] += 1

        if not lines:
            skipped += 1
            continue

        out_img = DATASET_DIR / "images" / split / info["file_name"]
        out_lbl = DATASET_DIR / "labels" / split / (Path(info["file_name"]).stem + ".txt")
        cv2.imwrite(str(out_img), gray_bgr)
        out_lbl.write_text("\n".join(lines))
        kept += 1

    print(f"Built seg dataset: {kept} images, {skipped} skipped (no valid mask)")
    print(f"Instances per class: {[(CLASS_NAMES[c], n) for c, n in class_counts.items()]}")

    names_block = "\n".join(f"  {cid}: {name}" for cid, name in CLASS_NAMES.items())
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(f"""\
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val
names:
{names_block}
""")
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    build()
