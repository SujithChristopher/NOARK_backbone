"""Build a YOLO-pose dataset: grayscale images + trunk keypoints from DensePose torso masks.

Trunk keypoints per person instance (5 pts), derived from the combined torso
mask (DensePose part indices 1,2 = torso back/front):
  0 centroid, 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left
(corners from cv2.minAreaRect over the torso mask contour, consistently ordered).
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
ANNOT_PATH = DATA_DIR / "densepose_coco_2014_train.json"

DATASET_DIR = ROOT / "dataset"
MIN_TORSO_AREA_PX = 400  # skip degenerate/tiny torso masks
VAL_FRACTION = 0.2
TORSO_PARTS = {1, 2}


def decode_torso_mask(dp_masks_entry, dp_i, bbox, img_w, img_h) -> np.ndarray:
    """Decode combined torso (parts 1,2) RLE grid into a full-image-size binary mask."""
    x, y, w, h = bbox
    grid = np.zeros((256, 256), dtype=np.uint8)
    for part_idx, rle in enumerate(dp_masks_entry, start=1):
        if rle and part_idx in TORSO_PARTS:
            grid[mask_utils.decode(rle) > 0] = 1

    resized = cv2.resize(grid, (max(int(w), 1), max(int(h), 1)), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((img_h, img_w), dtype=np.uint8)
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1, y1 = min(int(x) + resized.shape[1], img_w), min(int(y) + resized.shape[0], img_h)
    full[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return full


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def trunk_keypoints(torso_mask: np.ndarray):
    """Return (centroid, tl, tr, br, bl) in pixel coords, or None if mask too small."""
    if torso_mask.sum() < MIN_TORSO_AREA_PX:
        return None

    # Bridge small gaps (e.g. an arm crossing the chest splitting the mask)
    # so occluded torsos still yield one contour covering the full region.
    ys, xs = np.nonzero(torso_mask)
    span = max(xs.max() - xs.min(), ys.max() - ys.min(), 1)
    kernel_size = max(3, int(span * 0.1)) | 1  # odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(torso_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_TORSO_AREA_PX:
        return None

    rect = cv2.minAreaRect(contour)
    box = order_quad(cv2.boxPoints(rect))

    moments = cv2.moments(contour)
    centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)

    return np.vstack([centroid, box])  # (5, 2): centroid, tl, tr, br, bl


def bbox_from_points(pts: np.ndarray, pad_frac: float = 0.15):
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    w, h = x1 - x0, y1 - y0
    x0 -= w * pad_frac
    x1 += w * pad_frac
    y0 -= h * pad_frac
    y1 += h * pad_frac
    return x0, y0, x1, y1


def to_yolo_label_line(kpts: np.ndarray, box_pts: np.ndarray, img_w: int, img_h: int) -> str:
    x0, y0, x1, y1 = bbox_from_points(box_pts)
    x0, x1 = np.clip([x0, x1], 0, img_w)
    y0, y1 = np.clip([y0, y1], 0, img_h)
    xc, yc = (x0 + x1) / 2 / img_w, (y0 + y1) / 2 / img_h
    w, h = (x1 - x0) / img_w, (y1 - y0) / img_h

    parts = [0, xc, yc, w, h]
    for px, py in kpts:
        parts += [np.clip(px / img_w, 0, 1), np.clip(py / img_h, 0, 1), 2]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in parts)


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
            torso_mask = decode_torso_mask(a["dp_masks"], a.get("dp_I"), a["bbox"], w, h)
            kpts = trunk_keypoints(torso_mask)
            if kpts is None:
                continue
            centroid, box = kpts[0], kpts[1:]
            lines.append(to_yolo_label_line(kpts, box, w, h))

        if not lines:
            skipped += 1
            continue

        out_img = DATASET_DIR / "images" / split / info["file_name"]
        out_lbl = DATASET_DIR / "labels" / split / (Path(info["file_name"]).stem + ".txt")
        cv2.imwrite(str(out_img), gray_bgr)
        out_lbl.write_text("\n".join(lines))
        kept += 1

    print(f"Built dataset: {kept} images with trunk labels, {skipped} skipped (no valid torso mask)")

    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text(f"""\
path: {DATASET_DIR.resolve()}
train: images/train
val: images/val
names:
  0: trunk
kpt_shape: [5, 3]
flip_idx: [0, 2, 1, 4, 3]
""")
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    build()
