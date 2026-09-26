"""Create four poster figures illustrating the trunk/arm YOLO-seg training data.

The same two color images are used in every figure:

1. Color images, stacked horizontally.
2. Color images with torso and arm polygons + class labels.
3. Grayscale images, stacked horizontally.
4. Grayscale images with torso and arm polygons + class labels.

The polygons are generated with the same DensePose decoding and contour
processing used by ``trunkpose/segdataset.py``.  Head and lower-body parts are
intentionally omitted from the poster.
"""

import importlib
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
seg = importlib.import_module("segdataset")


OUT_DIR = ROOT / "poster_figures"
IMAGE_NAMES = [
    "COCO_train2014_000000000036.jpg",  # first selected sample
    "COCO_train2014_000000000110.jpg",  # fourth selected sample
]
CLASS_NAMES = {0: "torso", 1: "arms"}
CLASS_COLORS = {0: (31, 119, 180), 1: (214, 39, 40)}  # BGR for OpenCV
MASK_ALPHA = 0.9


def poster_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 15,
            "savefig.dpi": 300,
        }
    )


def masks_for_image(coco, image_name):
    image_id = int(Path(image_name).stem.split("_")[-1])
    info = coco.loadImgs(image_id)[0]
    masks = []
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=image_id)):
        if "dp_masks" not in ann:
            continue
        decoded = seg.decode_part_masks(
            ann["dp_masks"], ann["bbox"], info["width"], info["height"]
        )
        for class_id in (0, 1):
            mask = decoded.get(class_id)
            if mask is None:
                continue
            polygon = seg.part_polygon(mask, seg.MIN_AREA_PX[class_id])
            if polygon is not None:
                masks.append((class_id, polygon))
    return masks


def annotated_image(image, polygons, grayscale=False):
    if grayscale:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image.copy()

    overlay = canvas.copy()
    for class_id, polygon in polygons:
        points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        color = CLASS_COLORS[class_id]
        cv2.fillPoly(overlay, [points], color)
    canvas = cv2.addWeighted(overlay, MASK_ALPHA, canvas, 1.0 - MASK_ALPHA, 0)

    for class_id, polygon in polygons:
        points = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)
        color = CLASS_COLORS[class_id]
        cv2.polylines(canvas, [points], True, color, 3, cv2.LINE_AA)
        x, y = points[:, 0, :].min(axis=0)
        cv2.putText(
            canvas,
            CLASS_NAMES[class_id],
            (int(x), max(int(y) - 8, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def square_crop(image):
    """Center-crop an image to a square so every collage cell is identical."""
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    return image[y0 : y0 + side, x0 : x0 + side]


def square_crop_with_polygons(image, polygons):
    """Square-crop an image and translate its polygons into the cropped frame."""
    height, width = image.shape[:2]
    side = min(height, width)
    y0 = (height - side) // 2
    x0 = (width - side) // 2
    cropped = image[y0 : y0 + side, x0 : x0 + side]
    shifted = [(class_id, polygon - (x0, y0)) for class_id, polygon in polygons]
    return cropped, shifted


def make_figure(images, title, output_path, cmap=None):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 6), squeeze=False)
    for index, (ax, image, name) in enumerate(zip(axes.flat, images, IMAGE_NAMES)):
        ax.imshow(image, cmap=cmap, aspect="equal")
        ax.set_title(chr(65 + index), loc="left", fontsize=14, pad=2, fontweight="bold")
        ax.axis("off")
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94, wspace=0.012)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved {output_path}")


def main():
    poster_style()
    OUT_DIR.mkdir(exist_ok=True)
    coco = seg.COCO(str(seg.ANNOT_PATH))

    color_images = []
    color_masks = []
    for image_name in IMAGE_NAMES:
        image_path = seg.IMG_DIR / image_name
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)
        polygons = masks_for_image(coco, image_name)
        if not polygons:
            raise RuntimeError(f"No torso/arm polygons found for {image_name}")
        color_images.append(square_crop(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        color_masks.append(square_crop_with_polygons(image, polygons))

    make_figure(
        color_images,
        "Training examples: color images",
        OUT_DIR / "figure_1_color_images.png",
    )
    make_figure(
        [annotated_image(image, polygons) for image, polygons in color_masks],
        "Training examples: torso and arm segmentation labels",
        OUT_DIR / "figure_2_color_segmentation.png",
    )
    make_figure(
        [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image, _ in color_masks],
        "Training examples: grayscale images",
        OUT_DIR / "figure_3_grayscale_images.png",
        cmap="gray",
    )
    make_figure(
        [annotated_image(image, polygons, grayscale=True) for image, polygons in color_masks],
        "Training examples: grayscale images with torso and arm labels",
        OUT_DIR / "figure_4_grayscale_segmentation.png",
    )


if __name__ == "__main__":
    main()
