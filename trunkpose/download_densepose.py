"""Download DensePose COCO minival annotations + N images, plot IUV labels."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm

ANNOT_URL = "https://dl.fbaipublicfiles.com/densepose/densepose_coco_2014_minival.json"
IMG_URL_TMPL = "http://images.cocodataset.org/val2014/COCO_val2014_{:012d}.jpg"

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
IMG_DIR = DATA_DIR / "images"
PLOT_DIR = ROOT / "plots"
ANNOT_PATH = DATA_DIR / "densepose_coco_2014_minival.json"

# DensePose body-part index -> color (24 parts)
PART_CMAP = plt.get_cmap("tab20b", 24)


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                pbar.update(len(chunk))


def download_annotations() -> None:
    download_file(ANNOT_URL, ANNOT_PATH)


def load_densepose_image_ids(coco: COCO, n: int) -> list[int]:
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    dp_anns = [a for a in anns if "dp_masks" in a]
    seen, img_ids = set(), []
    for a in dp_anns:
        if a["image_id"] not in seen:
            seen.add(a["image_id"])
            img_ids.append(a["image_id"])
        if len(img_ids) >= n:
            break
    return img_ids


def download_images(coco: COCO, img_ids: list[int]) -> None:
    for img_id in tqdm(img_ids, desc="images"):
        info = coco.loadImgs(img_id)[0]
        dest = IMG_DIR / info["file_name"]
        download_file(IMG_URL_TMPL.format(img_id), dest)


def decode_dp_mask(dp_masks_entry, bbox) -> np.ndarray:
    """Decode per-part RLE (256x256 grid) into a full-size part-index mask."""
    _, _, w, h = bbox
    part_grid = np.zeros((256, 256), dtype=np.uint8)
    for part_idx, rle in enumerate(dp_masks_entry, start=1):
        if not rle:
            continue
        m = mask_utils.decode(rle)
        part_grid[m > 0] = part_idx
    resized = Image.fromarray(part_grid).resize((max(int(w), 1), max(int(h), 1)), Image.NEAREST)
    return np.array(resized)


def plot_sample(coco: COCO, img_id: int, out_path: Path) -> None:
    info = coco.loadImgs(img_id)[0]
    img_path = IMG_DIR / info["file_name"]
    img = plt.imread(img_path)

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = [a for a in coco.loadAnns(ann_ids) if "dp_masks" in a]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img)
    axes[0].set_title(f"image {img_id}")
    axes[0].axis("off")

    axes[1].imshow(img)
    for a in anns:
        x, y, w, h = a["bbox"]
        part_mask = decode_dp_mask(a["dp_masks"], a["bbox"])
        masked = np.ma.masked_equal(part_mask, 0)
        axes[1].imshow(masked, extent=(x, x + w, y + h, y), cmap=PART_CMAP, vmin=1, vmax=24, alpha=0.6)
        if "dp_x" in a:
            px = np.array(a["dp_x"]) / 255.0 * w + x
            py = np.array(a["dp_y"]) / 255.0 * h + y
            axes[1].scatter(px, py, s=2, c="white", alpha=0.5)
    axes[1].set_title("DensePose IUV part labels")
    axes[1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--n-plots", type=int, default=6)
    args = parser.parse_args()

    print("Downloading DensePose annotations...")
    download_annotations()

    coco = COCO(str(ANNOT_PATH))
    img_ids = load_densepose_image_ids(coco, args.n_images)
    print(f"Selected {len(img_ids)} images with DensePose labels")

    print("Downloading images...")
    download_images(coco, img_ids)

    print(f"Plotting {args.n_plots} samples...")
    for img_id in img_ids[: args.n_plots]:
        plot_sample(coco, img_id, PLOT_DIR / f"densepose_{img_id}.png")

    print(f"Done. Images: {IMG_DIR}  Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
