# %% 26_method_compare_video.py
# Side-by-side video of the three trunk trackers on one protocol take: ICP (15),
# the SAM 3D Body teacher (20/21) and the miniSAM3D student (external/minisam3d,
# scored through score_student_noark.py -> 21), all against mocap.
#
# Layout (1600x900):
#   left   undistorted cam0 frame (19's sam3d/img_*.png) with the teacher (magenta) and
#          student (cyan) upper-body skeletons, and a per-frame table of the three
#          methods' angles and |error| vs mocap
#   right  flexion / lateral / axial traces over the whole take with a moving cursor
#
# Each method's angles are already zeroed and S-fitted by 07.compare_mocap, so the
# series are exactly the ones behind the reported scores. S is fitted per method, so
# each has its own mocap series; the traces draw the teacher's (they agree with the
# others to ~1 deg median), while the table's errors use each method's own mocap.
#
# Only the frames 19 exported (the mocap window) are rendered. Game takes are not
# supported: 21 cannot score oct game takes without an ICP result.
#
#   uv run python trunkpose/dual_notebooks/26_method_compare_video.py \
#       --session data/trunk_take_oct/oct10_26 --subject s3_arjun_wc \
#       --student-dir trunkpose/runs/minisam3d/r18b_loso_kabil_octtest/s3_arjun_wc
#
# Needs <student-dir>/{student_out.npz, student/, teacher/sam3d_vs_mocap.npz} (written
# by score_student_noark.py --save-dir) and <rec>/icp_vs_mocap<sfx>.npz.
# Writes <student-dir>/compare_<subject>.mp4

import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "minisam3d"))
from minisam3d.keypoints import joint_set

AX = ("flexion", "lateral", "axial")
W, H = 1600, 900
IMG_W, IMG_H = 960, 600
PLOT_W, PLOT_H = W - IMG_W, H // 3
# BGR for cv2, hex for matplotlib
METHODS = (
    ("ICP", "icp", (80, 200, 255), "#ffc850"),
    ("teacher", "teacher", (255, 80, 255), "#ff50ff"),
    ("student", "student", (255, 220, 0), "#00dcff"),
)
MOCAP_COL = "#ffffff"


def load_series(student_dir, rec, sfx):
    """{method: {axis: (angle, mocap)}} on the take's frame index."""
    out = {}
    for _, key, _, _ in METHODS:
        if key == "icp":
            d = np.load(rec / f"icp_vs_mocap{sfx}.npz", allow_pickle=True)
            ok = {
                ax: d["valid_" + ax] & np.isfinite(d["icp_" + ax]) & d["in_win"]
                for ax in AX
            }
            out[key] = {
                ax: (np.where(ok[ax], d["icp_" + ax], np.nan), d["moc_" + ax])
                for ax in AX
            }
        else:
            d = np.load(student_dir / key / "sam3d_vs_mocap.npz")
            out[key] = {
                ax: (np.where(d["valid_" + ax], d["sam_" + ax], np.nan), d["moc_" + ax])
                for ax in AX
            }
    return out


def plot_background(t, series, sel):
    """Static traces for the three axes + a function mapping (axis, time) -> pixel x."""
    fig, axes = plt.subplots(
        3, 1, figsize=(PLOT_W / 100, H / 100), dpi=100, sharex=True
    )
    fig.patch.set_facecolor("#111111")
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.07, hspace=0.25)
    for ax, name in zip(axes, AX):
        ax.set_facecolor("#111111")
        ax.plot(
            t[sel],
            series["teacher"][name][1][sel],
            color=MOCAP_COL,
            lw=1.0,
            ls="--",
            alpha=0.8,
            label="mocap",
        )
        for label, key, _, col in METHODS:
            ax.plot(
                t[sel],
                series[key][name][0][sel],
                color=col,
                lw=1.0,
                alpha=0.9,
                label=label,
            )
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"{name} (deg)", color="white", fontsize=8)
        ax.tick_params(colors="gray", labelsize=7)
        ax.set_xlim(t[sel][0], t[sel][-1])
    axes[0].legend(
        loc="upper right",
        fontsize=7,
        ncol=4,
        facecolor="#222222",
        labelcolor="white",
        framealpha=0.6,
    )
    axes[-1].set_xlabel("time (s)", color="gray", fontsize=8)
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    # data -> display -> image pixel (matplotlib's display origin is bottom-left)
    spans = []
    for ax in axes:
        (x0, _), (x1, _) = ax.transData.transform([(t[sel][0], 0), (t[sel][-1], 0)])
        bb = ax.get_window_extent()
        spans.append((x0, x1, H - bb.y1, H - bb.y0))
    plt.close(fig)
    t0, t1 = t[sel][0], t[sel][-1]

    def xpix(k, tt):
        x0, x1, _, _ = spans[k]
        return round(x0 + (tt - t0) / (t1 - t0) * (x1 - x0))

    return img, spans, xpix


def draw_skeleton(img, pts, bones, col):
    for a, b in bones:
        pa, pb = pts[a], pts[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(
                img,
                tuple(int(v) for v in pa),
                tuple(int(v) for v in pb),
                col,
                3,
                cv2.LINE_AA,
            )
    for p in pts:
        if np.isfinite(p).all():
            cv2.circle(img, (int(p[0]), int(p[1])), 5, col, -1, cv2.LINE_AA)


TABLE_COLS = (
    12,
    230,
    470,
    710,
)  # fixed x: Hershey is proportional, spaces do not align


def table_row(canvas, cells, y, col, th=1):
    for x, c in zip(TABLE_COLS, cells):
        cv2.putText(
            canvas, c, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, th, cv2.LINE_AA
        )


def fmt(v):
    return f"{v:+6.1f}" if np.isfinite(v) else "    --"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--subject", required=True, help="protocol take, e.g. s3_arjun_wc")
    ap.add_argument("--student-dir", required=True)
    ap.add_argument(
        "--frames", type=int, default=0, help="cap rendered frames (0 = all)"
    )
    a = ap.parse_args()

    m15 = importlib.import_module("15_subject_icp")
    rec, _, _ = m15.resolve_recording(Path(a.session).resolve(), a.subject, False)
    sd = Path(a.student_dir).resolve()
    os.environ.update(TRUNK_RECORDING_NAME=a.subject, TRUNK_RECORDING_DIR=str(rec))
    sfx = importlib.import_module("06_trunk_axis").RESULT_SUFFIX

    series = load_series(sd, rec, sfx)
    t = np.load(rec / f"icp_vs_mocap{sfx}.npz", allow_pickle=True)["t_s"].astype(float)
    meta = np.load(rec / "sam3d" / "meta.npz")
    stu = np.load(sd / "student_out.npz")
    tea = np.load(rec / "sam3d" / "sam3d_out.npz")
    js = joint_set(False)
    bones = js["bones"]
    mhr = np.asarray(js["mhr_idx"])

    idx = meta["idx"]
    if a.frames:
        idx = idx[: a.frames]
    sel = np.zeros(len(t), bool)
    sel[meta["idx"]] = True
    bg, spans, xpix = plot_background(t, series, sel)
    fps = float(1.0 / np.median(np.diff(t[meta["idx"]])))

    out = sd / f"compare_{a.subject}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for j, i in enumerate(tqdm(idx, desc=a.subject)):
        img = cv2.imread(str(rec / "sam3d" / f"img_{i:05d}.png"))
        if img is None:
            continue
        if tea["ok"][j]:
            draw_skeleton(img, tea["pred_keypoints_2d"][j][mhr], bones, METHODS[1][2])
        if stu["ok"][j]:
            draw_skeleton(img, stu["uv"][j], bones, METHODS[2][2])
        sx, sy = IMG_W / img.shape[1], IMG_H / img.shape[0]
        s = min(sx, sy)
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))

        canvas = np.full((H, W, 3), 17, np.uint8)
        canvas[: img.shape[0], : img.shape[1]] = img
        cv2.putText(
            canvas,
            f"{a.subject}   frame {i}   t {t[i]:+.2f}s",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # table: angle (|error vs that method's own mocap|)
        y = img.shape[0] + 40
        table_row(canvas, ["deg  (|err|)"] + list(AX), y, (200, 200, 200))
        table_row(
            canvas,
            ["mocap (teacher S)"] + [fmt(series["teacher"][ax][1][i]) for ax in AX],
            y + 34,
            (255, 255, 255),
        )
        for r, (label, key, col, _) in enumerate(METHODS):
            cells = [label]
            for ax in AX:
                v, m = series[key][ax][0][i], series[key][ax][1][i]
                e = abs(v - m) if np.isfinite(v) and np.isfinite(m) else np.nan
                cells.append(
                    f"{fmt(v)}  ({'--' if not np.isfinite(e) else f'{e:.1f}'})"
                )
            table_row(canvas, cells, y + 68 + 34 * r, col, 2)
        cv2.putText(
            canvas,
            "(|err|) = vs that method's own S-fitted mocap.   skeleton: magenta = teacher, cyan = student",
            (12, H - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        plot = bg.copy()
        for k in range(3):
            x = xpix(k, t[i])
            _, _, ytop, ybot = spans[k]
            cv2.line(plot, (x, int(ytop)), (x, int(ybot)), (255, 255, 255), 1)
        canvas[:, IMG_W:] = cv2.resize(plot, (PLOT_W, H))
        writer.write(canvas)
    writer.release()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
