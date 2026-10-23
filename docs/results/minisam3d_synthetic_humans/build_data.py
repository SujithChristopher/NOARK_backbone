"""Collect every number and picture the synthetic-humans results report shows (../minisam3d_synthetic_humans.html).

Writes data.js (window.REPORT_DATA = {...}, loaded by the page with a plain <script> tag so it opens from disk) and
img/*.jpg. Reads only finished outputs; it trains or infers nothing:
  mocap         trunkpose/runs/minisam3d/<run>_octtest/<take>/student/sam3d_vs_mocap.npz (score_student_noark.py),
                the teacher's in r18b_loso_kabil_octtest, ICP's icp_vs_mocap_mode.npz in the take folder
  vs teacher    minisam3d runs/*/test_trunk_zeroed.json (eval_trunk_zeroed.py), runs/*/test_eval.json (eval_teacher.py)
  shrinkage     data/shrinkage.json, written by shrinkage.py (needs the GPU and the minisam3d venv)
  flexion       per-take (real) / per-person (synthetic) zeroed trunk flexion from the label shards and synth_gt.npz
  synthetic     teacher (sam3d_out.npz) vs rendered truth (synth_gt.npz), as in minisam3d synth/check_teacher.py

  .venv/Scripts/python docs/results/minisam3d_synthetic_humans/build_data.py        (from the NOARK_backbone root)
"""

import glob
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
NOARK = HERE.parents[2]
MS = Path("E:/NOARK_CV/minisam3d")
SYN = Path("E:/NOARK_CV/synth_data")
OCT = NOARK / "trunkpose" / "runs" / "minisam3d"
DAYS = {"arjun": "oct10_26", "jerome": "oct10_26", "lee": "oct9_26"}
AX = ("flexion", "lateral", "axial", "magnitude")
MODELS = {  # report key -> (octtest folder, file kind, minisam3d run)
    "v2": ("r18_v2_all_octtest", "student", "r18_v2_all"),
    "v3": ("r18_v3_all_octtest", "student", "r18_v3_all"),
    "v3mix": ("r18_v3mix_all_octtest", "student", "r18_v3mix_all"),
    "v3mix_rep2": ("r18_v3mix_all_rep2_octtest", "student", "r18_v3mix_all_rep2"),
    "v4mix": ("r18_v4mix_all_octtest", "student", "r18_v4mix_all"),
    "teacher": ("r18b_loso_kabil_octtest", "teacher", None),
}
C = np.diag([1.0, -1.0, -1.0])  # MHR model axes -> OpenCV camera axes


def mocap():
    """Per take and method: median |err|, bias, SD and n vs mocap on ICP-solved frames (21_sam3d_score.py)."""
    out = {}
    for subj, day in DAYS.items():
        for kind in ("woc", "wc"):
            take = f"s3_{subj}_{kind}"
            icp = np.load(
                NOARK
                / "data/trunk_take_oct"
                / day
                / "std_movement"
                / take
                / "icp_vs_mocap_mode.npz",
                allow_pickle=True,
            )
            row = {}
            for m, (folder, sub, _) in list(MODELS.items()) + [
                ("icp", (None, None, None))
            ]:
                row[m] = {}
                x = (
                    None
                    if m == "icp"
                    else np.load(OCT / folder / take / sub / "sam3d_vs_mocap.npz")
                )
                for ax in AX:
                    ok = (
                        icp["valid_" + ax]
                        & np.isfinite(icp["icp_" + ax])
                        & icp["in_win"]
                        & np.isfinite(icp["moc_" + ax])
                    )
                    if x is None:
                        e = (icp["icp_" + ax] - icp["moc_" + ax])[ok]
                    else:
                        ok = ok & x["valid_" + ax] & np.isfinite(x["moc_" + ax])
                        e = (x["sam_" + ax] - x["moc_" + ax])[ok]
                    row[m][ax] = {
                        "med": round(float(np.median(np.abs(e))), 3),
                        "bias": round(float(e.mean()), 3),
                        "sd": round(float(e.std()), 3),
                        "n": int(ok.sum()),
                    }
            out[take] = row
    return out


def vs_teacher():
    """Zeroed trunk agreement and keypoint / arm errors vs the teacher, per run and test take (latest file wins)."""
    zeroed, kp = {}, {}
    files_z = ["r18_v3_all", "r18_v3mix_all", "r18_v4mix_all", "r18_v3mix_all_rep2"]
    files_e = ["r18_v3_all", "r18_v3mix_all", "r18_v4mix_all", "r18_v3mix_all_rep2"]
    runs = {v[2]: k for k, v in MODELS.items() if v[2]}
    for f in files_z:
        p = MS / "runs" / f / "test_trunk_zeroed.json"
        if p.exists():
            for ck, takes in json.loads(Path(p).read_text(encoding="utf-8"))[
                "results"
            ].items():
                if ck in runs:
                    zeroed[runs[ck]] = {t.split("/")[-1]: v for t, v in takes.items()}
    for f in files_e:
        p = MS / "runs" / f / "test_eval.json"
        if p.exists():
            for ck, takes in json.loads(Path(p).read_text(encoding="utf-8"))[
                "results"
            ].items():
                if ck in runs:
                    kp[runs[ck]] = {t.split("/")[-1]: v for t, v in takes.items()}
    return zeroed, kp


def zero_rotvec(R, groups):
    v = np.empty((len(R), 3))
    for g in np.unique(groups):
        m = groups == g
        v[m] = Rotation.from_matrix(
            R[m] @ Rotation.from_matrix(R[m]).mean().as_matrix().T
        ).as_rotvec(degrees=True)
    return v


def flexion_distributions():
    """Zeroed trunk flexion (+ = toward the camera) per data source, as 2.5-deg histograms and |flex| < 5 shares."""
    man = json.loads(
        Path(MS / "datasets" / "noark_v4.json").read_text(encoding="utf-8")
    )
    src = {
        "real train, game": [],
        "real train, wc + woc": [],
        "test, wc + woc": [],
        "test, game": [],
    }
    for split in ("train", "test"):
        for root in man["splits"][split]:
            info = man["roots"][root]
            if not root.startswith("noark_pack/") and not root.startswith(
                "noark_pack_oct/"
            ):
                continue
            rot = np.concatenate(
                [
                    np.load(f)["rot"][:, 0]
                    for f in sorted(
                        glob.glob(str(MS / "data" / root / "labels" / "shard_*.npz"))
                    )
                ]
            )
            flex = zero_rotvec(rot.astype(np.float64), np.zeros(len(rot)))[:, 0]
            game = info.get("kind") == "game" or root.endswith("_game")
            key = ("real train, " if split == "train" else "test, ") + (
                "game" if game else "wc + woc"
            )
            src[key].append(flex)
    for name in ("syn15k", "synflex15k"):
        gt = np.load(SYN / name / "sam3d" / "synth_gt.npz")
        src[name] = [
            zero_rotvec(gt["trunk_R_cv"].astype(np.float64), gt["group"])[:, 0]
        ]
    edges = np.arange(-30, 47.5, 2.5)
    out = {"edges": edges.tolist(), "sets": []}
    for k, v in src.items():
        f = np.concatenate(v)
        h = np.histogram(np.clip(f, edges[0], edges[-1] - 1e-6), edges)[0]
        out["sets"].append(
            {
                "name": k,
                "n": len(f),
                "pct": (100 * h / len(f)).round(2).tolist(),
                "within5": round(float(np.mean(np.abs(f) < 5) * 100), 1),
                "over15": round(float(np.mean(np.abs(f) >= 15) * 100), 1),
                "p5": round(float(np.percentile(f, 5)), 1),
                "p95": round(float(np.percentile(f, 95)), 1),
            }
        )
    return out


def synth_teacher(name, n_keep=1500, seed=0):
    """Teacher vs rendered truth on one synthetic set: per-person zeroed rotvec components, gains, a scatter sample."""
    gt, res = (
        np.load(SYN / name / "sam3d" / "synth_gt.npz"),
        np.load(SYN / name / "sam3d" / "sam3d_out.npz"),
    )
    ok = res["ok"] & np.isfinite(res["pred_global_rots"][:, 37]).all((1, 2))
    g = gt["group"]
    full = np.isin(g, [x for x in np.unique(g) if ok[g == x].all()])
    vt = zero_rotvec(
        C @ res["pred_global_rots"][full, 37].astype(np.float64) @ C, g[full]
    )
    vg = zero_rotvec(gt["trunk_R_cv"][full].astype(np.float64), g[full])
    stats = {}
    for k, ax in enumerate(("flexion", "axial", "lateral")):
        stats[ax] = {
            "gain": round(float(vt[:, k] @ vg[:, k] / (vg[:, k] @ vg[:, k])), 3),
            "r": round(float(np.corrcoef(vt[:, k], vg[:, k])[0, 1]), 3),
            "med": round(float(np.median(np.abs(vt[:, k] - vg[:, k]))), 2),
        }
    pick = np.random.default_rng(seed).choice(
        len(vt), min(n_keep, len(vt)), replace=False
    )
    return {
        "n": int(full.sum()),
        "people": len(np.unique(g[full])),
        "stats": stats,
        "truth": vg[pick].round(1).tolist(),
        "teacher": vt[pick].round(1).tolist(),
    }


def composition():
    out = {}
    for ver in ("noark_v2", "noark_v3", "noark_v4"):
        m = json.loads(
            Path(MS / "datasets" / f"{ver}.json").read_text(encoding="utf-8")
        )
        grp = {
            "COCO 2014 (teacher labels)": 0,
            "real NOARK, sep23": 0,
            "real NOARK, oct": 0,
            "synthetic": 0,
        }
        for root in m["splits"]["train"]:
            n = m["roots"][root]["frames"]
            key = (
                "COCO 2014 (teacher labels)"
                if root.startswith("coco")
                else "synthetic"
                if "synth" in root
                else "real NOARK, oct"
                if "_oct/" in root
                else "real NOARK, sep23"
            )
            grp[key] += n
        test = sum(m["roots"][r]["frames"] for r in m["splits"]["test"])
        out[ver] = {
            "id": m["id"],
            "train": grp,
            "train_roots": len(m["splits"]["train"]),
            "test_frames": test,
            "test_roots": len(m["splits"]["test"]),
        }
    return out


def images():
    """Synthetic pictures only (no recorded participants): raw render vs camera look, teacher overlays, a lean series."""
    d = HERE / "img"
    d.mkdir(exist_ok=True)
    q = [cv2.IMWRITE_JPEG_QUALITY, 82]
    shown = {}
    for name in ("g00007_03", "g00412_05", "g01111_02"):
        meta = np.load(SYN / "syn15k" / "sam3d" / "meta.npz")
        i = int(np.where(meta["names"] == name)[0][0])
        cv2.imwrite(
            str(d / f"syn15k_{name}_render.jpg"),
            cv2.imread(str(SYN / "syn15k" / "frames" / f"{name}.png")),
            q,
        )
        cv2.imwrite(
            str(d / f"syn15k_{name}_look.jpg"),
            cv2.imread(str(SYN / "syn15k" / "sam3d" / f"img_{meta['idx'][i]:05d}.png")),
            q,
        )
        shown.setdefault("look", []).append(name)
    for s in ("syn15k", "synflex15k"):
        for k in (3, 7):
            cv2.imwrite(
                str(d / f"{s}_overlay_{k}.jpg"),
                cv2.imread(str(SYN / s / "sam3d" / f"overlay_{k:05d}.png")),
                q,
            )
    # one synflex15k person across its flexion range (rendered truth, zeroed on the person's neutral = pool delta)
    gt = np.load(SYN / "synflex15k" / "sam3d" / "synth_gt.npz")
    meta = np.load(SYN / "synflex15k" / "sam3d" / "meta.npz")
    flex = Rotation.from_matrix(gt["trunk_R_rel_cv"].astype(np.float64)).as_rotvec(
        degrees=True
    )[:, 0]
    best = max(np.unique(gt["group"]), key=lambda g: np.ptp(flex[gt["group"] == g]))
    m = np.where(gt["group"] == best)[0]
    m = m[np.argsort(flex[m])]
    series = []
    for j in m[np.linspace(0, len(m) - 1, 4).round().astype(int)]:
        name = str(gt["names"][j])
        k = int(np.where(meta["names"] == name)[0][0])
        x0, y0, x1, y1 = meta["boxes"][k].astype(int)
        img = cv2.imread(
            str(SYN / "synflex15k" / "sam3d" / f"img_{meta['idx'][k]:05d}.png")
        )[y0 : y1 + 1, x0 : x1 + 1]
        cv2.imwrite(str(d / f"synflex15k_{name}_crop.jpg"), img, q)
        lab = json.loads(
            Path(SYN / "synflex15k" / "frames" / f"{name}.json").read_text(
                encoding="utf-8"
            )
        )
        series.append(
            {
                "name": name,
                "flex": round(float(flex[j]), 1),
                "mirrored": bool(lab["pose"]["mirrored"]),
            }
        )
    shown["lean"] = series
    return shown


def main():
    zeroed, kp = vs_teacher()
    shr = HERE / "data" / "shrinkage.json"
    data = {
        "mocap": mocap(),
        "zeroed": zeroed,
        "kp": kp,
        "shrinkage": json.loads(Path(shr).read_text(encoding="utf-8"))
        if shr.exists()
        else None,
        "flexion": flexion_distributions(),
        "synth_teacher": {s: synth_teacher(s) for s in ("syn15k", "synflex15k")},
        "composition": composition(),
        "images": images(),
    }
    (HERE / "data.js").write_text(
        "window.REPORT_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(
        f"data.js {(HERE / 'data.js').stat().st_size / 1e3:.0f} kB; images {len(list((HERE / 'img').glob('*.jpg')))}"
    )


if __name__ == "__main__":
    main()
