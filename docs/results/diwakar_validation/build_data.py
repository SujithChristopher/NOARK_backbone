"""Collect every number and picture the Diwakar validation report shows (../diwakar_validation.html).

Writes data.js (window.REPORT_DATA = {...}, loaded by the page with a plain <script> tag so it opens from disk) and
img/*.jpg. Reads the recordings and the finished outputs of diwakar_validation/ (it re-runs no detector or solver):
  takes        data/diwakar/validaton_data/<take>/{samples,sync,marks}.csv + mocap.csv
  jig          shipped calib/board_geometry.json and out/jig_T0*.json (build_jig_T0.py)
  T1 holds     out/T1_holds.csv, out/T1_errors.csv (analyze_T1.py)
  corners      out/summary_errors.csv, out/summary_errors_jig*.csv (analyze_all.py)
  his metric   out/compare_raw_detect.txt (compare_raw_detect.py / reproduce_diwakar.py)
  jitter       out/static_jitter/{summary,vs_distance}.csv (static_jitter_T1.py)
  paths        out/paths/{summary.csv,corners.csv} (compare_paths.py)
  mocap still  mocap rigid body inside each T1 hold, computed here

  .venv/Scripts/python docs/results/diwakar_validation/build_data.py        (from the NOARK_backbone root)
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
NOARK = HERE.parents[2]
DV = NOARK / "diwakar_validation"
sys.path.insert(0, str(DV))
from common import DATA, MOCAP, OUT, load_jig, load_mocap, mocap_frame_of, shipped_jig_path, take_dir  # noqa: E402

IMG = HERE / "img"
IMG.mkdir(exist_ok=True)
T1 = ["2026-10-03_T1_grid_r1", "2026-10-03_T1_grid_r2"]


def r(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def takes():
    rows = []
    for d in sorted(DATA.glob("2026-*")):
        name = d.name
        s = pd.read_csv(d / "samples.csv", usecols=["t_cam0_s", "qw"])
        edges = len(pd.read_csv(d / "sync.csv"))
        mf = None
        if (MOCAP / name / "mocap.csv").exists():
            with open(MOCAP / name / "mocap.csv") as f:
                head = f.readline().split(",")
            mf = int(head[head.index("Total Exported Frames") + 1])
        holds = 0
        if (d / "marks.csv").exists():
            holds = int(pd.read_csv(d / "marks.csv").label.str.startswith("hold_start").sum())
        task = name.split("_")[1]
        rows.append({
            "take": name, "task": task, "date": name[:10],
            "dur_s": r(s.t_cam0_s.iloc[-1] - s.t_cam0_s.iloc[0], 1), "samples": len(s),
            "pose_frac": r(s.qw.notna().mean(), 3), "edges": edges, "mocap_frames": mf, "holds": holds,
            "images": (d / "images_cam0.bin").exists(),
        })
    return rows


def jig():
    out = {}
    for key, path in {"shipped": shipped_jig_path(), "T0": OUT / "jig_T0.json",
                      "T0_april": OUT / "jig_T0_april.json"}.items():
        j = load_jig(path)
        tags = {}
        for k, (R, t) in j["tags"].items():
            n = R @ np.array([0, 0, 1.0])
            tags[k] = {"t_mm": [r(v * 1000, 2) for v in t], "n": [r(v, 3) for v in n],
                       "corners_mm": [[r(v * 1000, 2) for v in c] for c in j["obj"][k]]}
        out[key] = {"tags": tags, "grip_mm": [r(v * 1000, 2) for v in j["grip"]]}
    # T0 vs shipped tag centre / normal differences
    diff = []
    for k in out["shipped"]["tags"]:
        a, b = np.array(out["shipped"]["tags"][k]["t_mm"]), np.array(out["T0"]["tags"][k]["t_mm"])
        na, nb = np.array(out["shipped"]["tags"][k]["n"]), np.array(out["T0"]["tags"][k]["n"])
        diff.append({"id": int(k), "dpos_mm": r(np.linalg.norm(a - b), 2),
                     "dang_deg": r(np.degrees(np.arccos(np.clip(na @ nb, -1, 1))), 2)})
    out["diff"] = diff
    st = json.loads((take_dir(T1[0]) / "calib" / "stereo_extrinsics.json").read_text())
    out["baseline_mm"] = r(np.linalg.norm(st["tx"]) * 1000, 1)
    out["stereo_angle_deg"] = r(np.degrees(Rotation.from_matrix(np.array(st["Rx"])).magnitude()), 1)
    return out


def t1_holds():
    df = pd.read_csv(OUT / "T1_holds.csv")
    keep = ["take", "hold", "target_x", "target_y", "mc_x", "mc_y", "mc_z", "n_cam"]
    rows = []
    for _, h in df.iterrows():
        row = {k: (h[k] if k in ("take",) else r(h[k] * (1000 if k.startswith("mc_") else 1), 2)) for k in keep}
        row["take"] = h["take"][-2:]
        for m in ("cam0", "cam1", "joint", "joint_logged"):
            row[m] = [r(h[f"{m}_ex_mm"], 3), r(h[f"{m}_ez_mm"], 3), r(h[f"{m}_sd_mm"], 3)]
        rows.append(row)
    err = pd.read_csv(OUT / "T1_errors.csv")
    errs = [{k: (v if isinstance(v, str) else r(v, 3)) for k, v in e.items()} for e in err.to_dict("records")]
    return {"holds": rows, "errors": errs}


def mocap_still():
    rows, timeline = [], {}
    for name in T1:
        mc = load_mocap(name)
        mk = pd.read_csv(take_dir(name) / "marks.csv")
        st = mk[mk.label.str.startswith("hold_start")].reset_index(drop=True)
        en = mk[mk.label.str.startswith("hold_end")].reset_index(drop=True)
        spans = []
        for i in range(min(len(st), len(en))):
            a, b = mocap_frame_of(name, np.array([st.t_s[i], en.t_s[i]]))
            if np.isnan(a) or np.isnan(b):
                continue
            idx = np.arange(int(np.ceil(a)), int(np.floor(b)) + 1)
            v = mc["valid"][idx]
            p = mc["p"][idx][v] * 1000
            if len(p) < 20:
                continue
            t = idx[v] * 0.01
            mean = p.mean(0)
            sd3 = np.sqrt(((p - mean) ** 2).sum(1).mean())
            noise = np.sqrt((np.diff(p, axis=0) ** 2).sum(1).mean() / 2)
            slope = np.polyfit(t - t[0], p, 1)[0]
            rot = Rotation.from_matrix(mc["R"][idx][v])
            ang = np.degrees((rot.mean().inv() * rot).magnitude())
            lab = st.label[i].split()
            rows.append({
                "take": name[-2:], "hold": int(lab[1]), "tx": float(lab[2]), "ty": float(lab[3]),
                "dur": r(t[-1] - t[0], 2), "valid": r(v.mean(), 3), "sd3": r(sd3, 3), "noise": r(noise, 3),
                "maxdev": r(np.linalg.norm(p - mean, axis=1).max(), 3),
                "drift": r(np.linalg.norm(slope) * (t[-1] - t[0]), 3),
                "rot_max": r(ang.max(), 3), "rot_sd": r(np.sqrt((ang**2).mean()), 3),
                "x": r(mean[0], 1), "z": r(mean[2], 1),
            })
            spans.append([r(a * 0.01, 2), r(b * 0.01, 2)])
        # 10 Hz mocap X / Z timeline for the take figure
        k = np.arange(0, len(mc["p"]), 10)
        p = mc["p"][k] * 1000
        ok = mc["valid"][k]
        timeline[name[-2:]] = {"t": [r(v, 1) for v in k[ok] * 0.01], "x": [r(v, 1) for v in p[ok, 0]],
                               "z": [r(v, 1) for v in p[ok, 2]], "holds": spans}
    return {"holds": rows, "timeline": timeline}


def errors_by_take():
    out = []
    for f, study in (("summary_errors.csv", "corners"), ("summary_errors_jig.csv", "jig"),
                     ("summary_errors_jig_april.csv", "jig_april")):
        d = pd.read_csv(OUT / f)
        for _, e in d.iterrows():
            out.append({"study": study, "take": e["take"].replace("2026-10-03_", ""), "task": e.task,
                        "source": e.source, "method": e.method, "n": int(e.n),
                        "x": r(e.x_rmse_mm), "z": r(e.z_rmse_mm), "xz": r(e.xz_rmse_mm),
                        "xb": r(e.x_bias_mm), "zb": r(e.z_bias_mm), "xmax": r(e.x_max_mm), "zmax": r(e.z_max_mm)})
    return out


def his_metric():
    """Parse compare_raw_detect.txt: corner offsets vs logged and his 13-parameter fit table."""
    txt = (OUT / "compare_raw_detect.txt").read_text().splitlines()
    corners, fit = [], []
    for ln in txt:
        p = ln.split()
        if len(p) > 8 and p[0] in ("r1", "r2") and p[2] == "n":
            corners.append({"run": p[0], "source": p[1], "n": int(p[3]), "med_px": float(p[5]),
                            "p95_px": float(p[7]), "side_pct": float(p[9])})
        if len(p) == 18 and p[0] in ("logged", "none", "contour", "april"):
            v = [float(x) for x in p[2:] if x != "|"]
            fit.append({"source": p[0], "scale": float(p[1]), "r1_in": v[0:3], "r2_in": v[3:6],
                        "r1_r2": v[6:9], "r2_r1": v[9:12]})
    return {"corners": corners, "fit": fit}


def jitter():
    sm = pd.read_csv(OUT / "static_jitter" / "summary.csv")
    vd = pd.read_csv(OUT / "static_jitter" / "vs_distance.csv")
    return {"summary": [{k: (v if isinstance(v, str) else r(v, 4)) for k, v in e.items()} for e in sm.to_dict("records")],
            "distance": [{k: (v if isinstance(v, str) else r(v, 4)) for k, v in e.items()} for e in vd.to_dict("records")]}


def paths():
    sm = pd.read_csv(OUT / "paths" / "summary.csv")
    c = pd.read_csv(OUT / "paths" / "corners.csv", usecols=["refine", "radius_px", "corner_diff_px", "side_ratio"])
    edges = [0, 150, 250, 350, 450, 800]
    c["bin"] = pd.cut(c.radius_px, edges, labels=False)
    g = c.groupby(["refine", "bin"]).agg(gap=("corner_diff_px", "median"), side=("side_ratio", "median"),
                                         n=("corner_diff_px", "size")).reset_index()
    return {"summary": [{k: (v if isinstance(v, str) else r(v, 4)) for k, v in e.items()} for e in sm.to_dict("records")],
            "radius": [{"refine": e.refine, "lo": edges[int(e.bin)], "hi": edges[int(e.bin) + 1],
                        "gap": r(e.gap, 3), "side_pct": r((e.side - 1) * 100, 3), "n": int(e.n)} for e in g.itertuples()],
            "edges": edges}


# ---------------------------------------------------------------- images


def crop(name, sample, cam):
    ix = pd.read_csv(take_dir(name) / "images.csv")
    e = ix[(ix["sample"] == sample) & (ix.cam == cam)].iloc[0]
    with open(take_dir(name) / f"images_cam{cam}.bin", "rb") as f:
        f.seek(int(e.offset))
        a = np.frombuffer(f.read(int(e.width * e.height)), np.uint8).reshape(int(e.height), int(e.width))
    return a, int(e.x0), int(e.y0)


def overlay(name, sample, out, scale=2):
    """Both cameras' raw fisheye crops with the logged (undistorted-space) corners mapped back onto them."""
    meta = json.loads((take_dir(name) / "meta.json").read_text())
    c = pd.read_csv(take_dir(name) / "corners.csv")
    c = c[c["sample"] == sample]
    tiles = []
    for cam in (0, 1):
        img, x0, y0 = crop(name, sample, cam)
        K = np.array(meta["lens"][f"camera_matrix_cam{cam}"])
        D = np.array(meta["lens"][f"dist_coeffs_cam{cam}"])
        Ku = np.array(meta["undistorted"][f"camera_matrix_cam{cam}"])
        col = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        for _, t in c[c.cam == cam].iterrows():
            uv = t[["u0", "v0", "u1", "v1", "u2", "v2", "u3", "v3"]].to_numpy(float).reshape(4, 2)
            nrm = (np.c_[uv, np.ones(4)] @ np.linalg.inv(Ku).T)[:, :2]
            raw = cv2.fisheye.distortPoints(np.ascontiguousarray(nrm).reshape(-1, 1, 2), K, D).reshape(4, 2) - [x0, y0]
            pts = np.round(raw * scale * 4).astype(np.int32)
            cv2.polylines(col, [pts], True, (214, 120, 42), 2, cv2.LINE_AA, shift=2)
            cv2.circle(col, tuple(pts[0]), 4 * 4, (52, 104, 235), -1, cv2.LINE_AA, shift=2)
            cen = raw.mean(0) * scale
            cv2.putText(col, str(int(t.marker_id)), (int(cen[0]) - 10, int(cen[1]) + 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(col, str(int(t.marker_id)), (int(cen[0]) - 10, int(cen[1]) + 7), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(col)
    for i, tl in enumerate(tiles):
        cv2.imwrite(str(IMG / f"{out}_cam{i}.jpg"), tl, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return [f"{out}_cam{i}.jpg" for i in range(2)]


def pick_sample(name, hold_idx):
    mk = pd.read_csv(take_dir(name) / "marks.csv")
    st = mk[mk.label.str.startswith("hold_start")].reset_index(drop=True)
    en = mk[mk.label.str.startswith("hold_end")].reset_index(drop=True)
    return int((st["sample"][hold_idx] + en["sample"][hold_idx]) // 2)


def copy_plots():
    out = {}
    for take in ("T2_circle_comfortable_r1", "T2_eight_fast_r1", "T3_comfortable_r1"):
        src = OUT / "plots_jig" / f"2026-10-03_{take}.png"
        im = cv2.imread(str(src))
        im = cv2.resize(im, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(IMG / f"{take}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        out[take] = f"{take}.jpg"
    for f in ("corners", "jitter", "accuracy"):
        shutil.copy(OUT / "paths" / f"{f}.png", IMG / f"paths_{f}.png")
    return out


def main():
    D = {
        "takes": takes(),
        "jig": jig(),
        "t1": t1_holds(),
        "still": mocap_still(),
        "byTake": errors_by_take(),
        "his": his_metric(),
        "jitter": jitter(),
        "paths": paths(),
        "img": {
            "t1_centre": overlay(T1[0], pick_sample(T1[0], 40), "t1_hold40"),
            "t1_corner": overlay(T1[0], pick_sample(T1[0], 0), "t1_hold0"),
            "plots": copy_plots(),
        },
    }
    js = "window.REPORT_DATA = " + json.dumps(D, separators=(",", ":"), allow_nan=False) + ";\n"
    (HERE / "data.js").write_text(js)
    print("data.js", len(js) // 1024, "KB;", len(D["t1"]["holds"]), "T1 holds;", len(D["still"]["holds"]), "still rows")


if __name__ == "__main__":
    main()
