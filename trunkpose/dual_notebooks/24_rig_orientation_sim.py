# %% 24_rig_orientation_sim.py
# Which orientation of the Radxa dual OV9281 rig gives the best torso depth?
#
# Simulation only, no recordings needed. The rig keeps its sep23 mounting position
# (upside down above the screen, torso ~1.1 m away). From there we vary:
#   pitch    whole-rig tilt; + aims the optical axis up the torso (toward the shoulders)
#   yaw      whole-rig pan; + aims toward the reference camera's +x
#   toe      per-camera convergence; each camera turns this many degrees toward the other
#   roll     0 = horizontal baseline (now), 90 = vertical baseline
#   baseline the ~2 cm of freedom: 77.5 -> 97.5 mm
# plus the pipeline knobs that are not orientation (rectified focal, sensor binning),
# so all options sit on one scale.
#
# The torso is an elliptic cylinder fitted to the sep23 geometry caches, in the raw
# reference-camera (cam1, physical left) frame: centroid ~(0.03, 0.08, 1.10) m, shell
# spanning y -0.03..0.35 m (+y is world UP because the rig hangs inverted), chest
# normal tilted ~10 deg. It goes through a pose set: neutral, forward flexion 15/30 deg,
# lateral lean +/-15 deg, axial rotation +/-25 deg, reach shifts x +/-0.12 m, z 0.85 m.
#
# Each config builds the real cv2.fisheye.stereoRectify (balance=0, as 06 does) from
# the sep18 intrinsics. A torso point counts when it faces both cameras, lies inside
# both source images, and lands inside both rectified images. Per point, the depth
# noise is
#   sigma_Z = Z^2 / (f_rect * B) * sigma_d * max(1, m)
# where sigma_d is the SGBM disparity noise in rectified px and m is the local
# rectified-px-per-source-px stretch along the epipolar direction (a stretched
# periphery has no more information than the source pixels behind it).
# Figures of merit per config, averaged over the pose set:
#   cover    fraction of the front torso surface usable for stereo
#   npx      torso pixels in the rectified reference image (~ shell npts, 06/23's
#            confidence signal), discounted where rectification only upsamples
#   sigZ     median per-point depth noise along the viewing ray, mm
#   tilt     plane-normal noise floor sig_n / (sqrt(npx) * spread), deg, with sig_n
#            the noise along the chest normal -- the number
#            the ICP / trunk-axis fit actually feels
#
# Run: python trunkpose/dual_notebooks/24_rig_orientation_sim.py
# Outputs land in trunkpose/dual_notebooks/outputs/24_rig_orientation_sim/.
import itertools
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIB = PROJECT_ROOT / "data/calibration/dual_160/calib_radxa_sep18_26/stereo_calibration.toml"
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "24_rig_orientation_sim"

SIGMA_D_PX = 0.25      # SGBM sub-pixel disparity noise, rectified px
FOV_LIMIT_DEG = 80.0   # usable fisheye half-angle
SCALE = 0.5            # sep23 records 640x400 binned
B_NOW_MM = 77.45

# Torso model (camera frame, metres). Matches the sep23 shell clouds.
TORSO_A, TORSO_B = 0.17, 0.11          # lateral / antero-posterior half-axes
TORSO_Y0, TORSO_Y1 = -0.05, 0.37       # waist .. shoulders along +y
TORSO_X, TORSO_FRONT_Z = 0.03, 1.04    # centre x, nearest surface depth
TORSO_TILT_DEG = 10.0                  # top leans away, chest faces ~10 deg up
N_THETA, N_H = 160, 90


# %% Camera model
def load_cams(scale=SCALE):
    c = toml.load(CALIB)
    S = np.diag([scale, scale, 1.0])
    # swap: cam1 is the physical left camera and 06's reference
    K_ref = S @ np.array(c["cam1"]["camera_matrix"])
    K_oth = S @ np.array(c["cam0"]["camera_matrix"])
    D_ref = np.array(c["cam1"]["dist_coeffs"], float).reshape(4)
    D_oth = np.array(c["cam0"]["dist_coeffs"], float).reshape(4)
    size = (int(round(1280 * scale)), int(round(800 * scale)))
    return K_ref, D_ref, K_oth, D_oth, size


def kb_project(Xc, K, D):
    """Kannala-Brandt (cv2.fisheye) projection of camera-frame points (N,3)."""
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    r = np.hypot(x, y)
    th = np.arctan2(r, z)
    th2 = th * th
    thd = th * (1 + D[0] * th2 + D[1] * th2**2 + D[2] * th2**3 + D[3] * th2**4)
    s = np.where(r > 1e-12, thd / np.maximum(r, 1e-12), 1.0 / np.maximum(z, 1e-12))
    u = K[0, 0] * x * s + K[0, 2]
    v = K[1, 1] * y * s + K[1, 2]
    return np.stack([u, v], 1), np.degrees(th)


def rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) * np.radians(deg))[0]


def build_rig(pitch=0.0, yaw=0.0, toe=0.0, roll=0.0, baseline_mm=B_NOW_MM):
    """Camera centres + camera-to-world rotations, world = current reference-cam frame.
    The rig turns about its baseline midpoint, so the cameras stay where they are."""
    b = baseline_mm / 1000.0
    mid = np.array([B_NOW_MM / 2000.0, 0, 0])
    # +pitch aims up the torso (+y); Rx(+a) would swing z toward -y
    R_rig = rot([0, 1, 0], yaw) @ rot([1, 0, 0], -pitch) @ rot([0, 0, 1], roll)
    C_ref = mid + R_rig @ np.array([-b / 2, 0, 0])
    C_oth = mid + R_rig @ np.array([b / 2, 0, 0])
    Rcw_ref = R_rig @ rot([0, 1, 0], toe)    # left camera turns toward +x
    Rcw_oth = R_rig @ rot([0, 1, 0], -toe)
    return (C_ref, Rcw_ref), (C_oth, Rcw_oth)


# %% Torso + poses
def torso_surface():
    """Points, outward normals and area weights of the cylinder side, posed neutral."""
    t = np.linspace(-np.pi, np.pi, N_THETA, endpoint=False)
    h = np.linspace(TORSO_Y0, TORSO_Y1, N_H)
    T, H = np.meshgrid(t, h)
    x = TORSO_A * np.sin(T)
    z = -TORSO_B * np.cos(T)                  # theta = 0 faces the camera (-z)
    P = np.stack([x, H, z], -1).reshape(-1, 3)
    n = np.stack([np.sin(T) / TORSO_A, 0 * T, -np.cos(T) / TORSO_B], -1).reshape(-1, 3)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    # local frame: origin on the axis at waist height; tilt so the top leans back
    R0 = rot([1, 0, 0], TORSO_TILT_DEG)
    origin = np.array([TORSO_X, 0.0, TORSO_FRONT_Z + TORSO_B])
    return P, n, R0, origin


POSES = {
    "neutral": dict(),
    "flex15": dict(flex=15), "flex30": dict(flex=30),
    "lean+15": dict(lean=15), "lean-15": dict(lean=-15),
    "axial+25": dict(axial=25), "axial-25": dict(axial=-25),
    "reach_x-12cm": dict(dx=-0.12), "reach_x+12cm": dict(dx=0.12),
    "near_z0.85": dict(dz=-0.19),
    "flex30_axial25": dict(flex=30, axial=25),
}


def pose_torso(P, n, R0, origin, flex=0, lean=0, axial=0, dx=0.0, dz=0.0):
    # pivot at the hips (waist line, y = TORSO_Y0 - 0.1). Forward flexion brings the top
    # toward the camera (-z) and the screen.
    Rp = rot([0, 0, 1], lean) @ rot([1, 0, 0], -flex) @ R0 @ rot([0, 1, 0], axial)
    hip = np.array([0, TORSO_Y0 - 0.10, 0])
    Pw = (P - hip) @ Rp.T + hip + origin + np.array([dx, 0, dz])
    return Pw, n @ Rp.T


# %% Per-config evaluation
def evaluate(cfg, cams, surf, sigma_d=SIGMA_D_PX, fov_scale=1.0):
    K_ref, D_ref, K_oth, D_oth, size = cams
    W, Hh = size
    (C_r, Rr), (C_o, Ro) = build_rig(**cfg)
    # OpenCV stereo: X_oth = R X_ref + T
    R_st = Ro.T @ Rr
    T_st = (Ro.T @ (C_r - C_o)).reshape(3, 1) * 1000.0
    R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
        K_ref, D_ref.reshape(4, 1), K_oth, D_oth.reshape(4, 1), size, R_st, T_st,
        flags=cv2.CALIB_ZERO_DISPARITY, balance=0.0, fov_scale=fov_scale)
    f_rect = P1[0, 0]
    B = np.linalg.norm(T_st) / 1000.0
    # rectified epipolar direction in the reference rect image (x for horizontal stereo,
    # y when stereoRectify decides it is vertical)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    ax = 1 if vertical else 0

    P, n, R0, origin = surf
    out = []
    for pname, pp in POSES.items():
        Pw, nw = pose_torso(P, n, R0, origin, **pp)
        ok = np.ones(len(Pw), bool)
        rect = {}
        for key, (C, Rcw, K, D, Rr_rect, Pr) in {
            "ref": (C_r, Rr, K_ref, D_ref, R1, P1),
            "oth": (C_o, Ro, K_oth, D_oth, R2, P2),
        }.items():
            Xc = (Pw - C) @ Rcw
            facing = np.einsum("ij,ij->i", nw, C - Pw) > 0
            uv, th = kb_project(Xc, K, D)
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
            Xr = Xc @ Rr_rect.T
            front = Xr[:, 2] > 1e-6
            ur = (Xr[:, :2] / np.where(front, Xr[:, 2], 1)[:, None]) * Pr[[0, 1], [0, 1]] + Pr[:2, 2]
            in_rect = front & (ur[:, 0] >= 0) & (ur[:, 0] < W) & (ur[:, 1] >= 0) & (ur[:, 1] < Hh)
            ok &= facing & inside & in_rect & (th < FOV_LIMIT_DEG)
            rect[key] = (Xc, Xr, ur, uv)
        # front shell = surface facing the reference camera (what segmentation can see)
        Xc_r = (Pw - C_r) @ Rr
        front_shell = np.einsum("ij,ij->i", nw, C_r - Pw) > 0
        uv_src, _ = kb_project(Xc_r, K_ref, D_ref)
        front_shell &= (uv_src[:, 0] >= 0) & (uv_src[:, 0] < W) & (uv_src[:, 1] >= 0) & (uv_src[:, 1] < Hh)
        cover = ok.sum() / max(front_shell.sum(), 1)
        if ok.sum() < 20:
            out.append(dict(pose=pname, cover=cover, npx=0, sigZ=np.nan, tilt=np.nan))
            continue

        Xc, Xr, ur, uv = (a[ok] for a in rect["ref"])
        nv = nw[ok]
        # local stretch: rect px per source px along each rectified axis, ref cam
        ray = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
        stretch = []
        for k in (0, 1):
            e_rect = np.zeros(3)
            e_rect[k] = 1.0
            e_cam = R1.T @ e_rect                # rect axis in source-camera frame
            dX = e_cam - ray * (ray @ e_cam)[:, None]
            dX *= (1e-4 * np.linalg.norm(Xc, axis=1) / np.linalg.norm(dX, axis=1))[:, None]
            uv2, _ = kb_project(Xc + dX, K_ref, D_ref)
            Xr2 = (Xc + dX) @ R1.T
            ur2 = Xr2[:, :2] / Xr2[:, 2:3] * P1[[0, 1], [0, 1]] + P1[:2, 2]
            stretch.append(np.linalg.norm(ur2 - ur, axis=1)
                           / np.maximum(np.linalg.norm(uv2 - uv, axis=1), 1e-12))
        m = stretch[ax]
        Zr = Xr[:, 2]
        # rectified pinhole: the error vector lies along the ray, |dP| = dZ * range / Z
        rng_ = np.linalg.norm(Xr, axis=1)
        sig_ray = Zr**2 / (f_rect * B) * sigma_d * np.maximum(1.0, m) * rng_ / Zr
        # what the plane / ICP feels: the component along the surface normal
        cos_n = np.abs(np.einsum("ij,ij->i", nv, ray))
        sig_n = sig_ray * cos_n
        # torso px in the rectified reference image, discounted where rectification
        # stretches the source (those extra px carry no extra information)
        hull = cv2.convexHull(ur.astype(np.float32))
        info = np.minimum(1.0, 1.0 / (stretch[0] * stretch[1]))
        npx = cv2.contourArea(hull) * float(np.mean(info))
        spread = np.sqrt(np.trace(np.cov(Xr[:, :2].T)))
        tilt = np.degrees(np.median(sig_n) / (np.sqrt(npx) * spread))
        out.append(dict(pose=pname, cover=cover, npx=npx, sigZ=np.median(sig_ray) * 1000,
                        tilt=tilt, m=np.median(m)))
    agg = {k: float(np.nanmean([o[k] for o in out])) for k in ("cover", "npx", "sigZ", "tilt")}
    agg["worst_cover"] = float(min(o["cover"] for o in out))
    agg["f_rect"] = float(f_rect)
    agg["vertical"] = bool(vertical)
    return agg, out


# %% Sweeps
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cams = load_cams()
    surf = torso_surface()
    base, per_pose = evaluate({}, cams, surf)
    print(f"current rig: f_rect={base['f_rect']:.1f}px  cover={base['cover']:.3f}  "
          f"npx={base['npx']:.0f}  sigZ={base['sigZ']:.2f}mm  tilt={base['tilt']:.3f}deg")
    for o in per_pose:
        print(f"   {o['pose']:<16} cover={o['cover']:.3f} npx={o['npx']:6.0f} "
              f"sigZ={o['sigZ']:5.2f}mm  m={o.get('m', np.nan):.2f}")

    rows = []

    def run(group, label, cfg=None, fs=1.0, **kw):
        a, _ = evaluate(cfg or {}, cams, surf, fov_scale=fs, **kw)
        a.update(group=group, label=label, fs=fs)
        rows.append(a)
        return a

    # orientation sweeps twice: with 06's rectification (fov_scale 1.0, which shrinks
    # the centre focal 299 -> 245 px) and with fov_scale 0.82 (centre kept 1:1)
    for fs in (1.0, 0.82):
        for p in range(-25, 26, 5):
            run("pitch", f"{p:+d}", dict(pitch=p), fs)
        for y in range(-20, 21, 5):
            run("yaw", f"{y:+d}", dict(yaw=y), fs)
        for t in (-4, -2, 0, 1, 2, 3, 5, 8, 12):
            run("toe", f"{t:+d}", dict(toe=t), fs)
        for r in (0, 90):
            run("roll", f"{r}", dict(roll=r), fs)
        for b in (77.45, 87.5, 97.5):
            run("baseline", f"{b:.0f}mm", dict(baseline_mm=b), fs)
    for fs in (1.0, 0.9, 0.82, 0.7):
        run("fov_scale", f"{fs}", fs=fs)

    # joint grid pitch x toe, 97.5 mm baseline, fov_scale 0.82
    pitches = list(range(-20, 21, 5))
    toes = [-2, 0, 2, 4, 6, 8]
    G = np.zeros((len(toes), len(pitches)))
    for (i, t), (j, p) in itertools.product(enumerate(toes), enumerate(pitches)):
        G[i, j] = evaluate(dict(pitch=p, toe=t, baseline_mm=97.5), cams, surf,
                           fov_scale=0.82)[0]["tilt"]

    # recommended combo vs now
    best = [
        ("now", {}, {}),
        ("+2cm baseline", dict(baseline_mm=97.5), {}),
        ("fov_scale 0.82", {}, dict(fov_scale=0.82)),
        ("+2cm + fov_scale 0.82", dict(baseline_mm=97.5), dict(fov_scale=0.82)),
        ("+2cm + fov 0.82 + pitch+8", dict(baseline_mm=97.5, pitch=8), dict(fov_scale=0.82)),
    ]
    combo = []
    for name, cfg, kw in best:
        a, _ = evaluate(cfg, cams, surf, **kw)
        combo.append((name, a))
    full = load_cams(1.0)
    for name, cfg, kw in (("full-res 1280x800", {}, {}),
                          ("full-res + 2cm + fov 0.82", dict(baseline_mm=97.5),
                           dict(fov_scale=0.82, sigma_d=SIGMA_D_PX))):
        a, _ = evaluate(cfg, full, surf, **kw)
        combo.append((name, a))

    # ---- report
    header = (f"{'group':<10}{'label':>9}{'fov_sc':>7}{'f_rect':>8}{'cover':>7}{'worst':>7}"
              f"{'npx':>7}{'sigZ_mm':>9}{'tilt_deg':>10}{'vs_now':>8}")
    lines = [header]
    for r in rows:
        lines.append(f"{r['group']:<10}{r['label']:>9}{r['fs']:7.2f}{r['f_rect']:8.1f}{r['cover']:7.3f}"
                     f"{r['worst_cover']:7.3f}{r['npx']:7.0f}{r['sigZ']:9.2f}{r['tilt']:10.4f}"
                     f"{base['tilt'] / r['tilt']:8.2f}x")
    lines.append("")
    lines.append("combos (tilt noise floor, higher vs_now = better):")
    for name, a in combo:
        lines.append(f"  {name:<28} sigZ={a['sigZ']:5.2f}mm npx={a['npx']:6.0f} "
                     f"tilt={a['tilt']:.4f}deg  {base['tilt'] / a['tilt']:.2f}x")
    txt = "\n".join(lines)
    print(txt)
    (OUT_DIR / "summary.txt").write_text(txt, encoding="utf-8")

    # ---- figures
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    for ax, g, xl in zip(axs.flat, ("pitch", "yaw", "toe", "baseline", "fov_scale"),
                         ("rig pitch (deg, + = aim up torso)", "rig yaw (deg)",
                          "toe-in per camera (deg)", "baseline", "rectify fov_scale")):
        for fs, mk in ((1.0, "o-"), (0.82, "s-")):
            rr = [r for r in rows if r["group"] == g and (g == "fov_scale" or r["fs"] == fs)]
            x = np.arange(len(rr))
            ax.plot(x, [base["tilt"] / r["tilt"] for r in rr], mk,
                    label=f"precision gain, fov_scale {fs}")
            ax.plot(x, [r["worst_cover"] for r in rr], "^:", color=ax.lines[-1].get_color(),
                    alpha=0.6, label=f"worst-pose torso coverage, fov_scale {fs}")
            if g == "fov_scale":
                break
        ax.axhline(1.0, color="0.6", lw=0.8)
        ax.set_xticks(x, [r["label"] for r in rr], rotation=45)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 2.0)
        ax.grid(alpha=0.3)
    axs.flat[0].legend(fontsize=8)
    ax = axs.flat[5]
    G = base["tilt"] / G
    im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pitches)), pitches)
    ax.set_yticks(range(len(toes)), toes)
    ax.set_xlabel("pitch (deg)")
    ax.set_ylabel("toe-in (deg)")
    ax.set_title("gain vs now: 97.5 mm, fov_scale 0.82")
    for (i, j), v in np.ndenumerate(G):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
    fig.colorbar(im, ax=ax)
    fig.suptitle("Radxa dual rig orientation sweep (sep23 framing, 640x400, torso ~1.1 m)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sweeps.png", dpi=130)
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
