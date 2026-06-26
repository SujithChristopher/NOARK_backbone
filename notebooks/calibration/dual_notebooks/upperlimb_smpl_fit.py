"""Upper limb SMPL fit — dual OV9281 stereo + MediaPipe PoseLandmarker + SMPL.

Pipeline
--------
1. Load, rectify, triangulate MediaPipe joints to metric 3D  (reuses kinematics)
2. Map MediaPipe joints -> SMPL joint targets (meters)
3. Fit SMPL per frame: optimize global_orient, transl, body_pose, betas
   via LBFGS on GPU, warm-started from previous frame for temporal stability
4. Render: [cam0 + projected SMPL verts | pyrender shaded mesh] -> MP4
5. Write per-frame SMPL params (betas, pose, transl) to .npz

Requirements (NOT in requirements.txt — install separately):
    pip install torch smplx trimesh pyrender

pyrender note: on Windows desktop it renders via native OpenGL (GPU).
On a headless box set  PYOPENGL_PLATFORM=egl  (or osmesa) before running.

SMPL model file (register at https://smpl.is.tue.mpg.de):
    Place SMPL_NEUTRAL.pkl at  <SMPL_MODEL_DIR>/smpl/SMPL_NEUTRAL.pkl
    Default SMPL_MODEL_DIR below = PROJECT_ROOT/data/body_models

Run:
    python notebooks/calibration/dual_notebooks/upperlimb_smpl_fit.py
"""

import os

import cv2
import numpy as np
import mediapipe as mp

import torch
import smplx
import trimesh
import pyrender

from upperlimb_3d_analysis import (
    PROJECT_ROOT, CALIB_TOML, DATA_DIR, CAM0_VIDEO, CAM1_VIDEO,
    CAM0_TIMESTAMP, CAM1_TIMESTAMP, CAM_SIZE, PANEL_HEIGHT,
    load_calib, build_rectify_maps, rectify,
    load_all_frames, load_timestamps, estimate_fps,
)
from upperlimb_3d_kinematics import (
    make_pose_landmarker, triangulate_joint, make_projector,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
# smplx.create(model_path, model_type=X) resolves <model_path>/<X>/<MODEL_FILE>.
# Repo root holds smplx/SMPLX_NEUTRAL.npz (and smpl/SMPL_NEUTRAL.pkl as fallback).
SMPL_MODEL_DIR = PROJECT_ROOT
MODEL_TYPE     = "smplx"        # "smplx" (body+hands+face) or "smpl"
MODEL_GENDER   = "neutral"
N_BODY_POSE    = 63 if MODEL_TYPE == "smplx" else 69   # 21 vs 23 body joints * 3
OUT_VIDEO      = DATA_DIR / "upperlimb_smpl_fit.mp4"
OUT_PARAMS     = DATA_DIR / "upperlimb_smpl_params.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VISIBILITY_THRESH = 0.5
Z_MAX_MM          = 4000

# Optimization
STAGE1_ITERS = 30    # global orient + translation only
STAGE2_ITERS = 60    # full body pose + betas
LR           = 1.0   # LBFGS step
W_BETA       = 1e-3  # shape regularization
W_POSE       = 5e-4  # pose regularization (toward rest pose)
W_TEMPORAL   = 1e-2  # penalize pose change between frames

VERT_DRAW_STRIDE = 12   # subsample SMPL verts when overlaying / scattering

# ---------------------------------------------------------------------------
# MediaPipe (BlazePose 33) -> SMPL (24-joint) correspondence
# SMPL joint order: 0 pelvis 1 L_hip 2 R_hip 4 L_knee 5 R_knee 7 L_ankle 8 R_ankle
#                   15 head 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow
#                   20 L_wrist 21 R_wrist
# MediaPipe idx: 0 nose 11/12 shoulders 13/14 elbows 15/16 wrists
#                23/24 hips 25/26 knees 27/28 ankles
# ---------------------------------------------------------------------------
# name -> (mediapipe_idx, smpl_joint_idx).  Pelvis handled specially (mid-hips).
# Upper-body only: legs are frozen, so knee/ankle targets are excluded (they would
# fight the fixed leg pose). Hips are kept — they anchor pelvis/global translation
# without depending on leg joint rotations.
JOINT_MAP = {
    "L_shoulder": (11, 16), "R_shoulder": (12, 17),
    "L_elbow":    (13, 18), "R_elbow":    (14, 19),
    "L_wrist":    (15, 20), "R_wrist":    (16, 21),
    "L_hip":      (23,  1), "R_hip":      (24,  2),
    "head":       (0,  15),
}
# Upper-limb joints get higher confidence weight (primary interest)
UPPER_NAMES = {"L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist"}

# Body-pose joints to OPTIMIZE (kinematic joint indices 1..21; pelvis=0 is global_orient).
# Everything not listed (hips, knees, ankles, feet) stays frozen at the rest pose.
#   3,6,9 = spine1/2/3   12 = neck   13,14 = collars   15 = head
#   16,17 = shoulders    18,19 = elbows    20,21 = wrists
_UPPER_BODY_JOINTS   = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
# body_pose param index for kinematic joint j is 3*(j-1)+k  (root excluded)
UPPER_POSE_PARAM_IDX = [3 * (j - 1) + k for j in _UPPER_BODY_JOINTS for k in range(3)]


# ---------------------------------------------------------------------------
# Triangulate a single MediaPipe landmark across the stereo pair
# ---------------------------------------------------------------------------
def landmark_px(lms, idx, W, H):
    """Return (px, py) clipped to image, or None if below visibility threshold."""
    if lms is None:
        return None
    lm = lms[idx]
    if lm.visibility < VISIBILITY_THRESH:
        return None
    return (int(np.clip(lm.x * W, 0, W - 1)),
            int(np.clip(lm.y * H, 0, H - 1)))


def triangulate_targets(lms0, lms1, Q, W, H):
    """Return (targets_m, weights, names) of joints visible in BOTH cameras.

    targets_m : (J, 3) float32 in METERS, SMPL coordinate convention
    weights   : (J,) float32
    smpl_idx  : (J,) int   SMPL joint indices the targets correspond to
    """
    tgt, wts, sidx = [], [], []
    hip_pts = {}
    for name, (mp_idx, smpl_j) in JOINT_MAP.items():
        p0 = landmark_px(lms0, mp_idx, W, H)
        p1 = landmark_px(lms1, mp_idx, W, H)
        if not (p0 and p1):
            continue
        xyz_mm = triangulate_joint(Q, p0[0], p0[1], p1[0], p1[1])
        if xyz_mm is None:
            continue
        xyz_m = xyz_mm.astype(np.float32) / 1000.0
        tgt.append(xyz_m)
        wts.append(2.0 if name in UPPER_NAMES else 1.0)
        sidx.append(smpl_j)
        if name in ("L_hip", "R_hip"):
            hip_pts[name] = xyz_m

    # Pelvis target = midpoint of both hips (SMPL joint 0), anchors global transl
    if "L_hip" in hip_pts and "R_hip" in hip_pts:
        tgt.append((hip_pts["L_hip"] + hip_pts["R_hip"]) / 2.0)
        wts.append(1.5)
        sidx.append(0)

    if not tgt:
        return None, None, None
    return (np.asarray(tgt, np.float32),
            np.asarray(wts, np.float32),
            np.asarray(sidx, np.int64))


# ---------------------------------------------------------------------------
# SMPL fitting
# ---------------------------------------------------------------------------
class SmplFitter:
    """Per-frame SMPL fit, warm-started from previous frame.

    Only the upper-body joints (UPPER_POSE_PARAM_IDX) are optimized; legs stay
    frozen at the rest pose. The free variable is `upper_pose`, scattered into a
    zero-initialized full body_pose each forward pass.
    """

    def __init__(self, model_dir, device):
        self.device = device
        kwargs = dict(
            model_path=str(model_dir),
            model_type=MODEL_TYPE,
            gender=MODEL_GENDER,
            num_betas=10,
            batch_size=1,
        )
        if MODEL_TYPE == "smplx":
            # Don't optimize hands/face — keep them fixed at a neutral flat pose.
            kwargs.update(use_pca=False, flat_hand_mean=True, use_face_contour=False)
        self.model = smplx.create(**kwargs).to(device)

        self.upper_idx = torch.as_tensor(UPPER_POSE_PARAM_IDX, device=device)

        z = lambda n: torch.zeros(1, n, device=device, requires_grad=True)  # noqa: E731
        self.global_orient = z(3)
        self.upper_pose    = z(len(UPPER_POSE_PARAM_IDX))   # only optimized joints
        self.betas         = z(10)
        self.transl        = z(3)
        self._prev_upper   = None   # detached upper_pose for temporal term

    def _body_pose(self):
        """Scatter upper_pose into a frozen (zero) full body_pose. Differentiable."""
        full = torch.zeros(1, N_BODY_POSE, device=self.device)
        return full.index_copy(1, self.upper_idx, self.upper_pose)

    def _forward(self):
        return self.model(
            betas=self.betas,
            body_pose=self._body_pose(),
            global_orient=self.global_orient,
            transl=self.transl,
        )

    def fit(self, targets_m, weights, smpl_idx):
        """Fit to 3D joint targets. Returns (vertices_m, body_joints_m)."""
        tgt = torch.as_tensor(targets_m, device=self.device)
        w   = torch.as_tensor(weights, device=self.device).unsqueeze(1)
        idx = torch.as_tensor(smpl_idx, device=self.device)

        def data_loss():
            out = self._forward()
            pred = out.joints[0, idx]            # (J, 3)
            return (w * (pred - tgt) ** 2).sum()

        # Stage 1: only global orient + translation (rigid alignment)
        opt1 = torch.optim.LBFGS(
            [self.global_orient, self.transl], lr=LR,
            max_iter=STAGE1_ITERS, line_search_fn="strong_wolfe")

        def closure1():
            opt1.zero_grad()
            loss = data_loss()
            loss.backward()
            return loss
        opt1.step(closure1)

        # Stage 2: upper-body pose + shape, with priors
        opt2 = torch.optim.LBFGS(
            [self.global_orient, self.transl, self.upper_pose, self.betas],
            lr=LR, max_iter=STAGE2_ITERS, line_search_fn="strong_wolfe")

        def closure2():
            opt2.zero_grad()
            loss = data_loss()
            loss = loss + W_BETA * (self.betas ** 2).sum()
            loss = loss + W_POSE * (self.upper_pose ** 2).sum()
            if self._prev_upper is not None:
                loss = loss + W_TEMPORAL * ((self.upper_pose - self._prev_upper) ** 2).sum()
            loss.backward()
            return loss
        opt2.step(closure2)

        self._prev_upper = self.upper_pose.detach().clone()

        with torch.no_grad():
            out = self._forward()
        return (out.vertices[0].cpu().numpy(),
                out.joints[0, :22].cpu().numpy())   # 22 body joints (SMPL-X / SMPL)

    def params(self):
        return {
            "global_orient": self.global_orient.detach().cpu().numpy().ravel(),
            "body_pose":     self._body_pose().detach().cpu().numpy().ravel(),
            "betas":         self.betas.detach().cpu().numpy().ravel(),
            "transl":        self.transl.detach().cpu().numpy().ravel(),
        }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def overlay_mesh_on_cam(bgr, verts_m, project, stride=VERT_DRAW_STRIDE):
    """Project SMPL vertices (meters) to cam0 and draw as green dots."""
    out = bgr.copy()
    for v in verts_m[::stride]:
        uv = project(v * 1000.0)   # projector expects mm
        if uv:
            cv2.circle(out, uv, 1, (60, 230, 90), -1, cv2.LINE_AA)
    return out


# OpenCV camera frame (+X right, +Y down, +Z forward) -> OpenGL/pyrender
# camera frame (+X right, +Y up, +Z backward). Flip Y and Z.
_CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


class MeshRenderer:
    """pyrender offscreen renderer viewing the SMPL mesh from cam0.

    Verts are in the cam0 (OpenCV) frame in meters, so we use cam0's
    rectified intrinsics — the shaded mesh aligns with the cam0 image.
    """

    def __init__(self, Q, size, faces):
        w, h = size
        self.w, self.h = int(w), int(h)
        self.faces = faces
        self.fx = self.fy = float(Q[2, 3])
        self.cx = float(-Q[0, 3])
        self.cy = float(-Q[1, 3])

        self.renderer = pyrender.OffscreenRenderer(self.w, self.h)
        self.cam = pyrender.IntrinsicsCamera(
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, znear=0.05, zfar=10.0)

        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1, roughnessFactor=0.7,
            baseColorFactor=(0.65, 0.74, 0.85, 1.0))

    def render(self, verts_m, target_size):
        scene = pyrender.Scene(bg_color=[0.07, 0.07, 0.07, 1.0],
                               ambient_light=[0.3, 0.3, 0.3])
        mesh = trimesh.Trimesh(vertices=verts_m, faces=self.faces, process=False)
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=self.material, smooth=True))
        scene.add(self.cam, pose=_CV_TO_GL)
        # Key light from camera-ish direction (also in GL frame)
        light = pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0)
        scene.add(light, pose=_CV_TO_GL)
        color, _ = self.renderer.render(scene)
        bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, target_size)

    def close(self):
        self.renderer.delete()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    sub = MODEL_TYPE                                  # "smplx" or "smpl"
    fname = f"{MODEL_TYPE.upper()}_{MODEL_GENDER.upper()}"
    cand = [SMPL_MODEL_DIR / sub / f"{fname}.npz",
            SMPL_MODEL_DIR / sub / f"{fname}.pkl"]
    if not any(p.exists() for p in cand):
        raise FileNotFoundError(
            f"{MODEL_TYPE} model not found. Looked for:\n  " +
            "\n  ".join(str(p) for p in cand)
        )

    print(f"Device: {DEVICE}")
    print("Loading calibration...")
    K0, D0, K1, D1, R, T = load_calib(CALIB_TOML)
    maps0, maps1, Q = build_rectify_maps(K0, D0, K1, D1, R, T, CAM_SIZE)
    project = make_projector(Q)

    print("Loading frames...")
    frames0 = load_all_frames(CAM0_VIDEO)
    frames1 = load_all_frames(CAM1_VIDEO)
    n = min(len(frames0), len(frames1))
    max_frames = int(os.environ.get("SMPL_MAX_FRAMES", "0"))   # 0 = all
    if max_frames > 0:
        n = min(n, max_frames)
    print(f"  {n} paired frames")

    ts0_ms = load_timestamps(CAM0_TIMESTAMP)
    ts1_ms = load_timestamps(CAM1_TIMESTAMP)
    if len(ts0_ms) != n or len(ts1_ms) != n:
        print("  Timestamp mismatch — synthetic 15 fps.")
        ts0_ms = [int(i * 1000 / 15) for i in range(n)]
        ts1_ms = list(ts0_ms)
    writer_fps = estimate_fps(ts0_ms)

    lmk0    = make_pose_landmarker()
    lmk1    = make_pose_landmarker()
    fitter  = SmplFitter(SMPL_MODEL_DIR, DEVICE)
    faces   = fitter.model.faces.astype(np.int64)
    mesh_renderer = MeshRenderer(Q, CAM_SIZE, faces)

    W, H        = CAM_SIZE
    cam_panel_w = int(W * PANEL_HEIGHT / H)
    mesh_w      = cam_panel_w               # mesh panel matches cam aspect (shared intrinsics)
    total_w     = cam_panel_w + mesh_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, writer_fps, (total_w, PANEL_HEIGHT))
    print(f"Output: {OUT_VIDEO}  ({total_w}x{PANEL_HEIGHT})")

    params_log = []

    for i in range(n):
        f0, f1 = frames0[i], frames1[i]
        f0_bgr = cv2.cvtColor(f0, cv2.COLOR_GRAY2BGR) if f0.ndim == 2 else f0
        f1_bgr = cv2.cvtColor(f1, cv2.COLOR_GRAY2BGR) if f1.ndim == 2 else f1

        r0 = rectify(f0_bgr, maps0)
        r1 = rectify(f1_bgr, maps1)

        res0 = lmk0.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(r0, cv2.COLOR_BGR2RGB)),
            ts0_ms[i])
        res1 = lmk1.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(r1, cv2.COLOR_BGR2RGB)),
            ts1_ms[i])
        lms0 = res0.pose_landmarks[0] if res0.pose_landmarks else None
        lms1 = res1.pose_landmarks[0] if res1.pose_landmarks else None

        targets, weights, smpl_idx = triangulate_targets(lms0, lms1, Q, W, H)

        if targets is not None and len(targets) >= 4:
            verts_m, joints_m = fitter.fit(targets, weights, smpl_idx)
            cam_panel  = overlay_mesh_on_cam(r0, verts_m, project)
            mesh_panel = mesh_renderer.render(verts_m, (mesh_w, PANEL_HEIGHT))
            params_log.append(fitter.params())
        else:
            cam_panel  = r0.copy()
            mesh_panel = np.zeros((PANEL_HEIGHT, mesh_w, 3), np.uint8)
            cv2.putText(mesh_panel, "no fit (too few joints)", (10, PANEL_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 200), 2, cv2.LINE_AA)

        cam_panel = cv2.resize(cam_panel, (cam_panel_w, PANEL_HEIGHT))
        cv2.putText(cam_panel, "Cam0 + SMPL verts", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(mesh_panel, "SMPL mesh", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(np.concatenate([cam_panel, mesh_panel], axis=1))

        if i % 25 == 0:
            njt = 0 if targets is None else len(targets)
            print(f"  {i:4d}/{n}  fit joints={njt}")

    writer.release()
    mesh_renderer.close()
    lmk0.__exit__(None, None, None)
    lmk1.__exit__(None, None, None)

    if params_log:
        np.savez(
            OUT_PARAMS,
            global_orient=np.stack([p["global_orient"] for p in params_log]),
            body_pose=np.stack([p["body_pose"] for p in params_log]),
            betas=np.stack([p["betas"] for p in params_log]),
            transl=np.stack([p["transl"] for p in params_log]),
        )
        print(f"Params -> {OUT_PARAMS}  ({len(params_log)} frames)")
    print(f"Done -> {OUT_VIDEO}")


if __name__ == "__main__":
    main()
