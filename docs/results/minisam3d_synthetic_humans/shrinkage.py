"""Shrinkage by teacher angle: on the unseen standard-movement takes of noark_v4's test split, each side's trunk is
zeroed on its take mean; frames are binned by |teacher angle| per rotation-vector component (camera axes: x flexion,
y axial, z lateral) and the median student / teacher ratio and median |error| are taken per bin.

Runs in the minisam3d venv on the GPU (from E:/NOARK_CV/minisam3d); writes the JSON the report's Figure 11 reads:
  .venv/Scripts/python <this> <out.json> r18_v2_all r18_v3_all r18_v3mix_all r18_v3mix_all_rep2 r18_v4mix_all
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "E:/NOARK_CV/minisam3d")
import numpy as np
from minisam3d.data import PseudoLabelDataset
from scipy.spatial.transform import Rotation
from scripts.eval_teacher import load_student
from scripts.eval_trunk_zeroed import trunk_rots, zero

D = "E:/NOARK_CV/minisam3d/"


def main():
    test = [
        r
        for r in json.loads(
            Path(D + "datasets/noark_v4.json").read_text(encoding="utf-8")
        )["splits"]["test"]
        if not r.endswith("_game")
    ]
    out = {}
    for ck in sys.argv[2:]:
        model, _ = load_student(D + f"runs/{ck}/best.pt", "cuda")
        P, T = [], []
        for r in test:
            ds = PseudoLabelDataset(D + "data/" + r, 256, 256, train=False, gray=True)
            Rp, Rt = trunk_rots(model, ds, 0, "cuda", 64, 4)
            P.append(Rotation.from_matrix(zero(Rp)).as_rotvec(degrees=True))
            T.append(Rotation.from_matrix(zero(Rt)).as_rotvec(degrees=True))
        P, T = np.concatenate(P), np.concatenate(T)
        out[ck] = {}
        for k, nm in ((0, "flexion"), (1, "axial"), (2, "lateral")):
            cells, rows = [], []
            for a, b in ((0, 5), (5, 10), (10, 20), (20, 90)):
                s = (np.abs(T[:, k]) >= a) & (np.abs(T[:, k]) < b)
                rows.append(
                    {
                        "lo": a,
                        "hi": b,
                        "n": int(s.sum()),
                        "ratio": float(np.median(P[s, k] / T[s, k])),
                        "err": float(np.median(np.abs(P[s, k] - T[s, k]))),
                    }
                )
                cells.append(
                    f"{a}-{b}: n {s.sum():5d} ratio {np.median(P[s, k] / T[s, k]):4.2f} "
                    f"err {np.median(np.abs(P[s, k] - T[s, k])):4.2f}"
                )
            out[ck][nm] = rows
            print(f"{ck:<15} {nm:<8} " + " | ".join(cells), flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
