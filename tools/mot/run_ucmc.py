#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-31): UCMCTrack (common/mot/ucmc.py) on the 2D detections of KITTI-MOTS val, same evaluation
and output schema as run_tracker.py (needs the per-sequence KITTI calibration and oxts, which run_tracker.py does
not pass to trackers, hence a separate runner).

    python tools/mot/run_ucmc.py                    # every ~/mot_logs/dets/kittimots/<det>.npz (not *_reid)
    python tools/mot/run_ucmc.py --dets cascade_rcnn --no-ego

Calibration / ego-motion: ~/Physical_AI/datasets/KITTI_tracking/training/{calib,oxts}/<seq>.txt (KITTI-MOTS val
sequences are KITTI tracking sequences).
Output: ~/mot_logs/track/kittimots_<det>/ucmctrack.json (ucmctrack_noego.json with --no-ego), summary.txt line.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, evaluate, load_kittimots  # noqa: E402
from common.mot.ucmc import DEFAULTS, UCMCTrack, kitti_mapper, oxts_poses  # noqa: E402


def read_calib(p):
    v = {}
    for line in Path(p).read_text().splitlines():
        if line.strip():
            k, *nums = line.replace(":", " ").split()
            v[k] = np.array(nums, float)
    return v["Tr_velo_cam"].reshape(3, 4), v["R_rect"].reshape(3, 3), v["P2"].reshape(3, 4), v["Tr_imu_velo"].reshape(3, 4)


def track(z, gt, kroot, kw=None, ego_on=True, tms=None):
    """-> predictions {seq: [frame arrays]} for one detections npz; kw overrides UCMCTrack parameters."""
    pred = {}
    for seq, frames in gt.items():
        V2C, R0, P2, Tiv = read_calib(kroot / "calib" / f"{seq}.txt")
        InvA = kitti_mapper(V2C, R0, P2)
        poses = oxts_poses(kroot / "oxts" / f"{seq}.txt", Tiv)
        rows = z[seq]
        trk = {c: UCMCTrack(InvA, frame_rate=10, **(kw or {})) for c in (0, 1)}
        out = []
        for f in range(len(frames)):
            ego = None if (not ego_on or f == 0) else np.linalg.inv(poses[f]) @ poses[f - 1]
            part = []
            for c, t in trk.items():
                d = rows[(rows[:, 0] == f) & (rows[:, 6] == c)][:, 1:7]
                t0 = time.perf_counter()
                o = t.update(d, ego)
                if tms is not None:
                    tms.append(1000 * (time.perf_counter() - t0))
                if len(o):
                    o[:, 4] += c * 100000
                    o[:, 5] = c
                    part.append(o)
            out.append(np.concatenate(part) if part else np.zeros((0, 7)))
        pred[seq] = out
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", nargs="*", default=None)
    ap.add_argument("--src", default="~/mot_logs/dets/kittimots")
    ap.add_argument("--kitti", default="~/Physical_AI/datasets/KITTI_tracking/training")
    ap.add_argument("--out", default="~/mot_logs/track")
    ap.add_argument("--no-ego", action="store_true")
    ap.add_argument("--params", default=None, help="json file of tuned parameters (tools/mot/tune_ucmc.py)")
    ap.add_argument("--name", default=None, help="tracker name in the results (default ucmctrack[_noego])")
    a = ap.parse_args()
    kw = json.loads(Path(a.params).expanduser().read_text())["params"] if a.params else None
    src, kroot = Path(a.src).expanduser(), Path(a.kitti).expanduser()
    names = a.dets or sorted(p.stem for p in src.glob("*.npz") if not p.stem.endswith("_reid"))
    gt, _ = load_kittimots(C.DATASETS_ROOT / "KITTI_MOTS_VOC", "val")
    tname = a.name or ("ucmctrack_noego" if a.no_ego else "ucmctrack")
    for name in names:
        z = np.load(src / f"{name}.npz", allow_pickle=False)
        tms = []
        pred = track(z, gt, kroot, kw, not a.no_ego, tms)
        res = evaluate(gt, pred, KITTI_CLASSES)
        d = np.concatenate([z[f"{s}__ms"] for s in gt])
        trk_ms = float(np.median(tms)) * 2
        rec = {"tracker": tname, "run": f"kittimots_{name}", "source": "kittimots", "split": "val",
               "detections": {"version": name}, "params": dict(DEFAULTS, **(kw or {}), ego_motion=not a.no_ego,
                                                   tuned_on=a.params),
               "tracker_ms": {"median": trk_ms, "p95": float(np.percentile(tms, 95)) * 2, "frames": len(tms) // 2},
               "detector_ms": {"median": float(np.median(d)), "p95": float(np.percentile(d, 95))}, "reid_ms": None,
               "score_thr": DEFAULTS["conf_thresh"], "causal_check": None, **res}
        o = Path(a.out).expanduser() / f"kittimots_{name}"
        o.mkdir(parents=True, exist_ok=True)
        (o / f"{tname}.json").write_text(json.dumps(rec, indent=1))
        m = res["mean"]
        line = (f"{tname:<16} HOTA {m['HOTA']:.4f}  DetA {m['DetA']:.4f}  AssA {m['AssA']:.4f}  MOTA {m['MOTA']:.4f}  "
                f"IDF1 {m['IDF1']:.4f}  IDSW {m['IDSW']:>5}  trk {trk_ms:.2f} ms")
        with open(o / "summary.txt", "a") as fh:
            fh.write(f"kittimots_{name}  ({time.strftime('%F %T')})\n{line}\n")
        print(f"{name:<16} {line}", flush=True)


if __name__ == "__main__":
    main()
