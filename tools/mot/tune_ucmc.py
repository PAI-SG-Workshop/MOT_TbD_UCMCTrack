#!/usr/bin/env python3
"""
MOT v1.1 (2026-10-31): choose KITTI parameters for UCMCTrack on the KITTI-MOTS TRAIN sequences (never val).

The official KITTI configuration of UCMCTrack is not released and its general defaults come from MOT17
(pedestrians, 30 fps). This grid search picks vmax, wx = wy, the assignment threshold a (= a1 = a2) and conf_thresh
by mean HOTA over the train detections of three detectors (x, cascade_rcnn, v6 - dumped for CAMEL), ego-motion on.
    python tools/mot/tune_ucmc.py                       # -> ~/mot_logs/ucmc_kitti_params.json
    python tools/mot/run_ucmc.py --params ~/mot_logs/ucmc_kitti_params.json --name ucmctrack_kitti
"""

import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import config as C  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, evaluate, load_kittimots  # noqa: E402
from run_ucmc import track  # noqa: E402

DETS = ["x", "cascade_rcnn", "v6"]
GRID = {"vmax": [0.5, 2.0, 5.0, 10.0], "wx": [0.1, 1.0, 5.0], "a1": [10.0, 20.0, 40.0], "conf_thresh": [0.3, 0.5]}
KROOT = Path("~/Physical_AI/datasets/KITTI_tracking/training").expanduser()
SRC = Path("~/mot_logs/dets/kittimots_train").expanduser()
GT = None


def run(cfg):
    kw = dict(cfg, wy=cfg["wx"], a2=cfg["a1"])
    scores = []
    for d in DETS:
        z = np.load(SRC / f"{d}.npz", allow_pickle=False)
        scores.append(evaluate(GT, track(z, GT, KROOT, kw), KITTI_CLASSES)["mean"]["HOTA"])
    return kw, float(np.mean(scores)), scores


def main():
    global GT
    GT, _ = load_kittimots(C.DATASETS_ROOT / "KITTI_MOTS_VOC", "train")
    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(cfgs)} configurations x {len(DETS)} detectors on {len(GT)} train sequences", flush=True)
    t0, res = time.time(), []
    with Pool(8) as p:
        for k, (kw, m, s) in enumerate(p.imap_unordered(run, cfgs), 1):
            res.append((m, kw, s))
            if k % 8 == 0:
                print(f"  {k}/{len(cfgs)}  best so far {max(r[0] for r in res):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    res.sort(key=lambda r: -r[0])
    default = [r for r in res if r[1]["vmax"] == 0.5 and r[1]["wx"] == 0.1 and r[1]["a1"] == 10.0
               and r[1]["conf_thresh"] == 0.5]
    print("\ntop 5 (train HOTA mean | x, cascade_rcnn, v6):")
    for m, kw, s in res[:5]:
        print(f"  {m:.4f} | {np.round(s, 4).tolist()} | vmax {kw['vmax']} wx=wy {kw['wx']} a {kw['a1']} conf {kw['conf_thresh']}")
    if default:
        print(f"official defaults: {default[0][0]:.4f} | {np.round(default[0][2], 4).tolist()}")
    out = Path("~/mot_logs/ucmc_kitti_params.json").expanduser()
    out.write_text(json.dumps({"params": res[0][1], "train_hota": res[0][0], "per_detector": res[0][2],
                               "detectors": DETS, "grid": GRID, "selection": "mean HOTA, KITTI-MOTS train"}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
