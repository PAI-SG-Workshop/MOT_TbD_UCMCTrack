# MOT_TbD_UCMCTrack

**UCMCTrack** - tracking by detection on the ground plane (metres) for KITTI-MOTS: 25 detectors, official
defaults and KITTI parameters chosen on the train sequences.

- Paper: Yi et al., AAAI 2024; official code: `corfyi/UCMCTrack`
- Principle: the bottom-centre of every box is mapped to the road plane with the camera calibration; a constant-velocity
  Kalman filter and the "mapped Mahalanobis distance" associate in metres (high-score then low-score detections).
- Implementation: `common/mot/ucmc.py` (official KITTI mapping `P2 * R_rect * Tr_velo_cam`, ground 1.73 m below the
  LiDAR). `[Unified]`: camera motion is removed with the vehicle pose (oxts) instead of image GMC; 30 fps frame counts
  scaled to 10 fps; one tracker per class.
- **KITTI parameters**: the official KITTI configuration is not released; `tools/mot/tune_ucmc.py` searches vmax, wx = wy,
  the assignment threshold and conf_thresh on the KITTI-MOTS **train** sequences (detections of x, cascade_rcnn, v6) -
  never on val. Chosen: `{'vmax': 5.0, 'wx': 5.0, 'a1': 40.0, 'conf_thresh': 0.5, 'wy': 5.0, 'a2': 40.0}` (train HOTA 0.661). wx and a are at the edge of
  the grid; a wider grid may gain a little more.

### Results (HOTA, KITTI-MOTS val, car / pedestrian mean)

| detector | UCMCTrack (official defaults) | **UCMCTrack (KITTI, tuned on train)** | without ego-motion | best v1.0 tracker | pipeline ms |
|---|---|---|---|---|---|
| cascade_rcnn | 0.509 | **0.600** | 0.420 | tracktrack 0.625 | 35.2 |
| libra_rcnn | 0.475 | **0.564** | - | tracktrack 0.599 | 26.9 |
| faster_rcnn | 0.475 | **0.563** | - | tracktrack 0.582 | 25.4 |
| dynamic_rcnn | 0.464 | **0.548** | - | tracktrack 0.579 | 25.4 |
| atss | 0.440 | **0.519** | - | tracktrack 0.527 | 21.8 |
| gfl | 0.431 | **0.513** | - | hybridsort 0.522 | 20.9 |
| retinanet | 0.419 | **0.491** | - | botsort_reid 0.535 | 21.2 |
| v6 | 0.405 | **0.489** | 0.336 | botsort_reid 0.509 | 6.7 |
| tood | 0.391 | **0.480** | - | hybridsort 0.506 | 24.9 |
| fcos | 0.400 | **0.477** | - | botsort_reid 0.493 | 21.7 |
| ppyoloe | 0.379 | **0.466** | - | tracktrack 0.487 | 6.4 |
| x | 0.373 | **0.451** | - | botsort_reid 0.478 | 5.6 |
| rtmdet | 0.367 | **0.448** | - | hybridsort 0.465 | 6.3 |
| v9 | 0.345 | **0.432** | - | botsort_reid 0.460 | 12.0 |
| x2 | 0.337 | **0.428** | - | hybridsort 0.452 | 5.1 |
| v10 | 0.347 | **0.422** | - | botsort_reid 0.455 | 7.9 |
| deformable_detr | 0.345 | **0.416** | - | botsort_reid 0.482 | 68.4 |
| v11 | 0.327 | **0.404** | - | hybridsort 0.435 | 6.4 |
| v8 | 0.315 | **0.403** | - | botsort_reid 0.443 | 5.4 |
| v12 | 0.324 | **0.390** | - | hybridsort 0.424 | 8.5 |
| v13 | 0.262 | **0.357** | - | tracktrack 0.372 | 10.7 |
| v26 | 0.241 | **0.351** | - | hybridsort 0.390 | 8.9 |
| centernet | 0.267 | **0.339** | 0.257 | hybridsort 0.395 | 7.5 |
| v1 | 0.251 | **0.324** | - | tracktrack 0.374 | 5.9 |
| ssd | 0.205 | **0.265** | - | botsort 0.288 | 29.3 |

Main findings: the ego-motion compensation is essential on a moving car (+0.09 HOTA, -40 % identity switches on
cascade_rcnn); with KITTI parameters UCMCTrack comes close to the best trackers of v1.0 at well under 1 ms per frame on
the CPU, and it is the only tracker of the comparison whose output is a metric trajectory from a single camera.

## Usage

```bash
pip install -r requirements.txt
pip install "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
# KITTI tracking calib + oxts (training): https://www.cvlibs.net/datasets/kitti/eval_tracking.php
# saved detections of the 25 detectors: Release of Statera-Workshop/MOT_Detectors_KITTIMOTS
python tools/mot/run_ucmc.py                                    # official defaults
python tools/mot/tune_ucmc.py                                   # KITTI parameters from train detections
python tools/mot/run_ucmc.py --params ~/mot_logs/ucmc_kitti_params.json --name ucmctrack_kitti
```

Part of the *Physical AI MOT* comparison of the **Statera Working Group** (v1.1 extension: UCMCTrack, CAMELTrack,
Seg2Track-SAM2, AB3DMOT). The v1.0 comparison (10 trackers x 25 detectors, JDT, E2E) and the v1.1 summary are in
`MOT_Detectors_KITTIMOTS` (`RESULTS.md`).

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

Results of this repository: `results/` (evaluation JSON per detector). Sources and licences: `THIRD_PARTY.md`.

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
