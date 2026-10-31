"""
MOT v1.1 (2026-10-31): UCMCTrack - uniform camera motion compensation (Yi et al., AAAI 2024, corfyi/UCMCTrack).

Detections are mapped from the image to the ground plane and tracked there in metres:
  mapping     bottom-centre of the box (u, v) -> ground (x, y) through A = [KiKo_1, KiKo_2, z0*KiKo_3 + KiKo_4]
              (KITTI: KiKo = P2 * R_rect * Tr_velo_cam, ground z0 = -1.73 m in the LiDAR frame - official
              readKittiCalib / Mapper); pixel noise sigma_u = sigma_v = clip(0.05 * box height, 2, 13 / 10) px is
              propagated to a ground covariance R (Jacobian of the mapping)
  motion      Kalman filter [x, vx, y, vy], constant velocity, process noise G diag(wx, wy) G^T
  association "mapped Mahalanobis distance" d^T S^-1 d + log det S; high-score detections (>= high_score) with
              confirmed + coasted tracks (threshold a1), low-score ones with the rest (a2), then tentative tracks (a1);
              a tentative track is confirmed after 2 more matches; coasted tracks die after cdt frames
  output      confirmed tracks matched in the frame, with the detection box (official online output, no --hp)
Official defaults (util/run_ucmc.py): wx 0.1, wy 0.1, vmax 0.5, a 10, cdt 30 frames at 30 fps, high_score 0.6,
conf_thresh 0.5.  The official KITTI script is not released.
[Unified] decisions:
  * cdt 30 frames at 30 fps -> 10 at 10 fps (the framework's frame-rate rule), dt = 1/10 s.
  * camera motion: official --cmc warps tracks with image GMC; KITTI gives the vehicle pose (oxts), so the track
    states are moved into the current LiDAR frame with the exact ego-motion instead (position, velocity, covariance).
  * one tracker per class, like every tracker here.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

DEFAULTS = dict(wx=0.1, wy=0.1, vmax=0.5, a1=10.0, a2=10.0, cdt_30fps=30, high_score=0.6, conf_thresh=0.5)
Z0 = -1.73


def kitti_mapper(V2C, R0, P2):
    Rr, Tv = np.eye(4), np.eye(4)
    Rr[:3, :3], Tv[:3, :4] = R0, V2C
    KiKo = P2 @ Rr @ Tv
    A = np.zeros((3, 3))
    A[:, :2] = KiKo[:, :2]
    A[:, 2] = Z0 * KiKo[:, 2] + KiKo[:, 3]
    return np.linalg.inv(A)


def map_box(InvA, box):
    """box x1 y1 x2 y2 -> ground y (2,1), covariance R (2,2), or (None, None) above the horizon."""
    w, h = box[2] - box[0], box[3] - box[1]
    uv1 = np.array([[(box[0] + box[2]) / 2], [box[3]], [1.0]])
    b = InvA @ uv1
    if b[2, 0] <= 1e-9:
        return None, None
    gamma = 1 / b[2, 0]
    C = gamma * InvA[:2, :2] - (gamma ** 2) * b[:2] * InvA[2, :2]
    su, sv = np.clip(0.05 * h, 2, 13), np.clip(0.05 * h, 2, 10)
    return b[:2] * gamma, C @ np.diag([su * su, sv * sv]) @ C.T


def lap_limit(cost, limit):
    """lap.lapjv(extend_cost=True, cost_limit=limit) equivalent: leaving a row / column unmatched costs limit / 2."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.empty((0, 2), int), np.arange(n), np.arange(m)
    big = 1e9
    M = np.full((n + m, n + m), big)
    M[:n, :m] = cost
    M[:n, m:][np.eye(n, dtype=bool)] = limit / 2
    M[n:, :m][np.eye(m, dtype=bool)] = limit / 2
    M[n:, m:] = 0
    r, c = linear_sum_assignment(M)
    pairs = np.array([(i, j) for i, j in zip(r, c) if i < n and j < m and cost[i, j] < big], int).reshape(-1, 2)
    return pairs, np.setdiff1d(np.arange(n), pairs[:, 0]), np.setdiff1d(np.arange(m), pairs[:, 1])


class _KT:
    def __init__(self, tid, y, R, wx, wy, vmax, dt):
        self.id, self.dt = tid, dt
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], float)
        G = np.array([[0.5 * dt * dt, 0], [dt, 0], [0, 0.5 * dt * dt], [0, dt]])
        self.Q = G @ np.diag([wx, wy]) @ G.T
        self.P = np.diag([1, vmax ** 2 / 3.0, 1, vmax ** 2 / 3.0])
        self.x = np.array([[y[0, 0]], [0.0], [y[1, 0]], [0.0]])
        self.status, self.death, self.birth, self.detidx = "tentative", 0, 0, -1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, y, R):
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def distance(self, y, R):
        d = y - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        return float((d.T @ np.linalg.inv(S) @ d)[0, 0]) + float(np.log(np.linalg.det(S)))

    def ego(self, T):
        """move the state into the new LiDAR frame: T = 4x4 transform old -> new."""
        Rz, t = T[:2, :2], T[:2, 3]
        J = np.zeros((4, 4))
        J[np.ix_([0, 2], [0, 2])] = Rz
        J[np.ix_([1, 3], [1, 3])] = Rz
        pos = Rz @ self.x[[0, 2], 0] + t
        self.x = J @ self.x
        self.x[0, 0], self.x[2, 0] = pos
        self.P = J @ self.P @ J.T


class UCMCTrack:
    def __init__(self, InvA, frame_rate=10, **kw):
        p = dict(DEFAULTS, **kw)
        self.InvA, self.dt = InvA, 1.0 / frame_rate
        self.wx, self.wy, self.vmax, self.a1, self.a2 = p["wx"], p["wy"], p["vmax"], p["a1"], p["a2"]
        self.max_age = max(1, int(round(p["cdt_30fps"] * frame_rate / 30.0)))
        self.high, self.conf = p["high_score"], p["conf_thresh"]
        self.trackers, self.count = [], 0

    def _cost(self, di, ti, Y):
        C = np.zeros((len(di), len(ti)))
        for a, i in enumerate(di):
            for b, j in enumerate(ti):
                C[a, b] = self.trackers[j].distance(*Y[i])
        return C

    def update(self, dets, ego=None):
        """dets [n, 6] x1 y1 x2 y2 score cls; ego = 4x4 LiDAR transform previous -> current frame (or None)."""
        dets = np.asarray(dets, float).reshape(-1, 6)
        dets = dets[(dets[:, 4] > self.conf) & ~(((dets[:, 2] - dets[:, 0]) <= 10) & ((dets[:, 3] - dets[:, 1]) <= 10))]
        Y = [map_box(self.InvA, d[:4]) for d in dets]
        ok = [i for i, y in enumerate(Y) if y[0] is not None]
        for t in self.trackers:
            if ego is not None:
                t.ego(ego)
            t.predict()
            t.detidx = -1
        high = [i for i in ok if dets[i, 4] >= self.high]
        low = [i for i in ok if dets[i, 4] < self.high]
        conf_ = [k for k, t in enumerate(self.trackers) if t.status in ("confirmed", "coasted")]
        remain_d, remain_t = list(high), list(conf_)
        if high and conf_:
            m, ud, ut = lap_limit(self._cost(high, conf_, Y), self.a1)
            for a, b in m:
                self._hit(conf_[b], high[a], Y, "confirmed")
            remain_d, remain_t = [high[a] for a in ud], [conf_[b] for b in ut]
        if low and remain_t:
            m, ud, ut = lap_limit(self._cost(low, remain_t, Y), self.a2)
            for b in ut:
                self.trackers[remain_t[b]].status = "coasted"
            for a, b in m:
                self._hit(remain_t[b], low[a], Y, "confirmed")
        tent = [k for k, t in enumerate(self.trackers) if t.status == "tentative"]
        if remain_d and tent:
            m, ud, ut = lap_limit(self._cost(remain_d, tent, Y), self.a1)
            for a, b in m:
                t = self.trackers[tent[b]]
                self._hit(tent[b], remain_d[a], Y, None)
                t.birth += 1
                if t.birth >= 2:
                    t.birth, t.status = 0, "confirmed"
            remain_d = [remain_d[a] for a in ud]
        for i in remain_d:
            self.count += 1
            t = _KT(self.count, Y[i][0], Y[i][1], self.wx, self.wy, self.vmax, self.dt)
            t.detidx = i
            self.trackers.append(t)
        keep = []
        for t in self.trackers:
            t.death += 1
            if not ((t.status == "coasted" and t.death >= self.max_age) or (t.status == "tentative" and t.death >= 2)):
                keep.append(t)
        self.trackers = keep
        out = [[*dets[t.detidx, :4], t.id, 0, dets[t.detidx, 4]] for t in self.trackers
               if t.status == "confirmed" and t.detidx >= 0]
        return np.array(out, float).reshape(-1, 7)

    def _hit(self, k, i, Y, status):
        t = self.trackers[k]
        t.update(*Y[i])
        t.death, t.detidx = 0, i
        if status:
            t.status = status


# ---------------------------------------------------------------- KITTI ego-motion from oxts
def oxts_poses(oxts_file, Tr_imu_velo):
    """KITTI raw / tracking oxts -> LiDAR poses in a common world frame (devkit convention, Mercator)."""
    rows = np.loadtxt(oxts_file, ndmin=2)
    er, scale = 6378137.0, np.cos(rows[0, 0] * np.pi / 180.0)
    T_iv = np.eye(4)
    T_iv[:3, :4] = Tr_imu_velo
    poses = []
    for r in rows:
        lat, lon, alt, roll, pitch, yaw = r[:6]
        t = np.array([scale * lon * np.pi * er / 180.0, scale * er * np.log(np.tan((90 + lat) * np.pi / 360.0)), alt])
        cx, sx, cy, sy, cz, sz = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rz @ Ry @ Rx, t
        poses.append(T @ np.linalg.inv(T_iv))           # world <- velo
    return poses
