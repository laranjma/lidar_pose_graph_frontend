#!/usr/bin/env python3
"""
ROS2 Jazzy SLAM pose-graph pipeline with explicit Front-end / Back-end split.

Front-end responsibilities
- Subscribe to /odom and /scan
- Keyframe selection (thresholds)
- Build factors to add:
    * Prior on first node
    * Odometry between consecutive keyframes
    * (later) Loop-closure factors from a plug-in module
- Publish a "raw" keyframe path (from odom poses)

Back-end responsibilities
- Own the GTSAM graph/values and optimizer (iSAM2)
- Accept factor additions through a narrow interface
- Produce optimized estimates
- Publish an "optimized" path

Loop-closure plug-in interface (stub)
- Called when a new keyframe is created
- Can propose loop closure constraints (BetweenFactor Pose2 + noise)
- The front-end decides whether to accept and forwards to back-end

This file is designed to be dropped into a ROS2 Python package and run as a node.
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

import gtsam
from gtsam import symbol


# -----------------------------
# Utilities
# -----------------------------

def quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def pose2_to_ros_pose_stamped(p: gtsam.Pose2, frame_id: str, sec: int, nsec: int) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp.sec = sec
    ps.header.stamp.nanosec = nsec
    ps.pose.position.x = float(p.x())
    ps.pose.position.y = float(p.y())
    ps.pose.position.z = 0.0
    yaw = float(p.theta())
    ps.pose.orientation.z = math.sin(yaw / 2.0)
    ps.pose.orientation.w = math.cos(yaw / 2.0)
    return ps

def compute_pose_ls(scan_i: Optional[LaserScan], scan_j: Optional[LaserScan]) -> Optional[gtsam.Pose2]:
    """Estimate a relative pose measurement z_ij between two 2D laser scans.

    Implementation: lightweight 2D ICP using only NumPy (no external dependencies).

    - Converts each ``LaserScan`` to a 2D point set (x,y) in the laser frame.
    - Aligns scan_j ("sensor" / current) to scan_i ("reference" / previous).

    ICP estimates ``T_ref_sens`` (transform that maps points from scan_j frame into
    scan_i frame). GTSAM ``BetweenFactorPose2`` expects ``z_ij = T_i^-1 T_j``
    (transform from pose i to pose j), so we return the inverse transform.

    Returns ``None`` if scan matching fails, so the caller can fall back to odometry.
    """

    # Basic validation
    if scan_i is None or scan_j is None:
        return None
    if not scan_i.ranges or not scan_j.ranges:
        return None
    if float(scan_i.angle_increment) == 0.0 or float(scan_j.angle_increment) == 0.0:
        return None

    # ----------------------
    # Helpers
    # ----------------------
    def _scan_to_xy(scan: LaserScan) -> np.ndarray:
        """Convert LaserScan to an (N,2) array of points in the scan frame."""
        r = np.asarray(scan.ranges, dtype=np.float64)
        n = r.shape[0]
        angles = float(scan.angle_min) + np.arange(n, dtype=np.float64) * float(scan.angle_increment)

        finite = np.isfinite(r)
        in_range = (r >= float(scan.range_min)) & (r <= float(scan.range_max))
        valid = finite & in_range
        if not np.any(valid):
            return np.empty((0, 2), dtype=np.float64)

        r = r[valid]
        a = angles[valid]
        x = r * np.cos(a)
        y = r * np.sin(a)
        return np.column_stack([x, y])

    def _best_fit_transform_2d(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute R,t that best align B to A in LS sense: A ~= R @ B + t."""
        centroid_A = A.mean(axis=0)
        centroid_B = B.mean(axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = BB.T @ AA
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Ensure det(R)=+1
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_A - (R @ centroid_B)
        return R, t

    def _nearest_neighbor_bruteforce(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """For each point in src, find nearest in dst (O(N*M) brute force)."""
        diff = src[:, None, :] - dst[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        idx = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(d2.shape[0]), idx]
        return idx, min_d2

    def _yaw_from_R(R: np.ndarray) -> float:
        return math.atan2(float(R[1, 0]), float(R[0, 0]))

    # ----------------------
    # Point sets
    # ----------------------
    ref = _scan_to_xy(scan_i)
    sens = _scan_to_xy(scan_j)
    if ref.shape[0] < 30 or sens.shape[0] < 30:
        return None

    # Cap compute cost (brute-force NN is O(N^2))
    max_points = 700
    if ref.shape[0] > max_points:
        ref = ref[::int(math.ceil(ref.shape[0] / max_points))]
    if sens.shape[0] > max_points:
        sens = sens[::int(math.ceil(sens.shape[0] / max_points))]

    # ----------------------
    # ICP parameters (safe defaults; tune per dataset)
    # ----------------------
    max_iters = 50
    max_corr_dist = 2.0  # meters
    min_corr = 40
    max_corr_d2 = float(max_corr_dist * max_corr_dist)
    eps_trans = 1e-4
    eps_rot = 1e-4

    # Current estimate of T_ref_sens
    R_total = np.eye(2, dtype=np.float64)
    t_total = np.zeros(2, dtype=np.float64)
    prev_err = None

    for _ in range(max_iters):
        # Transform sens points into ref using current estimate
        sens_tf = (sens @ R_total.T) + t_total

        nn_idx, nn_d2 = _nearest_neighbor_bruteforce(sens_tf, ref)
        good = nn_d2 <= max_corr_d2
        if int(np.count_nonzero(good)) < min_corr:
            return None

        A = ref[nn_idx[good]]
        B_current = sens_tf[good]

        # Increment that aligns current transformed sens to ref
        R_inc, t_inc = _best_fit_transform_2d(A, B_current)
        R_total = R_inc @ R_total
        t_total = (R_inc @ t_total) + t_inc

        mean_err = float(np.sqrt(nn_d2[good].mean()))
        if prev_err is not None and abs(prev_err - mean_err) < 1e-5:
            break
        prev_err = mean_err

        # Small motion => converged
        if (abs(float(t_inc[0])) < eps_trans and abs(float(t_inc[1])) < eps_trans and abs(_yaw_from_R(R_inc)) < eps_rot):
            break

    # Quality gate
    if prev_err is None or prev_err > 0.5:
        return None

    # Convert to z_ij: invert T_ref_sens -> T_sens_ref (i->j)
    # R_inv = R_total.T
    # t_inv = -(R_inv @ t_total)
    dx = float(t_total[0])
    dy = float(t_total[1])
    dth = wrap_angle(_yaw_from_R(R_total))

    # Sanity gate to avoid catastrophic factors
    if math.hypot(dx, dy) > 3.0 or abs(dth) > math.radians(120.0):
        return None

    return gtsam.Pose2(dx, dy, dth)

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class Stamp:
    sec: int
    nsec: int

    @property
    def ns(self) -> int:
        return int(self.sec) * 1_000_000_000 + int(self.nsec)

@dataclass
class Keyframe:
    idx: int
    stamp: Stamp
    odom_pose: gtsam.Pose2
    scan: Optional[LaserScan] = None


# -----------------------------
# Loop-closure plug-in interface
# -----------------------------

@dataclass
class LoopClosureConstraint:
    i: int
    j: int
    z_ij: gtsam.Pose2
    noise: gtsam.noiseModel.Base

class LoopClosureModule:
    """Lightweight loop-closure detector using gated scan matching."""
    def __init__(
        self,
        noise: gtsam.noiseModel.Base,
        min_idx_separation: int = 20,
        search_radius_m: float = 1.5,
        max_candidates: int = 3,
        max_trans_error_m: float = 1.0,
        max_rot_error_rad: float = math.radians(35.0),
    ):
        self.noise = noise
        self.min_idx_separation = int(min_idx_separation)
        self.search_radius_m = float(search_radius_m)
        self.max_candidates = int(max_candidates)
        self.max_trans_error_m = float(max_trans_error_m)
        self.max_rot_error_rad = float(max_rot_error_rad)

    @staticmethod
    def _pose_for_kf(kf: Keyframe, current_estimate: Optional[gtsam.Values]) -> gtsam.Pose2:
        if current_estimate is not None:
            k = symbol('x', kf.idx)
            if current_estimate.exists(k):
                return current_estimate.atPose2(k)
        return kf.odom_pose

    def on_new_keyframe(
        self,
        new_kf: Keyframe,
        keyframes: List[Keyframe],
        current_estimate: Optional[gtsam.Values],
    ) -> List[LoopClosureConstraint]:
        if new_kf.scan is None or len(keyframes) <= 1:
            return []

        j_idx = new_kf.idx
        j_pose = self._pose_for_kf(new_kf, current_estimate)
        candidate_by_dist: List[Tuple[float, Keyframe, gtsam.Pose2]] = []

        for kf in keyframes[:-1]:
            if (j_idx - kf.idx) < self.min_idx_separation:
                continue
            if kf.scan is None:
                continue

            i_pose = self._pose_for_kf(kf, current_estimate)
            dist = math.hypot(i_pose.x() - j_pose.x(), i_pose.y() - j_pose.y())
            if dist > self.search_radius_m:
                continue
            candidate_by_dist.append((dist, kf, i_pose))

        if not candidate_by_dist:
            return []

        candidate_by_dist.sort(key=lambda x: x[0])
        for _, old_kf, old_pose in candidate_by_dist[:self.max_candidates]:
            z_ij = compute_pose_ls(old_kf.scan, new_kf.scan)
            if z_ij is None:
                continue

            # Gate scan-based estimate against odom/optimized relative pose.
            pred_ij = old_pose.between(j_pose)
            err = pred_ij.between(z_ij)
            if math.hypot(err.x(), err.y()) > self.max_trans_error_m:
                continue
            if abs(wrap_angle(err.theta())) > self.max_rot_error_rad:
                continue

            return [LoopClosureConstraint(i=old_kf.idx, j=j_idx, z_ij=z_ij, noise=self.noise)]

        return []


class NoLoopClosure:
    """Default plug-in: produces no loop closures."""
    def on_new_keyframe(self, new_kf: Keyframe, keyframes: List[Keyframe], current_estimate: Optional[gtsam.Values]):
        return []


# -----------------------------
# Back-end: owns GTSAM + optimization
# -----------------------------

class BackendOptimizer:
    """
    Narrow interface: the front-end can only:
    - add_prior(idx, pose, noise)
    - add_between(i, j, measurement, noise, initial_guess_for_j)
    - update()
    - get_estimate()
    """

    def __init__(self, isam_params: Optional[gtsam.ISAM2Params] = None):
        self._graph = gtsam.NonlinearFactorGraph() # tmp container from new factors
        self._init = gtsam.Values() # TODO: rename it (it is a initial guess for the graph update)

        # Get iSAM params
        # TODO: expose tem as ROS params
        params = isam_params if isam_params is not None else gtsam.ISAM2Params()
        if isam_params is None:
            # Relinearize if variables shift more than threshold
            if hasattr(params, 'setRelinearizeThreshold'):
                params.setRelinearizeThreshold(0.01)
            else:
                params.relinearizeThreshold = 0.01
            # Relinearize at least every N updates
            if hasattr(params, 'setRelinearizeSkip'):
                params.setRelinearizeSkip(1)
            else:
                params.relinearizeSkip = 1
        # Create iSAM (interactive smoothing and mapping) optimizer
        self._isam = gtsam.ISAM2(params)

        self._has_initialized = False

    def add_prior(self, idx: int, pose: gtsam.Pose2, noise: gtsam.noiseModel.Base) -> None:
        k = symbol('x', idx)
        self._graph.add(gtsam.PriorFactorPose2(k, pose, noise))
        self._init.insert(k, pose)

    def add_between(
        self,
        i: int,
        j: int,
        z_ij: gtsam.Pose2,
        noise: gtsam.noiseModel.Base,
        initial_guess_for_j: gtsam.Pose2,
    ) -> None:
        ki = symbol('x', i)
        kj = symbol('x', j)
        self._graph.add(gtsam.BetweenFactorPose2(ki, kj, z_ij, noise))
        if not self._init.exists(kj):
            self._init.insert(kj, initial_guess_for_j)

    def update(self) -> None:
        if not self._has_initialized:
            self._has_initialized = True
        # Update the iSAM optimizer with the new factors and initial guesses.
        self._isam.update(self._graph, self._init)
        # End-of-cycle
        self._graph.resize(0)
        self._init.clear()

    def get_estimate(self) -> gtsam.Values:
        return self._isam.calculateEstimate()


# -----------------------------
# Front-end: keyframes + factor generation
# -----------------------------

class Frontend:
    """
    Holds:
    - keyframe logic
    - scan attachment
    - factor generation (odom between, plus optional loop closures)
    """

    def __init__(
        self,
        trans_thresh_m: float,
        rot_thresh_rad: float,
        noise_prior: gtsam.noiseModel.Base,
        noise_odom: gtsam.noiseModel.Base,
        loop_closure: LoopClosureModule,
    ):
        self.trans_thresh = float(trans_thresh_m)
        self.rot_thresh = float(rot_thresh_rad)
        self.noise_prior = noise_prior
        self.noise_odom = noise_odom
        self.loop_closure = loop_closure

        self.keyframes: List[Keyframe] = []
        self._pending_scans_by_time: Dict[int, LaserScan] = {}
        self._last_scan: Optional[LaserScan] = None

    def on_scan(self, scan: LaserScan) -> None:
        """
        Store a scan for later keyframe association.

        :param scan: input scan
        """
        # TODO: hard-coded values as constants
        self._last_scan = scan
        # create int nanosecond for dict key
        t_ns = int(scan.header.stamp.sec) * 1_000_000_000 + int(scan.header.stamp.nanosec)
        self._pending_scans_by_time[t_ns] = scan
        # Clean up old scans
        if len(self._pending_scans_by_time) > 5000:
            for k in sorted(self._pending_scans_by_time.keys())[:1000]:
                self._pending_scans_by_time.pop(k, None)

    def maybe_create_keyframe(
        self,
        odom_pose: gtsam.Pose2,
        stamp: Stamp,
        backend_estimate: Optional[gtsam.Values],
    ) -> Tuple[Optional[Keyframe], List[Tuple[str, dict]]]:
        """
        Decides whether to create a new keyframe based on odometry thresholds,
        and if so, which factors to add to the back-end.

        :param odom_pose: current odometry pose
        :type odom_pose: gtsam.Pose2
        :param stamp: timestamp of the odometry reading
        :type stamp: Stamp
        :param backend_estimate: current back-end estimate for all poses (can be None early on)
        :type backend_estimate: Optional[gtsam.Values]
        :return: new Keyframe if created (else None), and list of "actions" for the caller to apply to backend
        :rtype: Tuple[Keyframe | None, List[Tuple[str, dict]]]
        """

        actions: List[Tuple[str, dict]] = []

        # Create first keyframe if none exist (with a prior factor)
        if not self.keyframes:
            kf0 = self._make_keyframe(idx=0, odom_pose=odom_pose, stamp=stamp)
            self.keyframes.append(kf0)
            actions.append(("prior", {"idx": 0, "pose": odom_pose, "noise": self.noise_prior}))
            return kf0, actions

        # Calculate relative transform from last KF to current odom pose
        last_kf = self.keyframes[-1]
        rel = last_kf.odom_pose.between(odom_pose)
        trans = math.hypot(rel.x(), rel.y())
        rot = abs(wrap_angle(rel.theta()))
        # Return if distance is below thresholds (no new keyframe needed)
        if trans < self.trans_thresh and rot < self.rot_thresh:
            return None, actions
        # Otherwise, create new KF using latest scan
        new_idx = last_kf.idx + 1
        new_kf = self._make_keyframe(idx=new_idx, odom_pose=odom_pose, stamp=stamp)
        self.keyframes.append(new_kf)

        # Initial guess for the new KF j (use backend estimate if available)
        # TODO: maybe rename xj_init
        init_j = self._compose_from_backend_or_fallback(
            backend_estimate=backend_estimate,
            prev_idx=last_kf.idx,
            prev_fallback=last_kf.odom_pose,
            rel=rel,
        )

        # Scan-based relative pose measurement (i -> j); fallback to odometry if unavailable.
        z_ij = compute_pose_ls(last_kf.scan, new_kf.scan)
        if z_ij is None:
            z_ij = rel

        # Add between factor (scan-based when available, otherwise odometry).
        actions.append((
            "between",
            {
                "i": last_kf.idx,
                "j": new_idx,
                "z_ij": z_ij,
                "noise": self.noise_odom,   # TODO: add observation noise
                "init_j": init_j,
            }
        ))

        # Optional loop-closure constraints via plug-in
        for lc in self.loop_closure.on_new_keyframe(new_kf, self.keyframes, backend_estimate):
            init_guess_j = init_j  # typically already present; back-end will ignore duplicates
            actions.append((
                "between",
                {
                    "i": lc.i,
                    "j": lc.j,
                    "z_ij": lc.z_ij,
                    "noise": lc.noise,
                    "init_j": init_guess_j,
                }
            ))

        return new_kf, actions

    def _make_keyframe(self, idx: int, odom_pose: gtsam.Pose2, stamp: Stamp) -> Keyframe:
        scan = self._pending_scans_by_time.get(stamp.ns, self._last_scan)
        return Keyframe(idx=idx, stamp=stamp, odom_pose=odom_pose, scan=scan)

    @staticmethod
    def _compose_from_backend_or_fallback(
        backend_estimate: Optional[gtsam.Values],
        prev_idx: int,
        prev_fallback: gtsam.Pose2,
        rel: gtsam.Pose2,
    ) -> gtsam.Pose2:
        # If backend estimate is available,
        # use it to compose the initial guess for the new keyframe.
        if backend_estimate is not None:
            k_prev = symbol('x', prev_idx)
            if backend_estimate.exists(k_prev):
                prev_opt = backend_estimate.atPose2(k_prev)
                return prev_opt.compose(rel)
        return prev_fallback.compose(rel)

# -----------------------------
# ROS2 Node: wires Front-end and Back-end + I/O
# -----------------------------

class PoseGraphNode(Node):
    def __init__(self):
        super().__init__('pose_graph_node')

        # Topics
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value

        # Keyframe thresholds (Grisetti V-A defaults)
        self.declare_parameter('trans_thresh_m', 0.2)
        self.declare_parameter('rot_thresh_rad', 0.2)

        # Noise (since rf2o covariance appears zero; tune these)
        self.declare_parameter('prior_sigmas', [1e-3, 1e-3, 1e-3])
        self.declare_parameter('odom_sigmas',  [0.1, 0.1, 0.20])
        self.declare_parameter('loop_sigmas', [0.2, 0.2, 0.30])

        # Loop-closure plug-in parameters
        self.declare_parameter('loop_enable', True)
        self.declare_parameter('loop_min_idx_separation', 5)
        self.declare_parameter('loop_search_radius_m', 1.5)
        self.declare_parameter('loop_max_candidates', 3)
        self.declare_parameter('loop_max_trans_error_m', 3.0)
        self.declare_parameter('loop_max_rot_error_rad', 1.57)

        trans_thresh = float(self.get_parameter('trans_thresh_m').value)
        rot_thresh = float(self.get_parameter('rot_thresh_rad').value)
        prior_sigmas = self.get_parameter('prior_sigmas').value
        odom_sigmas = self.get_parameter('odom_sigmas').value
        loop_sigmas = self.get_parameter('loop_sigmas').value
        loop_enable = bool(self.get_parameter('loop_enable').value)
        loop_min_idx_separation = int(self.get_parameter('loop_min_idx_separation').value)
        loop_search_radius_m = float(self.get_parameter('loop_search_radius_m').value)
        loop_max_candidates = int(self.get_parameter('loop_max_candidates').value)
        loop_max_trans_error_m = float(self.get_parameter('loop_max_trans_error_m').value)
        loop_max_rot_error_rad = float(self.get_parameter('loop_max_rot_error_rad').value)

        # Define noise models
        # prior: first node (anchor), so very tight; odom: looser to allow for correction
        noise_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([float(prior_sigmas[0]), float(prior_sigmas[1]), float(prior_sigmas[2])], dtype=float)
        )
        noise_odom = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([float(odom_sigmas[0]), float(odom_sigmas[1]), float(odom_sigmas[2])], dtype=float)
        )
        noise_loop = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([float(loop_sigmas[0]), float(loop_sigmas[1]), float(loop_sigmas[2])], dtype=float)
        )

        loop_closure = (
            LoopClosureModule(
                noise=noise_loop,
                min_idx_separation=loop_min_idx_separation,
                search_radius_m=loop_search_radius_m,
                max_candidates=loop_max_candidates,
                max_trans_error_m=loop_max_trans_error_m,
                max_rot_error_rad=loop_max_rot_error_rad,
            )
            if loop_enable
            else NoLoopClosure()
        )

        # Build front-end and back-end
        self.frontend = Frontend(
            trans_thresh_m=trans_thresh,
            rot_thresh_rad=rot_thresh,
            noise_prior=noise_prior,
            noise_odom=noise_odom,
            loop_closure=loop_closure,
        )
        self.backend = BackendOptimizer()

        # Publishers
        self.path_raw_pub = self.create_publisher(Path, '/pose_graph/path_raw', 10)
        self.path_opt_pub = self.create_publisher(Path, '/pose_graph/path_opt', 10)

        # Subscribers
        self.create_subscription(LaserScan, self.scan_topic, self._on_scan, 50)
        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 50)

        self.get_logger().info(
            f"Subscribed to odom={self.odom_topic}, scan={self.scan_topic}; "
            f"keyframe thresholds: trans={trans_thresh} m, rot={rot_thresh} rad"
        )

    def _on_scan(self, msg: LaserScan):
        self.frontend.on_scan(msg)

    def _on_odom(self, msg: Odometry):
        # Convert odom pose -> Pose2
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        odom_pose = gtsam.Pose2(float(p.x), float(p.y), float(yaw))
        stamp = Stamp(sec=msg.header.stamp.sec, nsec=msg.header.stamp.nanosec)

        # Get current backend estimate (may be empty early)
        backend_est = None
        try:
            backend_est = self.backend.get_estimate()
        except Exception:
            backend_est = None

        # Front-end decides if a new keyframe is needed and which factors to add
        new_kf, actions = self.frontend.maybe_create_keyframe(
            odom_pose=odom_pose,
            stamp=stamp,
            backend_estimate=backend_est,
        )

        # No new keyframe -> no new factors -> no backend update needed
        if new_kf is None:
            return

        # Apply actions to backend (explicit boundary)
        for kind, payload in actions:
            if kind == "prior":
                self.backend.add_prior(payload["idx"], payload["pose"], payload["noise"])
            elif kind == "between":
                self.backend.add_between(
                    i=payload["i"],
                    j=payload["j"],
                    z_ij=payload["z_ij"],
                    noise=payload["noise"],
                    initial_guess_for_j=payload["init_j"],
                )
            else:
                self.get_logger().warn(f"Unknown action: {kind}")

        # Trigger back-end optimization update (incremental)
        self.backend.update()

        # Publish outputs
        self._publish_paths()

    def _publish_paths(self):

        # No keyframes, no path, return
        if not self.frontend.keyframes:
            return

        frame_id = "odom"
        last_stamp = self.frontend.keyframes[-1].stamp

        # Raw keyframe path (from odom poses)
        path_raw = Path()
        path_raw.header.frame_id = frame_id
        path_raw.header.stamp.sec = last_stamp.sec
        path_raw.header.stamp.nanosec = last_stamp.nsec

        for kf in self.frontend.keyframes:
            path_raw.poses.append(
                pose2_to_ros_pose_stamped(kf.odom_pose, frame_id, kf.stamp.sec, kf.stamp.nsec)
            )
        self.path_raw_pub.publish(path_raw)

        # Optimized path (from backend)
        path_opt = Path()
        path_opt.header = path_raw.header

        est = self.backend.get_estimate()
        for kf in self.frontend.keyframes:
            k = symbol('x', kf.idx)
            if not est.exists(k):
                continue
            p = est.atPose2(k)
            path_opt.poses.append(
                pose2_to_ros_pose_stamped(p, frame_id, kf.stamp.sec, kf.stamp.nsec)
            )
        self.path_opt_pub.publish(path_opt)


def main():
    rclpy.init()
    node = PoseGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
