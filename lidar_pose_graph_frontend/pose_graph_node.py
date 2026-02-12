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
from typing import Optional, Dict, List, Tuple, Protocol

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

class LoopClosureModule(Protocol):
    def on_new_keyframe(
        self,
        new_kf: Keyframe,
        keyframes: List[Keyframe],
        current_estimate: Optional[gtsam.Values],
    ) -> List[LoopClosureConstraint]:
        ...


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
        self._graph = gtsam.NonlinearFactorGraph()
        self._init = gtsam.Values()

        params = isam_params if isam_params is not None else gtsam.ISAM2Params()
        if isam_params is None:
            if hasattr(params, 'setRelinearizeThreshold'):
                params.setRelinearizeThreshold(0.01)
            else:
                params.relinearizeThreshold = 0.01

            if hasattr(params, 'setRelinearizeSkip'):
                params.setRelinearizeSkip(1)
            else:
                params.relinearizeSkip = 1
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
            self._isam.update(self._graph, self._init)
            self._has_initialized = True
        else:
            self._isam.update(self._graph, self._init)
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
        self._last_scan = scan
        t_ns = int(scan.header.stamp.sec) * 1_000_000_000 + int(scan.header.stamp.nanosec)
        self._pending_scans_by_time[t_ns] = scan
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
        Returns:
        - new Keyframe if created (else None)
        - list of "actions" for the caller to apply to backend:
            ("prior", {...}) or ("between", {...}) etc.
        """
        actions: List[Tuple[str, dict]] = []

        if not self.keyframes:
            kf0 = self._make_keyframe(idx=0, odom_pose=odom_pose, stamp=stamp)
            self.keyframes.append(kf0)
            actions.append(("prior", {"idx": 0, "pose": odom_pose, "noise": self.noise_prior}))
            return kf0, actions

        last_kf = self.keyframes[-1]
        rel = last_kf.odom_pose.between(odom_pose)
        trans = math.hypot(rel.x(), rel.y())
        rot = abs(wrap_angle(rel.theta()))

        if trans < self.trans_thresh and rot < self.rot_thresh:
            return None, actions

        new_idx = last_kf.idx + 1
        new_kf = self._make_keyframe(idx=new_idx, odom_pose=odom_pose, stamp=stamp)
        self.keyframes.append(new_kf)

        # Initial guess for j (use backend estimate if available)
        init_j = self._compose_from_backend_or_fallback(
            backend_estimate=backend_estimate,
            prev_idx=last_kf.idx,
            prev_fallback=last_kf.odom_pose,
            rel=rel,
        )

        actions.append((
            "between",
            {
                "i": last_kf.idx,
                "j": new_idx,
                "z_ij": rel,                 # odom-based measurement for now
                "noise": self.noise_odom,
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
        super().__init__('pose_graph_frontend_backend')

        # Topics
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('scan_topic', '/scan')
        self.odom_topic = self.get_parameter('odom_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value

        # Keyframe thresholds (Grisetti V-A defaults)
        self.declare_parameter('trans_thresh_m', 0.5)
        self.declare_parameter('rot_thresh_rad', 0.5)

        # Noise (since rf2o covariance appears zero; tune these)
        self.declare_parameter('prior_sigmas', [1e-3, 1e-3, 1e-3])
        self.declare_parameter('odom_sigmas',  [0.05, 0.05, 0.10])

        trans_thresh = float(self.get_parameter('trans_thresh_m').value)
        rot_thresh = float(self.get_parameter('rot_thresh_rad').value)
        prior_sigmas = self.get_parameter('prior_sigmas').value
        odom_sigmas = self.get_parameter('odom_sigmas').value

        noise_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([float(prior_sigmas[0]), float(prior_sigmas[1]), float(prior_sigmas[2])], dtype=float)
        )
        noise_odom = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([float(odom_sigmas[0]), float(odom_sigmas[1]), float(odom_sigmas[2])], dtype=float)
        )

        # Plug-in: swap NoLoopClosure() with a real module later
        loop_closure = NoLoopClosure()

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
