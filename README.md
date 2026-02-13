# lidar_pose_graph_frontend

> Lightweight ROS2 front-end for a 2D pose-graph pipeline (keyframing + factor generation).

This package implements the front-end of a simple pose-graph SLAM pipeline and a narrow back-end interface. It selects keyframes from odometry, attaches recent scans, and generates factors (prior / odometry / optional loop-closures) which are applied to a GTSAM-based back-end (iSAM2).

Features
- Keyframe selection using configurable translation and rotation thresholds.
- Attaches the most recent `sensor_msgs/LaserScan` message to keyframes for later loop-closure or mapping use.
- Generates GTSAM factors (PriorFactorPose2, BetweenFactorPose2) and provides an incremental update via iSAM2.
- Publishes both the raw (odometry) and optimized pose paths.

Node
- Name: `pose_graph_node`
- Main module: `lidar_pose_graph_frontend.pose_graph_node`

Topics
- Subscribed:
  - `odom` (default `/odom`) — `nav_msgs/Odometry`
  - `scan` (default `/scan`) — `sensor_msgs/LaserScan`
- Published:
  - `/pose_graph/path_raw` — `nav_msgs/Path` (raw odom keyframe path)
  - `/pose_graph/path_opt` — `nav_msgs/Path` (optimized path from backend)

Parameters
- `odom_topic` (string, default: `/odom`) — odometry topic to subscribe to.
- `scan_topic` (string, default: `/scan`) — scan topic to subscribe to.
- `trans_thresh_m` (float, default: `0.5`) — translation threshold (meters) for creating a new keyframe.
- `rot_thresh_rad` (float, default: `0.5`) — rotation threshold (radians) for creating a new keyframe.
- `prior_sigmas` (list, default: `[1e-3, 1e-3, 1e-3]`) — diagonal sigmas for the prior noise model.
- `odom_sigmas` (list, default: `[0.05, 0.05, 0.10]`) — diagonal sigmas for odometry between factors.

How it works (high level)
- On each incoming `Odometry` message the front-end:
  1. Converts the odometry pose to a `gtsam.Pose2`.
  2. Optionally creates a new keyframe if motion exceeds the configured thresholds.
  3. When a keyframe is created, the front-end emits actions for the back-end:
     - a `prior` for the first keyframe, and `between` factors for odometry / loop-closures.
  4. The back-end (GTSAM iSAM2 wrapper) accepts those factor additions and performs incremental updates.
  5. The node publishes both raw and optimized paths.

Loop-closure
- The package includes a simple `LoopClosureModule` plug-in interface. The default `NoLoopClosure` implementation returns no loop closures. You can implement the protocol to propose `LoopClosureConstraint`s which the front-end will forward to the back-end as `Between` factors.

Dependencies
- ROS2 Python libraries: `rclpy`, `nav_msgs`, `sensor_msgs`, `geometry_msgs`.
- GTSAM Python bindings (`gtsam`) for factor graph and iSAM2.

Development / Running
1. From your ROS2 workspace root:

```bash
colcon build --packages-select lidar_pose_graph_frontend
source install/setup.bash
```

2. Run the node (two options):

- If the package installs an executable entry point, use:

```bash
ros2 run lidar_pose_graph_frontend pose_graph_node
```

- Or run the module directly for quick testing:

```bash
python3 -m lidar_pose_graph_frontend.pose_graph_node
```

Notes
- The front-end keeps a small buffer of incoming scans and attaches the scan whose timestamp best matches the keyframe timestamp (or the most recent scan if none match exactly).
- The back-end wrapper is implemented as `BackendOptimizer` and uses `gtsam.ISAM2` for incremental optimization.
- Tweak `trans_thresh_m`, `rot_thresh_rad`, and noise parameters to suit your sensor and odometry characteristics.

License
- See `LICENSE` in the package (if present) or add your preferred license.

Contact
- For questions or contribution, inspect `lidar_pose_graph_frontend/pose_graph_node.py` for implementation details and extend the loop-closure plugin to integrate external detectors.
