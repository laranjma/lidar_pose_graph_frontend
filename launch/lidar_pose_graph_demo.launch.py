from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # RViz config stored in your package share (recommended).
    # If you keep it in repo root, hardcoding is fragile; better install it like configs.
    pkg_share = get_package_share_directory("lidar_pose_graph_frontend")
    rviz_config = os.path.join(pkg_share, "rviz", "lidar_pose_graph.rviz")

    return LaunchDescription([
        # Dataset player (your existing package)
        Node(
            package="slam_dataset_player",
            executable="carmen_player",
            name="carmen_player",
            output="screen",
            parameters=[{"log_path": "/home/matheus/data/datasets/csail-newcarmen.log/mit-csail-3rd-floor-2005-12-17-run4.log",
                         "mode": "fixed_rate",
                         "rate_hz": 10.0,
                         "scan_topic": "/scan",
                         "scan_frame_id": "laser"}],
        ),

        # Odometry node (RF2O)
        Node(
            package="rf2o_laser_odometry",
            executable="rf2o_laser_odometry_node",
            name="rf2o_laser_odometry",
            output="screen",
            parameters=[{"laser_scan_topic": "/scan",
                         "odom_topic": "/odom",
                         "publish_tf": True,
                         "base_frame_id": "base_link",
                         "odom_frame_id": "odom",
                         "init_pose_from_topic": "",
                         "freq": 20.0}],
        ),

        # Pose graph frontend backend wrapper
        Node(
            package="lidar_pose_graph_frontend",
            executable="pose_graph_node",
            name="pose_graph_node",
            output="screen",
        ),

        # Static TF base_link -> laser (identity for now)
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_to_laser_tf",
            output="screen",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "laser"],
        ),

        # RViz
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            additional_env={
                "LIBGL_ALWAYS_SOFTWARE": "1",
            },
        ),
    ])
