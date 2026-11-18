import os
import sys

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nav2_start import MANAGER, NAV2

PARAMS = os.environ.get("NAV_PARAMS", "/nav/params")


def generate_launch_description():
    sim_time = {"use_sim_time": True}
    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource("/opt/ros/jazzy/share/slam_toolbox/launch/online_async_launch.py"),
        launch_arguments={"slam_params_file": f"{PARAMS}/slam_params.yaml", "use_sim_time": "true",
                          "autostart": "true"}.items())]
    actions += [Node(package=pkg, executable=exe, name=exe, output="screen",
                     parameters=[f"{PARAMS}/nav2_params.yaml", sim_time]) for pkg, exe in NAV2]
    actions.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name=MANAGER, output="screen",
                        parameters=[{"use_sim_time": False, "autostart": False, "bond_timeout": 0.0,
                                     "node_names": [exe for _, exe in NAV2]}]))
    return LaunchDescription(actions)
