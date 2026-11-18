import json
import math
import os

from launch import LaunchDescription
from launch.actions import GroupAction
from launch_ros.actions import Node, PushRosNamespace

ROBOTS = int(os.environ.get("NAV_ROBOTS", "1"))
FIRST = int(os.environ.get("NAV_FIRST", "0"))
AMCL = os.environ.get("NAV_AMCL", "0") == "1"
PARAMS = os.environ.get("NAV_PARAMS", "/nav/fleet_params.yaml")
MAP = os.environ.get("NAV_MAP", "/maps/warehouse_lidar.yaml")
LAYOUT = os.environ.get("NAV_LAYOUT", "/maps/layout.json")
OUTPUT = os.environ.get("NAV_OUTPUT", "log")
NAV2 = [("nav2_controller", "controller_server"), ("nav2_planner", "planner_server"),
        ("nav2_behaviors", "behavior_server"), ("nav2_bt_navigator", "bt_navigator"),
        ("nav2_collision_monitor", "collision_monitor")]
REMAP = [("/tf", "tf"), ("/tf_static", "tf_static")]


def generate_launch_description():
    sim = {"use_sim_time": True}
    starts = [r["start"] for r in json.load(open(LAYOUT))["robots"]]
    actions = [
        Node(package="nav2_map_server", executable="map_server", name="map_server", output=OUTPUT,
             parameters=[sim, {"yaml_filename": MAP, "topic_name": "map", "frame_id": "map"}]),
        Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_map", output=OUTPUT,
             parameters=[sim, {"autostart": True, "bond_timeout": 0.0, "node_names": ["map_server"]}]),
    ]
    for i in range(FIRST, FIRST + ROBOTS):
        nodes = ([("nav2_amcl", "amcl")] if AMCL else []) + list(NAV2)
        group = [PushRosNamespace(f"go2_{i}")]
        for pkg, exe in nodes:
            extra = {}
            if exe == "amcl":
                x, y, yaw = starts[i]
                extra = {"initial_pose.x": float(x), "initial_pose.y": float(y), "initial_pose.z": 0.0,
                         "initial_pose.yaw": math.radians(yaw)}
            group.append(Node(package=pkg, executable=exe, name=exe, output=OUTPUT, remappings=REMAP + [("scan", f"/go2_{i}/scan")] + ([("cmd_vel", "cmd_vel_ctrl")] if exe == "controller_server" else []) + ([("cmd_vel", "cmd_vel_nav")] if exe == "behavior_server" else []),
                              parameters=[PARAMS, sim, extra]))
        group.append(Node(executable="python3", arguments=["/nav/turn_arc.py"], name="turn_arc", output=OUTPUT,
                          remappings=REMAP, parameters=[sim]))
        group.append(Node(package="nav2_lifecycle_manager", executable="lifecycle_manager", name="lifecycle_manager_navigation",
                          output=OUTPUT, parameters=[sim, {"autostart": True, "bond_timeout": 0.0,
                                                           "node_names": [exe for _, exe in nodes]}]))
        actions.append(GroupAction(group))
    return LaunchDescription(actions)
