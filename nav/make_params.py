import os
import sys
from pathlib import Path

import yaml

SHARE = Path("/opt/ros/jazzy/share")
DEFAULTS = SHARE / "nav2_bringup/params/nav2_params.yaml"
BT_XML = SHARE / "nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml"
SLAM_DEFAULTS = SHARE / "slam_toolbox/config/mapper_params_online_async.yaml"
FOOTPRINT = "[[0.36, 0.2], [0.36, -0.2], [-0.45, -0.2], [-0.45, 0.2]]"
INFLATION = float(os.environ.get("INFLATION_RADIUS", 0.55))


def scan_layer(obstacle_range, raytrace_range):
    return {
        "plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True, "observation_sources": "scan",
        "scan": {"topic": "/scan", "data_type": "LaserScan", "marking": True, "clearing": True,
                 "obstacle_max_range": obstacle_range, "obstacle_min_range": 0.0,
                 "raytrace_max_range": raytrace_range, "raytrace_min_range": 0.0,
                 "max_obstacle_height": 2.0, "min_obstacle_height": -1.0, "inf_is_valid": True},
    }


def costmap(frame, size, resolution, update_hz, publish_hz, rolling, static):
    plugins = (["static_layer"] if static else []) + ["obstacle_layer", "inflation_layer"]
    out = {
        "update_frequency": update_hz, "publish_frequency": publish_hz, "global_frame": frame,
        "robot_base_frame": "base_link", "use_sim_time": True, "rolling_window": rolling, "width": size,
        "height": size, "resolution": resolution, "footprint": FOOTPRINT, "track_unknown_space": static,
        "transform_tolerance": 0.5, "always_send_full_costmap": True, "plugins": plugins,
        "obstacle_layer": scan_layer(8.0 if rolling else 18.0, 10.0 if rolling else 20.0),
        "inflation_layer": {"plugin": "nav2_costmap_2d::InflationLayer", "cost_scaling_factor": 3.0,
                            "inflation_radius": INFLATION},
    }
    if static:
        out["static_layer"] = {"plugin": "nav2_costmap_2d::StaticLayer", "map_subscribe_transient_local": True,
                               "subscribe_to_updates": True}
    return out


def main(out_dir):
    defaults = yaml.safe_load(DEFAULTS.read_text())
    params = {
        "bt_navigator": {"ros__parameters": {**defaults["bt_navigator"]["ros__parameters"], "global_frame": "map",
                                              "robot_base_frame": "base_link", "use_sim_time": True,
                                              "transform_tolerance": 0.5, "default_nav_to_pose_bt_xml": str(BT_XML)}},
        "controller_server": {"ros__parameters": {
            "use_sim_time": True, "controller_frequency": 20.0, "min_x_velocity_threshold": 0.001,
            "min_y_velocity_threshold": 0.5, "min_theta_velocity_threshold": 0.001, "failure_tolerance": 0.5,
            "progress_checker_plugins": ["progress_checker"], "goal_checker_plugins": ["general_goal_checker"],
            "controller_plugins": ["FollowPath"],
            "progress_checker": {"plugin": "nav2_controller::SimpleProgressChecker", "required_movement_radius": 0.3,
                                 "movement_time_allowance": 25.0},
            "general_goal_checker": {"plugin": "nav2_controller::SimpleGoalChecker", "stateful": True,
                                     "xy_goal_tolerance": 0.5, "yaw_goal_tolerance": 3.14},
            "FollowPath": {
                "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
                "desired_linear_vel": 0.7, "lookahead_dist": 1.2, "min_lookahead_dist": 0.7, "max_lookahead_dist": 1.8,
                "lookahead_time": 1.5, "rotate_to_heading_angular_vel": 0.8, "transform_tolerance": 0.5,
                "use_velocity_scaled_lookahead_dist": True, "min_approach_linear_velocity": 0.1,
                "approach_velocity_scaling_dist": 1.0, "use_collision_detection": False,
                "max_allowed_time_to_collision_up_to_carrot": 1.0, "use_regulated_linear_velocity_scaling": True,
                "use_cost_regulated_linear_velocity_scaling": False, "regulated_linear_scaling_min_radius": 0.9,
                "regulated_linear_scaling_min_speed": 0.25, "use_rotate_to_heading": True,
                "rotate_to_heading_min_angle": 0.785, "max_angular_accel": 3.0, "max_robot_pose_search_dist": 10.0,
                "allow_reversing": False},
        }},
        "local_costmap": {"local_costmap": {"ros__parameters": costmap("odom", 8, 0.05, 5.0, 2.0, True, False)}},
        "global_costmap": {"global_costmap": {"ros__parameters": costmap("map", 120, 0.1, 2.0, 1.0, False, False)}},
        "planner_server": {"ros__parameters": {
            "use_sim_time": True, "expected_planner_frequency": 5.0, "planner_plugins": ["GridBased"],
            "GridBased": {"plugin": "nav2_navfn_planner::NavfnPlanner", "tolerance": 0.5, "use_astar": True,
                          "allow_unknown": True}}},
        "behavior_server": {"ros__parameters": {**defaults["behavior_server"]["ros__parameters"], "use_sim_time": True,
                                                "local_frame": "odom", "global_frame": "map",
                                                "robot_base_frame": "base_link", "transform_tolerance": 0.5}},
    }
    params["global_costmap"]["global_costmap"]["ros__parameters"].update(origin_x=-60.0, origin_y=-60.0)
    slam = yaml.safe_load(SLAM_DEFAULTS.read_text())["slam_toolbox"]["ros__parameters"]
    slam.update(use_sim_time=True, odom_frame="odom", map_frame="map", base_frame="base_link", scan_topic="/scan",
                mode="mapping", resolution=0.05, max_laser_range=20.0, minimum_travel_distance=0.2,
                minimum_travel_heading=0.2, map_update_interval=1.0, transform_timeout=0.5, tf_buffer_duration=30.0,
                do_loop_closing=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "nav2_params.yaml").write_text(yaml.safe_dump(params, sort_keys=False))
    (out / "slam_params.yaml").write_text(yaml.safe_dump({"slam_toolbox": {"ros__parameters": slam}}, sort_keys=False))
    print(f"wrote {out}/nav2_params.yaml and slam_params.yaml")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "params")
