import os
import sys
from pathlib import Path

import yaml

from make_params import DEFAULTS

FOOTPRINT = "[[0.58, 0.26], [0.58, -0.26], [-0.52, -0.26], [-0.52, 0.26]]"


def scan_layer(obstacle_range, raytrace_range):
    return {
        "plugin": "nav2_costmap_2d::ObstacleLayer", "enabled": True, "observation_sources": "scan",
        "scan": {"topic": "scan", "data_type": "LaserScan", "marking": True, "clearing": True,
                 "obstacle_max_range": obstacle_range, "obstacle_min_range": 0.0,
                 "raytrace_max_range": raytrace_range, "raytrace_min_range": 0.0,
                 "max_obstacle_height": 2.0, "min_obstacle_height": -1.0, "inf_is_valid": True},
    }


def costmap(frame, rolling, size, resolution, update_hz, publish_hz):
    plugins = ["obstacle_layer", "fleet_layer", "inflation_layer"] if rolling else ["static_layer", "fleet_layer", "inflation_layer"]
    out = {
        "update_frequency": update_hz, "publish_frequency": publish_hz, "global_frame": frame,
        "robot_base_frame": "base_link", "use_sim_time": True, "rolling_window": rolling, "resolution": resolution,
        "footprint": FOOTPRINT, "footprint_padding": 0.04, "track_unknown_space": not rolling, "transform_tolerance": 0.5,
        "always_send_full_costmap": False, "plugins": plugins, "use_maximum": True,
        "obstacle_layer": scan_layer(6.0 if rolling else 12.0, 8.0 if rolling else 14.0),
        "fleet_layer": {"plugin": "nav2_costmap_2d::StaticLayer", "enabled": True, "map_topic": "fleet_map",
                        "map_subscribe_transient_local": True, "subscribe_to_updates": False, "use_maximum": True,
                        "footprint_clearing_enabled": False},
        "inflation_layer": {"plugin": "nav2_costmap_2d::InflationLayer", "cost_scaling_factor": 3.0,
                            "inflation_radius": 0.9},
    }
    if rolling:
        out.update(width=size, height=size)
    else:
        out["static_layer"] = {"plugin": "nav2_costmap_2d::StaticLayer", "map_topic": "/map",
                               "map_subscribe_transient_local": True, "subscribe_to_updates": False}
    return out


MPPI = {
    "plugin": "nav2_mppi_controller::MPPIController", "time_steps": 30, "model_dt": 0.1, "batch_size": 2000,
    "vx_std": 0.2, "vy_std": 0.2, "wz_std": 0.4, "vx_max": 0.7, "vx_min": -0.2, "vy_max": 0.0, "wz_max": 0.8,
    "ax_max": 2.0, "ax_min": -2.0, "ay_max": 0.0, "az_max": 3.0, "iteration_count": 1, "prune_distance": 2.0,
    "transform_tolerance": 0.5, "temperature": 0.3, "gamma": 0.015, "motion_model": "DiffDrive", "visualize": False,
    "retry_attempt_limit": 1, "regenerate_noises": False,
    "critics": ["ConstraintCritic", "CostCritic", "GoalCritic", "GoalAngleCritic", "PathAlignCritic", "PathFollowCritic",
                "PathAngleCritic", "PreferForwardCritic"],
    "ConstraintCritic": {"enabled": True, "cost_power": 1, "cost_weight": 4.0},
    "GoalCritic": {"enabled": True, "cost_power": 1, "cost_weight": 5.0, "threshold_to_consider": 1.4},
    "GoalAngleCritic": {"enabled": True, "cost_power": 1, "cost_weight": 3.0, "threshold_to_consider": 0.5},
    "PreferForwardCritic": {"enabled": True, "cost_power": 1, "cost_weight": 5.0, "threshold_to_consider": 0.5},
    "CostCritic": {"enabled": True, "cost_power": 1, "cost_weight": 3.81, "critical_cost": 300.0, "consider_footprint": True,
                   "collision_cost": 1000000.0, "near_goal_distance": 1.0, "trajectory_point_step": 2},
    "PathAlignCritic": {"enabled": True, "cost_power": 1, "cost_weight": 14.0, "max_path_occupancy_ratio": 0.05,
                        "trajectory_point_step": 4, "threshold_to_consider": 0.5, "offset_from_furthest": 20,
                        "use_path_orientations": False},
    "PathFollowCritic": {"enabled": True, "cost_power": 1, "cost_weight": 5.0, "offset_from_furthest": 5,
                         "threshold_to_consider": 1.4},
    "PathAngleCritic": {"enabled": True, "cost_power": 1, "cost_weight": 2.0, "offset_from_furthest": 4,
                        "threshold_to_consider": 0.5, "max_angle_to_furthest": 1.0, "mode": 0},
}


def main(out_file):
    defaults = yaml.safe_load(DEFAULTS.read_text())
    p = {
        "/**/bt_navigator": {"ros__parameters": {**defaults["bt_navigator"]["ros__parameters"], "global_frame": "map",
                                                 "robot_base_frame": "base_link", "odom_topic": "odom", "use_sim_time": True,
                                                 "transform_tolerance": 0.5, "default_nav_to_pose_bt_xml": "/nav/bt_wait.xml"}},
        "/**/controller_server": {"ros__parameters": {
            "use_sim_time": True, "controller_frequency": 10.0, "odom_topic": "odom", "min_x_velocity_threshold": 0.001,
            "min_y_velocity_threshold": 0.5, "min_theta_velocity_threshold": 0.001, "failure_tolerance": 0.5,
            "progress_checker_plugins": ["progress_checker"], "goal_checker_plugins": ["general_goal_checker"],
            "controller_plugins": ["FollowPath"],
            "progress_checker": {"plugin": "nav2_controller::SimpleProgressChecker", "required_movement_radius": 0.3,
                                 "movement_time_allowance": 20.0},
            "general_goal_checker": {"plugin": "nav2_controller::SimpleGoalChecker", "stateful": True,
                                     "xy_goal_tolerance": 0.6, "yaw_goal_tolerance": 3.14},
            "FollowPath": {
                "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
                "desired_linear_vel": 0.7, "lookahead_dist": 1.2, "min_lookahead_dist": 0.8, "max_lookahead_dist": 1.8,
                "lookahead_time": 1.5, "rotate_to_heading_angular_vel": 0.5, "transform_tolerance": 0.5,
                "use_velocity_scaled_lookahead_dist": True, "min_approach_linear_velocity": 0.35,
                "approach_velocity_scaling_dist": 0.4, "use_collision_detection": True,
                "max_allowed_time_to_collision_up_to_carrot": 1.5, "use_regulated_linear_velocity_scaling": True,
                "use_cost_regulated_linear_velocity_scaling": False, "regulated_linear_scaling_min_radius": 1.2,
                "regulated_linear_scaling_min_speed": 0.35, "use_rotate_to_heading": True,
                "rotate_to_heading_min_angle": 1.0, "max_angular_accel": 6.0, "max_robot_pose_search_dist": 10.0,
                "allow_reversing": False},
        }},
        "/**/local_costmap/local_costmap": {"ros__parameters": costmap("odom", True, 6, 0.05, 5.0, 2.0)},
        "/**/global_costmap/global_costmap": {"ros__parameters": costmap("map", False, 0, 0.05, 1.0, 0.5)},
        "/**/planner_server": {"ros__parameters": {
            "use_sim_time": True, "expected_planner_frequency": 2.0, "planner_plugins": ["GridBased"],
            "GridBased": {"plugin": "nav2_navfn_planner::NavfnPlanner", "tolerance": 0.5, "use_astar": True,
                          "allow_unknown": False}}},
        "/**/behavior_server": {"ros__parameters": {**defaults["behavior_server"]["ros__parameters"], "use_sim_time": True,
                                                    "local_frame": "odom", "global_frame": "map", "robot_base_frame": "base_link",
                                                    "transform_tolerance": 0.5, "max_rotational_vel": 0.5,
                                                    "min_rotational_vel": 0.3, "rotational_acc_lim": 1.5}},
        "/**/collision_monitor": {"ros__parameters": {
            "use_sim_time": True, "base_frame_id": "base_link", "odom_frame_id": "odom",
            "cmd_vel_in_topic": "cmd_vel_nav", "cmd_vel_out_topic": "cmd_vel", "state_topic": "collision_monitor_state",
            "transform_tolerance": 0.5, "source_timeout": 1.0, "base_shift_correction": True, "stop_pub_timeout": 2.0,
            "polygons": ["Approach", "SlowZone"],
            "Approach": {"type": "polygon", "points": "[[0.6, 0.27], [0.6, -0.27], [-0.5, -0.27], [-0.5, 0.27]]",
                         "action_type": "approach", "time_before_collision": 1.2, "simulation_time_step": 0.05,
                         "min_points": 2, "visualize": False, "enabled": True},
            "SlowZone": {"type": "polygon", "points": "[[1.6, 0.5], [1.6, -0.5], [-0.5, -0.5], [-0.5, 0.5]]",
                         "action_type": "limit", "linear_limit": 0.42, "angular_limit": 1.0, "min_points": 3,
                         "visualize": False, "enabled": True},
            "observation_sources": ["scan", "fleet"],
            "scan": {"type": "scan", "topic": "scan", "enabled": True},
            "fleet": {"type": "pointcloud", "topic": "fleet_points", "min_height": 0.0, "max_height": 1.0, "enabled": True}}},
        "/**/amcl": {"ros__parameters": {
            "use_sim_time": True, "base_frame_id": "base_link", "odom_frame_id": "odom", "global_frame_id": "map",
            "scan_topic": "scan", "map_topic": "/map", "robot_model_type": "nav2_amcl::DifferentialMotionModel",
            "laser_model_type": "likelihood_field", "max_beams": 60, "laser_max_range": 30.0, "laser_min_range": 0.2,
            "min_particles": 300, "max_particles": 1500, "update_min_d": 0.1, "update_min_a": 0.1,
            "transform_tolerance": 0.5, "tf_broadcast": True, "set_initial_pose": True,
            "alpha1": 0.1, "alpha2": 0.1, "alpha3": 0.1, "alpha4": 0.1, "alpha5": 0.1}},
    }
    if os.environ.get("NAV_CONTROLLER", "rpp") == "mppi":
        p["/**/controller_server"]["ros__parameters"]["FollowPath"] = MPPI
        p["/**/controller_server"]["ros__parameters"]["controller_frequency"] = 10.0
    Path(out_file).write_text(yaml.safe_dump(p, sort_keys=False))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fleet_params.yaml")
