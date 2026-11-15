#!/bin/bash
HERE=$(cd "$(dirname "$0")" && pwd)
export ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}
source "$HERE/ros_env.sh"
export PUBLIC_IP=${PUBLIC_IP:-127.0.0.1} CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$HERE"
exec "$ISAACLAB_DIR/isaaclab.sh" -p run_warehouse.py --scene "$HERE/warehouse.usda" \
  --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint "${POLICY:-$HERE/policies/go2_flat/model_1499.pt}" \
  --experience "${EXP:-$HERE/apps/wh.python.rendering.kit}" physics=isaacsim_physx "$@"
