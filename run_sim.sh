#!/bin/bash
source "$(dirname "$0")/common.sh"
source "$HERE/ros_env.sh"
export CUDA_VISIBLE_DEVICES=${SIM_GPU:-2}
"$ISAACLAB_DIR/isaaclab.sh" -p run_warehouse.py --scene "$SCENE" \
  --task Isaac-Velocity-Flat-UnitreeGo2 --checkpoint "$POLICY" \
  --experience wh.python.rendering.kit --viz none --lidars none --robots "${ROBOTS:-8}" \
  --render_every 100000 --pace --stream "${SENSOR_PORT:-47556}" physics=isaacsim_physx "$@" 2>&1 | grep --line-buffered -v -E "$NOISE"
