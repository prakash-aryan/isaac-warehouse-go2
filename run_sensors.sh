#!/bin/bash
source "$(dirname "$0")/common.sh"
source "$HERE/ros_env.sh"
export CUDA_VISIBLE_DEVICES=${SENSOR_GPU:-2}
"$ISAACLAB_DIR/isaaclab.sh" -p sensors.py --scene "$SCENE" --robots "${ROBOTS:-8}" --port "${SENSOR_PORT:-47556}" \
  --experience wh.python.rendering.kit --viz none "$@" 2>&1 | grep --line-buffered -v -E "$NOISE"
