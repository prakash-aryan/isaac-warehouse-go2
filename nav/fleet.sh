#!/bin/bash
HERE=$(cd "$(dirname "$0")/.." && pwd)
N=${1:-2}
LAYOUT=${2:-$HERE/warehouse_layout.json}
MAP=${3:-$HERE/maps/warehouse.yaml}
DUR=${4:-0}
docker rm -f fleetnav >/dev/null 2>&1
docker run -d --name fleetnav --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -e ROS_LOG_DIR=/tmp/roslog \
  -e ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42} -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
  -e NAV_ROBOTS="$N" -e NAV_MAP=/maps/$(basename "$MAP") -e NAV_LAYOUT=/layout/$(basename "$LAYOUT") -e NAV_PARAMS=/tmp/fleet_params.yaml \
  -v "$HERE/nav":/nav -v "$(dirname "$(realpath "$MAP")")":/maps -v "$(dirname "$(realpath "$LAYOUT")")":/layout \
  jazzy-nav2:latest bash -c 'source /opt/ros/jazzy/setup.bash && cd /nav && python3 make_fleet_params.py /tmp/fleet_params.yaml && exec ros2 launch /nav/fleet_nav2.launch.py' >/dev/null
for try in 1 2 3; do
  sleep 40
  n=$(docker exec fleetnav bash -c "source /opt/ros/jazzy/setup.bash && timeout 10 ros2 action list" 2>/dev/null | grep -c navigate_to_pose)
  [ "$n" -ge "$N" ] && break
  echo "[fleet] Nav2 up for $n of $N robots, restarting"
  docker restart fleetnav >/dev/null
done
echo "[fleet] Nav2 running for $N robots"
docker exec fleetnav bash -c "source /opt/ros/jazzy/setup.bash && python3 -u /nav/fleet_patrol.py --layout /layout/$(basename "$LAYOUT") \
  --robots $N --stagger 0 --map /maps/$(basename "$MAP") --duration $DUR"
