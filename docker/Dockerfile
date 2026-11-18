FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox \
    ros-jazzy-pointcloud-to-laserscan ros-jazzy-robot-state-publisher \
    ros-jazzy-rmw-fastrtps-cpp ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-teleop-twist-keyboard ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
