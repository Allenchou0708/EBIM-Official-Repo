# EBiM 2026 Task 2 PI0.5 inference image.
# Build from the EBiM-Official-Repo root:
#   docker build -f task2_submission/Dockerfile -t ebim-task2-pi05:submission .
ARG TRAINING_IMAGE=ghcr.io/allenchou0708/ebim-task2-pi05@sha256:e69f329e94be38bc1b1431c35ee556c846c9ff4dbd2bb1036f1971961bd5e1a3
FROM ${TRAINING_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg software-properties-common \
    && add-apt-repository universe \
    && curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends ros-jazzy-ros-base \
    && rm -rf /var/lib/apt/lists/*

COPY task2_isaacsim /workspace/EBiM_Challenge/task2_isaacsim
COPY task2_submission/entrypoint.sh /submission-entrypoint.sh
RUN chmod 0755 /submission-entrypoint.sh

ENV PYTHONPATH=/workspace/EBiM_Challenge:/opt/ros/jazzy/lib/python3.12/site-packages \
    ROS_DISTRO=jazzy \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /workspace/EBiM_Challenge
ENTRYPOINT ["/submission-entrypoint.sh"]
CMD ["help"]
