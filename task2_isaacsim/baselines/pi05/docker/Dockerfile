ARG UV_VERSION=0.8.14
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ARG LEROBOT_COMMIT=30da8e687a6dfc617fcd94afc367ac7071c376ce

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache/xdg \
    HOME=/tmp/ebim-home

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libegl1 \
        libglib2.0-0 \
        libgl1 \
        libnss-wrapper \
        python3.12 \
        python3.12-venv \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
ENV PATH="/opt/lerobot/.venv/bin:${PATH}"

RUN git clone --filter=blob:none https://github.com/huggingface/lerobot.git /opt/lerobot \
    && git -C /opt/lerobot checkout --detach "${LEROBOT_COMMIT}"

COPY task2_isaacsim/baselines/pi05/patches/lerobot-v0.6.0-task2-relative-map.patch /tmp/task2-relative-map.patch
RUN git -C /opt/lerobot apply --check /tmp/task2-relative-map.patch \
    && git -C /opt/lerobot apply /tmp/task2-relative-map.patch \
    && cd /opt/lerobot \
    && uv sync --locked --extra training --extra pi --no-dev \
    && rm /tmp/task2-relative-map.patch

WORKDIR /opt/ebim
COPY task2_isaacsim/baselines/__init__.py /opt/ebim/task2_isaacsim/baselines/__init__.py
COPY task2_isaacsim/baselines/pi05 /opt/ebim/task2_isaacsim/baselines/pi05
COPY task2_isaacsim/baselines/pi05/docker/entrypoint.sh /usr/local/libexec/ebim-pi05-entrypoint
RUN python -c "import lerobot; from task2_isaacsim.baselines.pi05.contract import RELATIVE_ACTION_STATE_INDICES; assert len(RELATIVE_ACTION_STATE_INDICES) == 20"

ENV PYTHONPATH=/opt/ebim
ENTRYPOINT ["/usr/bin/tini", "--", "/bin/bash", "/usr/local/libexec/ebim-pi05-entrypoint"]
CMD ["doctor", "--profile", "smoke"]
