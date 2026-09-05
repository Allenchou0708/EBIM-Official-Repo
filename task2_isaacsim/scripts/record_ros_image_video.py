#!/usr/bin/env python3
"""Record a ROS 2 sensor_msgs/Image topic directly to an H.264 MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class ImageVideoRecorder(Node):
    def __init__(self, *, topic: str, output: Path, fps: float) -> None:
        super().__init__("image_video_recorder")
        self._output = output
        self._fps = fps
        self._encoder: subprocess.Popen[bytes] | None = None
        self._frame_count = 0
        qos = QoSProfile(
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Image, topic, self._on_image, qos)

    def _rgb(self, message: Image) -> np.ndarray:
        channels = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }.get(message.encoding.lower())
        if channels is None:
            raise ValueError(f"unsupported image encoding: {message.encoding}")
        rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
            message.height, message.step
        )
        image = rows[:, : message.width * channels].reshape(
            message.height, message.width, channels
        )
        encoding = message.encoding.lower()
        if encoding == "mono8":
            return np.repeat(image, 3, axis=2)
        if encoding in {"bgr8", "bgra8"}:
            return np.ascontiguousarray(image[:, :, 2::-1])
        return np.ascontiguousarray(image[:, :, :3])

    def _start_encoder(self, width: int, height: int) -> None:
        self._output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(self._fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(self._output),
        ]
        self._encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        print(
            f"recording {width}x{height} at {self._fps:g} fps to "
            f"{self._output}",
            flush=True,
        )

    def _on_image(self, message: Image) -> None:
        try:
            image = self._rgb(message)
            if self._encoder is None:
                self._start_encoder(message.width, message.height)
            assert self._encoder.stdin is not None
            self._encoder.stdin.write(image.tobytes())
            self._frame_count += 1
        except (BrokenPipeError, ValueError) as error:
            self.get_logger().error(str(error))
            rclpy.shutdown()

    def close(self) -> None:
        if self._encoder is not None:
            assert self._encoder.stdin is not None
            self._encoder.stdin.close()
            status = self._encoder.wait(timeout=30)
            if status != 0:
                raise RuntimeError(f"ffmpeg exited with status {status}")
        print(f"saved {self._frame_count} frames to {self._output}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/isaac/head_camera/image_raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    rclpy.init()
    recorder = ImageVideoRecorder(
        topic=args.topic, output=args.output, fps=args.fps
    )
    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
