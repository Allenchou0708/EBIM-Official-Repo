#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0
"""Replay the successful episode-19 grasp/place joints from GT-aligned poses."""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, String

from task2_isaacsim.baselines.pi05.live.ground_truth_pregrasp import (
    _normalize_quaternion,
    _quaternion_multiply_xyzw,
    _slerp,
    _yaw_from_wxyz,
)
from task2_isaacsim.common.state_contract import (
    LEFT_JOINTS,
    RIGHT_JOINTS,
    SPINE_JOINT,
)
from task2_isaacsim.scripts.topics import load_topics


TOPICS = load_topics()
LEFT_COMMAND_TOPIC = TOPICS["bridge"]["joint_groups"]["left_arm"]["command"]
LEFT_STATE_TOPIC = TOPICS["bridge"]["joint_groups"]["left_arm"]["state"]
RIGHT_COMMAND_TOPIC = TOPICS["bridge"]["joint_groups"]["right_arm"]["command"]
RIGHT_STATE_TOPIC = TOPICS["bridge"]["joint_groups"]["right_arm"]["state"]
LEFT_GRIPPER_TOPIC = TOPICS["cartesian_control"][
    "gripper_open_fraction_target"
]["left"]
RIGHT_GRIPPER_TOPIC = TOPICS["cartesian_control"][
    "gripper_open_fraction_target"
]["right"]
BASE_HOLD_TOPIC = TOPICS["cartesian_control"]["base_hold_target"]

REFERENCE_PAD_XYYAW = (1.75, 1.9500000477, math.pi / 2.0)
REFERENCE_TARGET_XYYAW = (2.1500000954, 1.9500000477, math.pi / 2.0)
REFERENCE_BASE_XYYAW = (
    2.0999855995,
    3.0731513500,
    -1.5703129768,
)
REFERENCE_BASE_Z = -0.001785
GRASP_DEPTH_BIAS_M = 0.0
REFERENCE_PRECONTACT_QUATERNION_XYZW = (
    -0.04021255299448967,
    0.9086313843727112,
    -0.4156099557876587,
    -0.006350508891046047,
)
REFERENCE_PLACE_QUATERNION_XYZW = (
    -0.03955800458788872,
    0.9875929951667786,
    -0.15194131433963776,
    0.003020714968442917,
)
REFERENCE_CONTACT_SWEEP_XY = (0.050, 0.010)
# Exact left-arm action target at successful dataset episode 19, frame 399.
# The arm is not static at home in the demonstration; reproducing it is part
# of the learned-policy observation and left-wrist camera geometry.
LEFT_PREGRASP_Q399 = (
    -0.2740990222,
    -0.3937259018,
    1.3518345356,
    -2.5705306530,
    0.5819327831,
    2.3917682171,
    1.3968001604,
)
# Post-arbitration targets from successful dataset episode 19.  Runtime uses
# only their deltas from frame 399, added to the live GT-aligned joint state.
JOINT_LANDMARKS = (
    (399, (0.7885726094, -1.7149579525, -1.8527580500, -2.3938157558, -0.8599357605, 3.7260856628, -0.4352681339), 1.0),
    (400, (0.7889355421, -1.7147537470, -1.8530831337, -2.3935761452, -0.8604794145, 3.7264652252, -0.4347275496), 0.0),
    (410, (0.7423399687, -1.7150638103, -1.8775746822, -2.4286305904, -0.8951972723, 3.7219901085, -0.4200360179), 0.0),
    (424, (0.8595721722, -1.6187088490, -1.8019258976, -2.4438645840, -0.7162916660, 3.8303887844, -0.4819193184), 0.0),
    (428, (0.9078787565, -1.5843703747, -1.7687107325, -2.4349033833, -0.6527580619, 3.8624584675, -0.4964440465), 0.0),
    (432, (0.9594421387, -1.5524977446, -1.7365283966, -2.4162809849, -0.5932618380, 3.8893420696, -0.5078317523), 0.0),
    (436, (1.0142579079, -1.5269838572, -1.7089647055, -2.3845071793, -0.5428704619, 3.9076528549, -0.5162927508), 0.0),
    (440, (1.0677460432, -1.5060102940, -1.6874445677, -2.3483734131, -0.5010870099, 3.9213037491, -0.5232530236), 0.0),
    (444, (1.1182349920, -1.4868384600, -1.6709809303, -2.3154091835, -0.4643150866, 3.9357354641, -0.5303720832), 0.0),
    (448, (1.1474568844, -1.4759111404, -1.6625598669, -2.2959055901, -0.4439349174, 3.9437921047, -0.5343841314), 0.0),
    (450, (1.1596411467, -1.4700100422, -1.6588783264, -2.2898967266, -0.4339841306, 3.9492475986, -0.5365116596), 0.0),
    (460, (1.1814332008, -1.4587154388, -1.6527100801, -2.2793896198, -0.4162698686, 3.9594631195, -0.5396893024), 0.0),
    (470, (1.2108048201, -1.4698035717, -1.6701500416, -2.2739734650, -0.4271338880, 3.9841964245, -0.5467279553), 0.0),
    (480, (1.2800550461, -1.5108091831, -1.7192924023, -2.2572767735, -0.4686418474, 4.0409588814, -0.5638266802), 0.0),
    (490, (1.3641756773, -1.5647782087, -1.7739905119, -2.2216129303, -0.5218674541, 4.0986962318, -0.5788876414), 0.0),
    (500, (1.4590312243, -1.6259269714, -1.8284658194, -2.1632061005, -0.5825185776, 4.1520547867, -0.5844883919), 0.0),
    (510, (1.5672756433, -1.6860008240, -1.8836717606, -2.0790402889, -0.6465449929, 4.2050399780, -0.5703322291), 0.0),
    (520, (1.6798437834, -1.7235401869, -1.9373990297, -1.9791699648, -0.6949417591, 4.2577381134, -0.5278141499), 0.0),
    (530, (1.8233025074, -1.7436655760, -2.0042936802, -1.8441941738, -0.7225244045, 4.3233757019, -0.4495747685), 0.0),
    (540, (1.9632176161, -1.7532464266, -2.0696740150, -1.7055747509, -0.7192561626, 4.3802819252, -0.3623764515), 0.0),
    (544, (2.0209493637, -1.7561228275, -2.0990011692, -1.6445274353, -0.7110523582, 4.4003977776, -0.3216704726), 0.0),
    (550, (2.0869235992, -1.7575260401, -2.1367602348, -1.5718412399, -0.6958870888, 4.4210119247, -0.2695985734), 0.0),
    (560, (2.1262383461, -1.7534098625, -2.1644954681, -1.5302928686, -0.6805466413, 4.4343061447, -0.2342208475), 0.0),
    (570, (2.1444017887, -1.7456810474, -2.1169040203, -1.5692254305, -0.5636597872, 4.4656338692, -0.3218007982), 0.0),
    (575, (2.1790821552, -1.7421998978, -2.0307934284, -1.5951075554, -0.3967229426, 4.5155253410, -0.4448097944), 0.0),
    (580, (2.2339007854, -1.7428473234, -1.9416904449, -1.5699927807, -0.2408988178, 4.5664386749, -0.5467184782), 0.0),
    (585, (2.3200755119, -1.7802520990, -1.7914535999, -1.4642337561, -0.1061648577, 4.6099057198, -0.6898595691), 0.0),
    (590, (2.3829104900, -1.7808314562, -1.7430908680, -1.3565442562, -0.0288176313, 4.6078405380, -0.7282894254), 0.0),
    (595, (2.4087414742, -1.7683380842, -1.7487936020, -1.3070541620, 0.0229083840, 4.6078329086, -0.7238755226), 0.0),
    (600, (2.4339790344, -1.7019202709, -1.8412007093, -1.2877006531, 0.1305984408, 4.6080603600, -0.6520813107), 0.0),
    (610, (2.4553315639, -1.5687577724, -2.0526750088, -1.3307958841, 0.3146459460, 4.5963549614, -0.4829337597), 0.0),
    (620, (2.4421072006, -1.5537986755, -2.1102597713, -1.3455054760, 0.3334635198, 4.5780534744, -0.4325684309), 0.0),
    (640, (2.3953521252, -1.5980528593, -2.1311647892, -1.3257864714, 0.2763074934, 4.5340423584, -0.4104880989), 0.0),
    (660, (2.3848857880, -1.6432936192, -2.1195356846, -1.2848665714, 0.2321315855, 4.5090360641, -0.4173381031), 0.0),
    (680, (2.3671743870, -1.6685420275, -2.1031470299, -1.2805528641, 0.1996020824, 4.5045809746, -0.4271553159), 0.0),
    (700, (2.3505640030, -1.6817674637, -2.0852789879, -1.2974538803, 0.1775392741, 4.5161194801, -0.4387686253), 0.0),
    (720, (2.3313512802, -1.6835333109, -2.0783882141, -1.3402298689, 0.1650166065, 4.5437254906, -0.4401488006), 0.0),
    (740, (2.3224370480, -1.6827074289, -2.0814864635, -1.3596571684, 0.1612046957, 4.5543537140, -0.4354578257), 0.0),
    (760, (2.2979094982, -1.6947876215, -2.1001534462, -1.3707847595, 0.1392024457, 4.5474147797, -0.4159819484), 0.0),
    (770, (2.3016786575, -1.6974114180, -2.0938110352, -1.3593429327, 0.1891828775, 4.5595631599, -0.4295451939), 0.0),
    (780, (2.3142859936, -1.6894094944, -2.1032896042, -1.3416973352, 0.2746699154, 4.5732083321, -0.4329054654), 0.0),
    (790, (2.3439998627, -1.6689455509, -2.1301107407, -1.3061997890, 0.3919198215, 4.5760722160, -0.4286847115), 0.0),
    (800, (2.4306094646, -1.6733947992, -2.1018447876, -1.1350140572, 0.5706534386, 4.5634293556, -0.5183476210), 0.0),
    (810, (2.4464576244, -1.6748716831, -2.1169455051, -1.0948325396, 0.6338057518, 4.5588617325, -0.5293304920), 0.0),
    (820, (2.4394929409, -1.6461492777, -2.1750416756, -1.1421976089, 0.7001007199, 4.5667190552, -0.4829126596), 0.0),
    (830, (2.5109913349, -1.6651560068, -2.1602890491, -0.9805489779, 0.8065087795, 4.5496301651, -0.5708952546), 0.0),
    (834, (2.5324571133, -1.6724518538, -2.1554434300, -0.9319114685, 0.8351953030, 4.5473957062, -0.5975975394), 0.0),
    (835, (2.5350086689, -1.6726378202, -2.1560673714, -0.9271779656, 0.8409535289, 4.5480074883, -0.6000602245), 1.0),
    (900, (2.5176262856, -1.0585579872, -2.4168953896, -1.4820151329, 1.3604292870, 4.5521087646, -0.4577407241), 1.0),
    (949, (2.1654112339, -0.6537351012, -2.1848337650, -1.7842056751, 1.5074640512, 4.5173859596, -0.6208035350), 1.0),
)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def anchored_base_pose(
    live_anchor_wxyz: tuple[float, ...],
    reference_anchor_xyyaw: tuple[float, float, float],
    *,
    max_yaw_delta_rad: float | None = None,
) -> tuple[float, float, float]:
    """Map the episode base pose through a live planar anchor transform."""
    qw, qx, qy, qz = live_anchor_wxyz[3:7]
    live_yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    delta = _wrap_angle(live_yaw - reference_anchor_xyyaw[2])
    if max_yaw_delta_rad is not None:
        delta = max(-max_yaw_delta_rad, min(max_yaw_delta_rad, delta))
    offset_x = REFERENCE_BASE_XYYAW[0] - reference_anchor_xyyaw[0]
    offset_y = REFERENCE_BASE_XYYAW[1] - reference_anchor_xyyaw[1]
    cosine, sine = math.cos(delta), math.sin(delta)
    return (
        live_anchor_wxyz[0] + cosine * offset_x - sine * offset_y,
        live_anchor_wxyz[1] + sine * offset_x + cosine * offset_y,
        _wrap_angle(REFERENCE_BASE_XYYAW[2] + delta),
    )


def interpolate_base_pose(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    fraction = max(0.0, min(1.0, fraction))
    return (
        start[0] + fraction * (end[0] - start[0]),
        start[1] + fraction * (end[1] - start[1]),
        _wrap_angle(start[2] + fraction * _wrap_angle(end[2] - start[2])),
    )


def bounded_base_pose_step(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    *,
    linear_speed_mps: float,
    angular_speed_rps: float,
    elapsed_s: float,
) -> tuple[float, float, float]:
    """Interpolate one base step while bounding translation and yaw speed."""
    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    yaw_error = abs(_wrap_angle(end[2] - start[2]))
    elapsed_s = max(0.0, elapsed_s)
    linear_fraction = (
        1.0
        if distance <= 1.0e-9
        else min(1.0, max(0.0, linear_speed_mps) * elapsed_s / distance)
    )
    angular_fraction = (
        1.0
        if yaw_error <= 1.0e-9
        else min(1.0, max(0.0, angular_speed_rps) * elapsed_s / yaw_error)
    )
    return interpolate_base_pose(start, end, min(linear_fraction, angular_fraction))


def deepen_grasp_base_pose(
    base: tuple[float, float, float],
    live_pad_wxyz: tuple[float, ...],
    depth_m: float = GRASP_DEPTH_BIAS_M,
) -> tuple[float, float, float]:
    toward_x = live_pad_wxyz[0] - base[0]
    toward_y = live_pad_wxyz[1] - base[1]
    norm = math.hypot(toward_x, toward_y)
    if norm < 1.0e-9:
        return base
    return (
        base[0] + depth_m * toward_x / norm,
        base[1] + depth_m * toward_y / norm,
        base[2],
    )


def interpolate_landmark(frame: float) -> tuple[tuple[float, ...], float]:
    for index in range(1, len(JOINT_LANDMARKS)):
        left_frame, left_q, left_grip = JOINT_LANDMARKS[index - 1]
        right_frame, right_q, right_grip = JOINT_LANDMARKS[index]
        if frame <= right_frame:
            fraction = max(0.0, (frame - left_frame) / (right_frame - left_frame))
            q = tuple(a + fraction * (b - a) for a, b in zip(left_q, right_q))
            grip = left_grip + fraction * (right_grip - left_grip)
            return q, grip
    return JOINT_LANDMARKS[-1][1], JOINT_LANDMARKS[-1][2]


def bounded_axis_step(error: float, speed: float, elapsed: float) -> float:
    """Return a signed Cartesian correction bounded by speed and elapsed."""
    maximum = max(0.0, speed) * max(0.0, elapsed)
    return max(-maximum, min(maximum, error))


def bounded_planar_offset(
    x: float, y: float, maximum_distance: float
) -> tuple[float, float]:
    """Clamp a planar offset to a circular displacement envelope."""
    distance = math.hypot(x, y)
    limit = max(0.0, maximum_distance)
    if distance <= limit or distance <= 1.0e-9:
        return x, y
    scale = limit / distance
    return scale * x, scale * y


def quaternion_error_deg(
    first: tuple[float, ...], second: tuple[float, ...]
) -> float:
    first = _normalize_quaternion(first)
    second = _normalize_quaternion(second)
    dot = abs(sum(a * b for a, b in zip(first, second, strict=True)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def bounded_orientation_step(
    current: tuple[float, ...],
    target: tuple[float, ...],
    speed_deg_s: float,
    elapsed: float,
) -> tuple[float, ...]:
    """Slerp toward target without exceeding an angular-speed bound."""
    error_deg = quaternion_error_deg(current, target)
    if error_deg <= 1.0e-9:
        return _normalize_quaternion(target)
    maximum_step_deg = max(0.0, speed_deg_s) * max(0.0, elapsed)
    return _slerp(current, target, min(1.0, maximum_step_deg / error_deg))


def yaw_rotated_reference_quaternion(
    reference: tuple[float, ...], yaw_delta: float
) -> tuple[float, ...]:
    yaw = (0.0, 0.0, math.sin(yaw_delta / 2.0), math.cos(yaw_delta / 2.0))
    return _quaternion_multiply_xyzw(yaw, reference)


def yaw_rotated_planar_offset(
    offset: tuple[float, float], yaw_delta: float
) -> tuple[float, float]:
    cosine, sine = math.cos(yaw_delta), math.sin(yaw_delta)
    return (
        cosine * offset[0] - sine * offset[1],
        sine * offset[0] + cosine * offset[1],
    )


def placement_release_ready(
    *,
    xy_error_m: float,
    pad_height_m: float,
    orientation_error_deg: float,
    release_xy_m: float,
    release_height_m: float,
    release_height_tolerance_m: float,
    orientation_tolerance_deg: float,
    require_target_xy: bool = True,
) -> bool:
    return (
        (not require_target_xy or xy_error_m <= release_xy_m)
        and abs(pad_height_m - release_height_m)
        <= release_height_tolerance_m
        and orientation_error_deg <= orientation_tolerance_deg
    )


class JointLiftNode(Node):
    def __init__(self) -> None:
        super().__init__("task2_ground_truth_joint_lift")
        self.sim_time: float | None = None
        self.left_joints: tuple[float, ...] | None = None
        self.joints: tuple[float, ...] | None = None
        self.pad_centroid: tuple[float, float, float] | None = None
        self.pad_z_span_m: float | None = None
        self.objects: dict[str, tuple[float, ...]] = {}
        self.base: tuple[float, float, float] | None = None
        self.right_ee: tuple[float, ...] | None = None
        self.publish_count = 0
        self.create_subscription(
            Clock, TOPICS["clock"], self._on_clock, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState, LEFT_STATE_TOPIC, self._on_left_joints, 10
        )
        self.create_subscription(
            JointState, RIGHT_STATE_TOPIC, self._on_joints, 10
        )
        self.create_subscription(
            Float32MultiArray,
            TOPICS["ground_truth"]["pad_points"],
            self._on_pad,
            10,
        )
        self.create_subscription(
            String,
            TOPICS["ground_truth"]["object_poses"],
            self._on_objects,
            10,
        )
        self.create_subscription(
            Odometry,
            TOPICS["recording"]["odom"],
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            TOPICS["recording"]["ee_pose"]["right"],
            self._on_right_ee,
            qos_profile_sensor_data,
        )
        self.left_joint_pub = self.create_publisher(
            JointState, LEFT_COMMAND_TOPIC, 10
        )
        self.joint_pub = self.create_publisher(JointState, RIGHT_COMMAND_TOPIC, 10)
        self.left_gripper_pub = self.create_publisher(
            JointState, LEFT_GRIPPER_TOPIC, 10
        )
        self.gripper_pub = self.create_publisher(JointState, RIGHT_GRIPPER_TOPIC, 10)
        self.base_pub = self.create_publisher(PoseStamped, BASE_HOLD_TOPIC, 10)
        self.ee_pub = self.create_publisher(
            PoseStamped, TOPICS["cartesian_control"]["ee_target"]["right"], 10
        )

    def _on_clock(self, message: Clock) -> None:
        self.sim_time = float(message.clock.sec) + float(message.clock.nanosec) * 1e-9

    def _on_joints(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in RIGHT_JOINTS):
            self.joints = tuple(float(by_name[name]) for name in RIGHT_JOINTS)

    def _on_left_joints(self, message: JointState) -> None:
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in LEFT_JOINTS):
            self.left_joints = tuple(
                float(by_name[name]) for name in LEFT_JOINTS
            )

    def _on_pad(self, message: Float32MultiArray) -> None:
        data = tuple(float(value) for value in message.data)
        if len(data) < 5:
            return
        count = int(round(data[1]))
        if count <= 0 or len(data) != 2 + 3 * count:
            return
        values = data[2:]
        self.pad_centroid = tuple(sum(values[axis::3]) / count for axis in range(3))
        z_values = values[2::3]
        self.pad_z_span_m = max(z_values) - min(z_values)

    def _on_objects(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            objects = {
                str(name): tuple(float(value) for value in pose)
                for name, pose in payload["objects"].items()
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if all(
            len(pose) == 7 and all(math.isfinite(value) for value in pose)
            for pose in objects.values()
        ):
            self.objects = objects

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        yaw = math.atan2(
            2.0
            * (
                pose.orientation.w * pose.orientation.z
                + pose.orientation.x * pose.orientation.y
            ),
            1.0
            - 2.0
            * (
                pose.orientation.y * pose.orientation.y
                + pose.orientation.z * pose.orientation.z
            ),
        )
        self.base = (pose.position.x, pose.position.y, yaw)

    def _on_right_ee(self, message: PoseStamped) -> None:
        pose = message.pose
        self.right_ee = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

    def publish_base(self, base_xyyaw: tuple[float, float, float]) -> None:
        message = PoseStamped()
        message.header.frame_id = "world"
        message.pose.position.x = base_xyyaw[0]
        message.pose.position.y = base_xyyaw[1]
        message.pose.position.z = REFERENCE_BASE_Z
        message.pose.orientation.z = math.sin(base_xyyaw[2] / 2.0)
        message.pose.orientation.w = math.cos(base_xyyaw[2] / 2.0)
        self.base_pub.publish(message)

    def publish_joint(self, joints: tuple[float, ...]) -> None:
        arm = JointState()
        arm.name = list(RIGHT_JOINTS)
        arm.position = list(joints)
        self.joint_pub.publish(arm)

    def publish_left_joint(self, joints: tuple[float, ...]) -> None:
        arm = JointState()
        arm.name = list(LEFT_JOINTS)
        arm.position = list(joints)
        self.left_joint_pub.publish(arm)

    def publish_gripper(self, gripper: float) -> None:
        grip = JointState()
        grip.name = ["right_gripper_open_fraction"]
        grip.position = [float(gripper)]
        self.gripper_pub.publish(grip)

    def publish_left_gripper(self, gripper: float) -> None:
        grip = JointState()
        grip.name = ["left_gripper_open_fraction"]
        grip.position = [float(gripper)]
        self.left_gripper_pub.publish(grip)

    def publish_left_pregrasp(
        self, joints: tuple[float, ...], gripper: float = 1.0
    ) -> None:
        self.publish_left_joint(joints)
        self.publish_left_gripper(gripper)

    def publish(self, joints: tuple[float, ...], gripper: float) -> None:
        self.publish_joint(joints)
        self.publish_gripper(gripper)
        self.publish_count += 1

    def publish_ee(self, target: tuple[float, ...]) -> None:
        message = PoseStamped()
        message.header.frame_id = "world"
        message.pose.position.x = target[0]
        message.pose.position.y = target[1]
        message.pose.position.z = target[2]
        message.pose.orientation.x = target[3]
        message.pose.orientation.y = target[4]
        message.pose.orientation.z = target[5]
        message.pose.orientation.w = target[6]
        self.ee_pub.publish(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration-s", type=float, default=180.0)
    parser.add_argument("--lift-height-m", type=float, default=0.13)
    parser.add_argument(
        "--absolute-dataset-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--stop-frame", type=int, default=949)
    parser.add_argument(
        "--pregrasp-only",
        action="store_true",
        help=(
            "stop after the verified absolute episode-19 frame-399 joint "
            "preposition, before any grasp command"
        ),
    )
    parser.add_argument("--preposition-tolerance-rad", type=float, default=0.065)
    parser.add_argument("--trajectory-rate-hz", type=float, default=40.0)
    parser.add_argument("--grasp-dwell-s", type=float, default=0.5)
    parser.add_argument("--grasp-depth-bias-m", type=float, default=0.005)
    parser.add_argument("--grasp-yaw-limit-deg", type=float, default=5.0)
    parser.add_argument("--grasp-joint-tolerance-rad", type=float, default=0.02)
    parser.add_argument(
        "--preposition-at-live-grasp-base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "legacy direct root-pose prepositioning; the safe default latches "
            "measured odometry and bounds the remaining GT alignment"
        ),
    )
    parser.add_argument("--preposition-base-speed-mps", type=float, default=0.10)
    parser.add_argument(
        "--preposition-base-angular-speed-rps", type=float, default=0.30
    )
    parser.add_argument(
        "--preposition-base-position-tolerance-m", type=float, default=0.002
    )
    parser.add_argument(
        "--preposition-base-yaw-tolerance-rad", type=float, default=0.01
    )
    parser.add_argument(
        "--transport-base-alignment-frame",
        type=float,
        default=0.0,
        help="Pause at this elevated frame, align pad XY with the GT target via the base, then resume; 0 disables it.",
    )
    parser.add_argument("--transport-base-speed-mps", type=float, default=0.10)
    parser.add_argument(
        "--transport-base-alignment-max-initial-xy-m",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--transport-base-alignment-max-displacement-m",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--feedback-release-frame",
        type=float,
        default=0.0,
        help="Pause at this frame for experimental GT base alignment; 0 disables it.",
    )
    parser.add_argument("--cartesian-handoff-frame", type=float, default=530.0)
    parser.add_argument("--cartesian-speed-mps", type=float, default=0.30)
    parser.add_argument("--cartesian-max-xy-displacement-m", type=float, default=0.16)
    parser.add_argument("--cartesian-descent-start-xy-m", type=float, default=0.040)
    parser.add_argument("--cartesian-descent-speed-mps", type=float, default=0.080)
    parser.add_argument(
        "--precontact-lift-m",
        type=float,
        default=0.105,
        help="Pad-centroid height above the table when its lower edge first touches while held upright.",
    )
    parser.add_argument("--precontact-lift-speed-mps", type=float, default=0.080)
    parser.add_argument("--precontact-angular-speed-deg-s", type=float, default=45.0)
    parser.add_argument("--precontact-orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--contact-height-tolerance-m", type=float, default=0.006)
    parser.add_argument(
        "--minimum-contact-ee-z-m",
        type=float,
        default=0.0,
        help=(
            "Abort before contact rotation when measured EE world Z is below "
            "this scene-audited gripper-clearance floor; 0 disables the gate."
        ),
    )
    parser.add_argument(
        "--contact-ee-tracking-margin-m",
        type=float,
        default=0.007,
        help="Commanded EE-Z margin above the measured clearance floor.",
    )
    parser.add_argument(
        "--contact-ee-clearance-tolerance-m",
        type=float,
        default=0.001,
        help="Tolerance for measured EE-Z noise at the clearance floor.",
    )
    parser.add_argument("--contact-angular-speed-deg-s", type=float, default=25.0)
    parser.add_argument("--contact-height-correction-speed-mps", type=float, default=0.015)
    parser.add_argument("--contact-wrist-z-drop-m", type=float, default=0.005)
    parser.add_argument(
        "--contact-sweep-x-m", type=float, default=REFERENCE_CONTACT_SWEEP_XY[0]
    )
    parser.add_argument(
        "--contact-sweep-y-m", type=float, default=REFERENCE_CONTACT_SWEEP_XY[1]
    )
    parser.add_argument("--place-orientation-tolerance-deg", type=float, default=3.0)
    parser.add_argument("--cartesian-handoff-settle-s", type=float, default=0.05)
    parser.add_argument("--alignment-release-xy-m", type=float, default=0.025)
    parser.add_argument("--alignment-release-height-m", type=float, default=0.006)
    parser.add_argument(
        "--alignment-release-height-tolerance-m", type=float, default=0.008
    )
    parser.add_argument("--alignment-stable-s", type=float, default=0.50)
    parser.add_argument("--post-release-settle-s", type=float, default=0.20)
    parser.add_argument("--post-release-retract-m", type=float, default=0.080)
    parser.add_argument("--post-release-retract-speed-mps", type=float, default=0.10)
    parser.add_argument("--post-release-retract-tolerance-m", type=float, default=0.010)
    parser.add_argument("--success-xy-m", type=float, default=0.020)
    parser.add_argument("--success-z-m", type=float, default=0.012)
    parser.add_argument(
        "--placement-contract",
        choices=("nominal", "randomized-flat"),
        default="nominal",
        help=(
            "nominal requires target XY overlap; randomized-flat accepts the "
            "same contact/rotate/release motion anywhere near the table"
        ),
    )
    parser.add_argument(
        "--flat-pad-z-span-m",
        type=float,
        default=0.010,
        help="Maximum GT pad-mesh Z span accepted as flat after release.",
    )
    args = parser.parse_args()
    rclpy.init()
    node = JointLiftNode()
    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    started_wall = time.monotonic()
    started_sim: float | None = None
    live_start: tuple[float, ...] | None = None
    live_left_start: tuple[float, ...] | None = None
    baseline_height: float | None = None
    initial_pad_centroid: tuple[float, float, float] | None = None
    trajectory_started_sim: float | None = None
    preposition_stable_since: float | None = None
    final_preposition_error: float | None = None
    final_left_preposition_error: float | None = None
    final_right_preposition_error: float | None = None
    preposition_base: tuple[float, float, float] | None = None
    approach_base: tuple[float, float, float] | None = None
    base_preposition_completed_sim: float | None = None
    joint_preposition_completed_sim: float | None = None
    base_approach_stable_since: float | None = None
    grasp_ready_stable_since: float | None = None
    last_preposition_sim: float | None = None
    maximum_pad_height = -math.inf
    stable_success_since: float | None = None
    grasp_base: tuple[float, float, float] | None = None
    target_base: tuple[float, float, float] | None = None
    trajectory_samples: list[dict[str, object]] = []
    sampled_frames: set[int] = set()
    alignment_started_sim: float | None = None
    alignment_completed_sim: float | None = None
    alignment_stable_since: float | None = None
    placement_base: tuple[float, float, float] | None = None
    last_active_sim: float | None = None
    release_started_sim: float | None = None
    grasp_dwell_started_sim: float | None = None
    grasp_dwell_completed_sim: float | None = None
    cartesian_handoff_started_wall: float | None = None
    cartesian_target: tuple[float, ...] | None = None
    cartesian_handoff_target: tuple[float, ...] | None = None
    cartesian_phase: str | None = None
    precontact_target_quaternion: tuple[float, ...] | None = None
    place_target_quaternion: tuple[float, ...] | None = None
    precontact_goal_z_m: float | None = None
    precontact_goal_xy_m: tuple[float, float] | None = None
    contact_started_sim: float | None = None
    contact_ee_z_m: float | None = None
    contact_ee_target_z_m: float | None = None
    minimum_observed_contact_ee_z_m = math.inf
    contact_rotation_completed_sim: float | None = None
    contact_pad_centroid_m: tuple[float, float, float] | None = None
    cartesian_stable_since: float | None = None
    release_pad_height_m: float | None = None
    release_xy_error_m: float | None = None
    release_ee_z_m: float | None = None
    release_orientation_error_deg: float | None = None
    retract_goal_z_m: float | None = None
    retract_started_sim: float | None = None
    retract_completed_sim: float | None = None
    retract_stable_since: float | None = None
    transport_checkpoint_passed = False
    transport_alignment_started_sim: float | None = None
    transport_alignment_completed_sim: float | None = None
    transport_alignment_stable_since: float | None = None
    reference_start = JOINT_LANDMARKS[0][1]
    reason = "host_watchdog_timeout"
    success = False
    try:
        while rclpy.ok() and time.monotonic() - started_wall < args.max_duration_s:
            rclpy.spin_once(node, timeout_sec=0.02)
            if (
                node.sim_time is None
                or node.left_joints is None
                or node.joints is None
                or node.pad_centroid is None
                or node.right_ee is None
                or node.base is None
                or "thermalpad" not in node.objects
                or "board_target" not in node.objects
            ):
                continue
            if (
                node.joint_pub.get_subscription_count() != 1
                or node.gripper_pub.get_subscription_count() != 1
                or node.base_pub.get_subscription_count() != 1
                or (
                    args.pregrasp_only
                    and (
                        node.left_joint_pub.get_subscription_count() != 1
                        or node.left_gripper_pub.get_subscription_count() != 1
                    )
                )
            ):
                continue
            if started_sim is None:
                started_sim = node.sim_time
                live_start = node.joints
                live_left_start = node.left_joints
                baseline_height = node.pad_centroid[2]
                initial_pad_centroid = node.pad_centroid
                grasp_base = deepen_grasp_base_pose(
                    anchored_base_pose(
                        node.objects["thermalpad"],
                        REFERENCE_PAD_XYYAW,
                        max_yaw_delta_rad=math.radians(
                            args.grasp_yaw_limit_deg
                        ),
                    ),
                    node.objects["thermalpad"],
                    args.grasp_depth_bias_m,
                )
                # The launcher physically drives and settles the base first.
                # Latch measured odometry without a jump, then bound only the
                # remaining centimetre-scale GT alignment before q399.  This
                # preserves the visible base -> spine -> grasp-pose order and
                # avoids dragging the open gripper across the pad edge.
                preposition_base = (
                    grasp_base
                    if args.preposition_at_live_grasp_base
                    else node.base
                )
                approach_base = preposition_base
                target_base = anchored_base_pose(
                    node.objects["board_target"], REFERENCE_TARGET_XYYAW
                )
            assert live_start is not None and baseline_height is not None
            assert live_left_start is not None
            assert grasp_base is not None and target_base is not None
            assert preposition_base is not None and approach_base is not None
            if args.absolute_dataset_joints and trajectory_started_sim is None:
                if base_preposition_completed_sim is None:
                    node.publish(live_start, 1.0)
                    if args.pregrasp_only:
                        node.publish_left_pregrasp(live_left_start)
                    delta_sim = max(
                        0.0,
                        node.sim_time - (last_preposition_sim or node.sim_time),
                    )
                    approach_base = bounded_base_pose_step(
                        approach_base,
                        grasp_base,
                        linear_speed_mps=args.preposition_base_speed_mps,
                        angular_speed_rps=(
                            args.preposition_base_angular_speed_rps
                        ),
                        elapsed_s=delta_sim,
                    )
                    node.publish_base(approach_base)
                    distance = math.hypot(
                        grasp_base[0] - approach_base[0],
                        grasp_base[1] - approach_base[1],
                    )
                    yaw_error = abs(
                        _wrap_angle(grasp_base[2] - approach_base[2])
                    )
                    if (
                        distance
                        <= args.preposition_base_position_tolerance_m
                        and yaw_error
                        <= args.preposition_base_yaw_tolerance_rad
                    ):
                        if base_approach_stable_since is None:
                            base_approach_stable_since = node.sim_time
                        elif node.sim_time - base_approach_stable_since >= 0.5:
                            base_preposition_completed_sim = node.sim_time
                            base_approach_stable_since = None
                    else:
                        base_approach_stable_since = None
                    last_preposition_sim = node.sim_time
                    continue

                node.publish(reference_start, 1.0)
                if args.pregrasp_only:
                    node.publish_left_pregrasp(LEFT_PREGRASP_Q399)
                node.publish_base(grasp_base)
                final_right_preposition_error = max(
                    abs(actual - target)
                    for actual, target in zip(node.joints, reference_start)
                )
                final_left_preposition_error = max(
                    abs(actual - target)
                    for actual, target in zip(
                        node.left_joints, LEFT_PREGRASP_Q399, strict=True
                    )
                )
                final_preposition_error = max(
                    final_right_preposition_error,
                    (
                        final_left_preposition_error
                        if args.pregrasp_only
                        else 0.0
                    ),
                )
                if (
                    joint_preposition_completed_sim is None
                    and final_preposition_error <= args.preposition_tolerance_rad
                ):
                    if preposition_stable_since is None:
                        preposition_stable_since = node.sim_time
                    elif node.sim_time - preposition_stable_since >= 0.5:
                        joint_preposition_completed_sim = node.sim_time
                else:
                    if joint_preposition_completed_sim is None:
                        preposition_stable_since = None
                if (
                    joint_preposition_completed_sim is not None
                    and final_preposition_error
                    <= args.grasp_joint_tolerance_rad
                ):
                    if grasp_ready_stable_since is None:
                        grasp_ready_stable_since = node.sim_time
                    elif node.sim_time - grasp_ready_stable_since >= 0.5:
                        if args.pregrasp_only:
                            success = True
                            reason = "stable_dataset_ground_truth_pregrasp"
                            break
                        trajectory_started_sim = node.sim_time
                        baseline_height = node.pad_centroid[2]
                else:
                    grasp_ready_stable_since = None
                continue
            if trajectory_started_sim is None:
                trajectory_started_sim = started_sim
            elapsed = node.sim_time - trajectory_started_sim
            paused = 0.0
            if grasp_dwell_started_sim is not None:
                paused += (
                    (grasp_dwell_completed_sim or node.sim_time)
                    - grasp_dwell_started_sim
                )
            if alignment_started_sim is not None:
                paused += (
                    (alignment_completed_sim or node.sim_time)
                    - alignment_started_sim
                )
            if transport_alignment_started_sim is not None:
                paused += (
                    (transport_alignment_completed_sim or node.sim_time)
                    - transport_alignment_started_sim
                )
            trajectory_elapsed = max(0.0, elapsed - paused)
            # Dataset frames advance at 30 Hz.  Hold frame 399 briefly so the
            # bridge sees the fresh direct-command owner before closure.
            frame = (
                399.0
                if trajectory_elapsed < 0.15
                else 399.0
                + (trajectory_elapsed - 0.15) * args.trajectory_rate_hz
            )
            if frame >= 410.0 and grasp_dwell_completed_sim is None:
                frame = 410.0
                if grasp_dwell_started_sim is None:
                    grasp_dwell_started_sim = node.sim_time
                elif node.sim_time - grasp_dwell_started_sim >= args.grasp_dwell_s:
                    grasp_dwell_completed_sim = node.sim_time
            if (
                args.transport_base_alignment_frame > 0.0
                and frame >= args.transport_base_alignment_frame
                and transport_alignment_completed_sim is None
            ):
                frame = args.transport_base_alignment_frame
                if transport_alignment_started_sim is None:
                    transport_alignment_started_sim = node.sim_time
                    placement_base = grasp_base
                assert placement_base is not None
                target_pose = node.objects["board_target"]
                error_x = target_pose[0] - node.pad_centroid[0]
                error_y = target_pose[1] - node.pad_centroid[1]
                xy_error = math.hypot(error_x, error_y)
                if (
                    xy_error
                    > args.transport_base_alignment_max_initial_xy_m
                    or node.pad_centroid[2] - target_pose[2] < 0.10
                ):
                    reason = "transport_base_alignment_gate_failed"
                    break
                delta_sim = max(
                    0.0,
                    node.sim_time - (last_active_sim or node.sim_time),
                )
                maximum_step = args.transport_base_speed_mps * delta_sim
                scale = min(1.0, maximum_step / max(xy_error, 1.0e-9))
                placement_base = (
                    placement_base[0] + scale * error_x,
                    placement_base[1] + scale * error_y,
                    placement_base[2],
                )
                if math.hypot(
                    placement_base[0] - grasp_base[0],
                    placement_base[1] - grasp_base[1],
                ) > args.transport_base_alignment_max_displacement_m:
                    reason = "transport_base_alignment_exceeded_limit"
                    break
                if xy_error <= 0.012:
                    if transport_alignment_stable_since is None:
                        transport_alignment_stable_since = node.sim_time
                    elif (
                        node.sim_time - transport_alignment_stable_since >= 0.3
                    ):
                        transport_alignment_completed_sim = node.sim_time
                else:
                    transport_alignment_stable_since = None
            if (
                args.feedback_release_frame > 0.0
                and frame >= args.feedback_release_frame
                and alignment_completed_sim is None
            ):
                frame = args.feedback_release_frame
                if alignment_started_sim is None:
                    alignment_started_sim = node.sim_time
                    placement_base = grasp_base
                assert placement_base is not None
                target_pose = node.objects["board_target"]
                error_x = target_pose[0] - node.pad_centroid[0]
                error_y = target_pose[1] - node.pad_centroid[1]
                xy_error = math.hypot(error_x, error_y)
                if (
                    args.feedback_release_frame >= 650.0
                    and xy_error > 0.15
                ):
                    reason = "placement_contact_checkpoint_failed"
                    break
                delta_sim = max(
                    0.0,
                    node.sim_time - (last_active_sim or node.sim_time),
                )
                maximum_step = 0.03 * delta_sim
                scale = min(1.0, maximum_step / max(xy_error, 1.0e-9))
                placement_base = (
                    placement_base[0] + scale * error_x,
                    placement_base[1] + scale * error_y,
                    placement_base[2],
                )
                if xy_error <= 0.015:
                    if alignment_stable_since is None:
                        alignment_stable_since = node.sim_time
                    elif node.sim_time - alignment_stable_since >= 0.5:
                        alignment_completed_sim = node.sim_time
                else:
                    alignment_stable_since = None
                if math.hypot(
                    placement_base[0] - grasp_base[0],
                    placement_base[1] - grasp_base[1],
                ) > 0.25:
                    reason = "placement_base_alignment_exceeded_limit"
                    break
                if (
                    args.feedback_release_frame < 650.0
                    and
                    node.pad_centroid[2] - node.objects["board_target"][2]
                    < 0.08
                ):
                    reason = "pad_lost_during_alignment"
                    break
            elif alignment_completed_sim is not None:
                frame = args.feedback_release_frame
                if release_started_sim is None:
                    release_started_sim = node.sim_time
            if (
                args.cartesian_handoff_frame > 520.0
                and frame >= 520.0
                and not transport_checkpoint_passed
            ):
                assert initial_pad_centroid is not None
                checkpoint_height = (
                    node.pad_centroid[2] - node.objects["board_target"][2]
                )
                checkpoint_xy_displacement = math.hypot(
                    node.pad_centroid[0] - initial_pad_centroid[0],
                    node.pad_centroid[1] - initial_pad_centroid[1],
                )
                if (
                    maximum_pad_height < args.lift_height_m
                    or checkpoint_height < 0.12
                    or checkpoint_xy_displacement < 0.15
                ):
                    reason = "transport_checkpoint_failed_at_frame_520"
                    break
                transport_checkpoint_passed = True
            cartesian_active = False
            if (
                args.feedback_release_frame <= 0.0
                and frame >= args.cartesian_handoff_frame
            ):
                frame = args.cartesian_handoff_frame
                if cartesian_handoff_started_wall is None:
                    current_height = (
                        node.pad_centroid[2] - node.objects["board_target"][2]
                    )
                    current_xy_error = math.hypot(
                        node.pad_centroid[0] - node.objects["board_target"][0],
                        node.pad_centroid[1] - node.objects["board_target"][1],
                    )
                    # Before frame 630 the pad should still be securely
                    # suspended.  At/after 630 the demonstrated wrist is in
                    # the placement descent, so keep Cartesian control alive
                    # after table contact and use it as a gentle planar servo.
                    table_servo = args.cartesian_handoff_frame >= 630.0
                    minimum_handoff_height = -0.01 if table_servo else 0.08
                    if (
                        maximum_pad_height < args.lift_height_m
                        or current_height < minimum_handoff_height
                        or (
                            table_servo
                            and current_height < 0.02
                            and current_xy_error
                            > args.cartesian_descent_start_xy_m
                        )
                        or (
                            table_servo and current_xy_error > 0.15
                        )
                    ):
                        reason = "lift_gate_failed_before_cartesian_handoff"
                        break
                    cartesian_handoff_started_wall = time.monotonic()
                    cartesian_target = node.right_ee
                    cartesian_handoff_target = node.right_ee
                    target_yaw_delta = _wrap_angle(
                        _yaw_from_wxyz(
                            node.objects["board_target"][3:7]
                        )
                        - REFERENCE_TARGET_XYYAW[2]
                    )
                    precontact_target_quaternion = (
                        yaw_rotated_reference_quaternion(
                            REFERENCE_PRECONTACT_QUATERNION_XYZW,
                            target_yaw_delta,
                        )
                    )
                    place_target_quaternion = (
                        yaw_rotated_reference_quaternion(
                            REFERENCE_PLACE_QUATERNION_XYZW,
                            target_yaw_delta,
                        )
                    )
                    contact_sweep = yaw_rotated_planar_offset(
                        (args.contact_sweep_x_m, args.contact_sweep_y_m),
                        target_yaw_delta,
                    )
                    # Compensate before contact because the deformable pad
                    # can slip out as soon as the inward wrist sweep lays it
                    # on the table; post-contact EE motion then cannot move it.
                    precontact_goal_xy_m = (
                        node.objects["board_target"][0] - contact_sweep[0],
                        node.objects["board_target"][1] - contact_sweep[1],
                    )
                    precontact_goal_z_m = (
                        cartesian_target[2]
                        + args.precontact_lift_m
                        - current_height
                    )
                    cartesian_phase = "align_xy"
                assert cartesian_target is not None
                assert cartesian_handoff_target is not None
                assert cartesian_phase is not None
                assert precontact_target_quaternion is not None
                assert place_target_quaternion is not None
                assert precontact_goal_xy_m is not None
                node.publish_ee(cartesian_target)
                cartesian_active = (
                    time.monotonic() - cartesian_handoff_started_wall
                    >= args.cartesian_handoff_settle_s
                )
                if cartesian_active:
                    target_pose = node.objects["board_target"]
                    error_x = target_pose[0] - node.pad_centroid[0]
                    error_y = target_pose[1] - node.pad_centroid[1]
                    xy_error = math.hypot(error_x, error_y)
                    alignment_error_x = (
                        precontact_goal_xy_m[0] - node.pad_centroid[0]
                    )
                    alignment_error_y = (
                        precontact_goal_xy_m[1] - node.pad_centroid[1]
                    )
                    alignment_xy_error = math.hypot(
                        alignment_error_x, alignment_error_y
                    )
                    pad_height = node.pad_centroid[2] - target_pose[2]
                    if release_started_sim is None:
                        delta_sim = max(
                            0.0,
                            node.sim_time - (last_active_sim or node.sim_time),
                        )
                        release_ready = False
                        if cartesian_phase == "align_xy":
                            if pad_height < 0.08:
                                reason = "pad_lost_during_xy_alignment"
                                break
                            maximum_step = args.cartesian_speed_mps * delta_sim
                            scale = min(
                                1.0,
                                maximum_step
                                / max(alignment_xy_error, 1.0e-9),
                            )
                            offset_x, offset_y = bounded_planar_offset(
                                cartesian_target[0]
                                + scale * alignment_error_x
                                - cartesian_handoff_target[0],
                                cartesian_target[1]
                                + scale * alignment_error_y
                                - cartesian_handoff_target[1],
                                args.cartesian_max_xy_displacement_m,
                            )
                            cartesian_target = (
                                cartesian_handoff_target[0] + offset_x,
                                cartesian_handoff_target[1] + offset_y,
                                *cartesian_target[2:],
                            )
                            if (
                                alignment_xy_error
                                <= args.cartesian_descent_start_xy_m
                            ):
                                cartesian_phase = "precontact"
                        elif cartesian_phase == "precontact":
                            maximum_step = args.cartesian_speed_mps * delta_sim
                            scale = min(
                                1.0,
                                maximum_step
                                / max(alignment_xy_error, 1.0e-9),
                            )
                            offset_x, offset_y = bounded_planar_offset(
                                cartesian_target[0]
                                + scale * alignment_error_x
                                - cartesian_handoff_target[0],
                                cartesian_target[1]
                                + scale * alignment_error_y
                                - cartesian_handoff_target[1],
                                args.cartesian_max_xy_displacement_m,
                            )
                            z_step = bounded_axis_step(
                                args.precontact_lift_m - pad_height,
                                args.precontact_lift_speed_mps,
                                delta_sim,
                            )
                            commanded_contact_floor = (
                                args.minimum_contact_ee_z_m
                                + args.contact_ee_tracking_margin_m
                                if args.minimum_contact_ee_z_m > 0.0
                                else -math.inf
                            )
                            cartesian_target = (
                                cartesian_handoff_target[0] + offset_x,
                                cartesian_handoff_target[1] + offset_y,
                                max(
                                    commanded_contact_floor,
                                    cartesian_target[2] + z_step,
                                ),
                                *cartesian_handoff_target[3:7],
                            )
                            precontact_ready = (
                                pad_height <= args.precontact_lift_m
                                + args.contact_height_tolerance_m
                            )
                            if precontact_ready:
                                if (
                                    node.right_ee[2]
                                    + args.contact_ee_clearance_tolerance_m
                                    < args.minimum_contact_ee_z_m
                                ):
                                    reason = "unsafe_gripper_table_clearance"
                                    break
                                contact_started_sim = node.sim_time
                                contact_ee_z_m = node.right_ee[2]
                                minimum_observed_contact_ee_z_m = min(
                                    minimum_observed_contact_ee_z_m,
                                    node.right_ee[2],
                                )
                                contact_ee_target_z_m = cartesian_target[2]
                                contact_pad_centroid_m = node.pad_centroid
                                cartesian_phase = "contact_rotate_precontact"
                        elif cartesian_phase == "contact_rotate_precontact":
                            minimum_observed_contact_ee_z_m = min(
                                minimum_observed_contact_ee_z_m,
                                node.right_ee[2],
                            )
                            if (
                                node.right_ee[2]
                                + args.contact_ee_clearance_tolerance_m
                                < args.minimum_contact_ee_z_m
                            ):
                                reason = "unsafe_gripper_table_clearance"
                                break
                            orientation = bounded_orientation_step(
                                cartesian_target[3:7],
                                precontact_target_quaternion,
                                args.precontact_angular_speed_deg_s,
                                delta_sim,
                            )
                            cartesian_target = (
                                cartesian_target[0],
                                cartesian_target[1],
                                cartesian_target[2],
                                *orientation,
                            )
                            if (
                                quaternion_error_deg(
                                    node.right_ee[3:7],
                                    precontact_target_quaternion,
                                )
                                <= args.precontact_orientation_tolerance_deg
                            ):
                                cartesian_phase = "contact_rotate"
                        elif cartesian_phase == "contact_rotate":
                            assert contact_ee_z_m is not None
                            minimum_observed_contact_ee_z_m = min(
                                minimum_observed_contact_ee_z_m,
                                node.right_ee[2],
                            )
                            if (
                                node.right_ee[2]
                                + args.contact_ee_clearance_tolerance_m
                                < args.minimum_contact_ee_z_m
                            ):
                                reason = "unsafe_gripper_table_clearance"
                                break
                            orientation = bounded_orientation_step(
                                cartesian_target[3:7],
                                place_target_quaternion,
                                args.contact_angular_speed_deg_s,
                                delta_sim,
                            )
                            z_step = bounded_axis_step(
                                contact_ee_z_m
                                - args.contact_wrist_z_drop_m
                                - cartesian_target[2],
                                args.contact_height_correction_speed_mps,
                                delta_sim,
                            )
                            cartesian_target = (
                                cartesian_target[0],
                                cartesian_target[1],
                                cartesian_target[2] + z_step,
                                *orientation,
                            )
                            orientation_error = quaternion_error_deg(
                                node.right_ee[3:7], place_target_quaternion
                            )
                            if (
                                orientation_error
                                <= args.place_orientation_tolerance_deg
                            ):
                                if contact_rotation_completed_sim is None:
                                    contact_rotation_completed_sim = node.sim_time
                            release_ready = placement_release_ready(
                                xy_error_m=xy_error,
                                pad_height_m=pad_height,
                                orientation_error_deg=orientation_error,
                                release_xy_m=args.alignment_release_xy_m,
                                release_height_m=args.alignment_release_height_m,
                                release_height_tolerance_m=(
                                    args.alignment_release_height_tolerance_m
                                ),
                                orientation_tolerance_deg=(
                                    args.place_orientation_tolerance_deg
                                ),
                                require_target_xy=(
                                    args.placement_contract == "nominal"
                                ),
                            )
                            release_orientation_error_deg = orientation_error
                        else:
                            raise AssertionError(
                                f"unsupported Cartesian phase: {cartesian_phase}"
                            )
                        node.publish_ee(cartesian_target)
                        if (
                            cartesian_phase == "contact_rotate"
                            and release_ready
                        ):
                            if cartesian_stable_since is None:
                                cartesian_stable_since = node.sim_time
                            elif (
                                node.sim_time - cartesian_stable_since
                                >= args.alignment_stable_s
                            ):
                                release_started_sim = node.sim_time
                                release_pad_height_m = pad_height
                                release_xy_error_m = xy_error
                                release_ee_z_m = node.right_ee[2]
                                retract_goal_z_m = (
                                    release_ee_z_m
                                    + args.post_release_retract_m
                                )
                        else:
                            cartesian_stable_since = None
                            if (
                                contact_rotation_completed_sim is not None
                                and node.sim_time
                                - contact_rotation_completed_sim
                                >= 2.0
                            ):
                                reason = "placement_offset_after_contact_rotation"
                                break
                    elif (
                        node.sim_time - release_started_sim
                        >= args.post_release_settle_s
                    ):
                        assert retract_goal_z_m is not None
                        if retract_started_sim is None:
                            retract_started_sim = node.sim_time
                        delta_sim = max(
                            0.0,
                            node.sim_time - (last_active_sim or node.sim_time),
                        )
                        z_step = bounded_axis_step(
                            retract_goal_z_m - cartesian_target[2],
                            args.post_release_retract_speed_mps,
                            delta_sim,
                        )
                        cartesian_target = (
                            cartesian_target[0],
                            cartesian_target[1],
                            cartesian_target[2] + z_step,
                            *cartesian_target[3:],
                        )
                        node.publish_ee(cartesian_target)
                        retracted = (
                            node.right_ee[2]
                            >= retract_goal_z_m
                            - args.post_release_retract_tolerance_m
                        )
                        if retracted:
                            if retract_stable_since is None:
                                retract_stable_since = node.sim_time
                            elif node.sim_time - retract_stable_since >= 0.20:
                                retract_completed_sim = node.sim_time
                        else:
                            retract_stable_since = None
                    if (
                        args.cartesian_handoff_frame < 630.0
                        and release_started_sim is None
                        and
                        node.pad_centroid[2]
                        - node.objects["board_target"][2]
                        < -0.02
                    ):
                        reason = "pad_below_target_before_release"
                        break
            last_active_sim = node.sim_time
            reference_q, grip = interpolate_landmark(frame)
            evaluate_release = release_started_sim is not None or (
                args.feedback_release_frame <= 0.0 and frame >= args.stop_frame
            )
            if evaluate_release:
                grip = 1.0
            # Keep the demonstrated grasp-relative transform until alignment,
            # then retain the low-speed GT feedback alignment used above.
            selected_base = placement_base or grasp_base
            node.publish_base(selected_base)
            target = (
                reference_q
                if args.absolute_dataset_joints
                else tuple(
                    live_start[i] + reference_q[i] - reference_start[i]
                    for i in range(len(RIGHT_JOINTS))
                )
            )
            if cartesian_active:
                node.publish_gripper(grip)
                node.publish_count += 1
            else:
                node.publish(target, grip)
            maximum_pad_height = max(
                maximum_pad_height,
                node.pad_centroid[2] - node.objects["board_target"][2],
            )
            for landmark_frame, _, _ in JOINT_LANDMARKS:
                if frame >= landmark_frame and landmark_frame not in sampled_frames:
                    trajectory_samples.append(
                        {
                            "frame": landmark_frame,
                            "pad_centroid_m": node.pad_centroid,
                            "base_xyyaw": selected_base,
                        }
                    )
                    sampled_frames.add(landmark_frame)
            if release_started_sim is not None:
                target_pose = node.objects["board_target"]
                xy_error = math.hypot(
                    node.pad_centroid[0] - target_pose[0],
                    node.pad_centroid[1] - target_pose[1],
                )
                z_error = abs(node.pad_centroid[2] - target_pose[2])
                gates_pass = (
                    maximum_pad_height >= args.lift_height_m
                    and (
                        args.placement_contract == "randomized-flat"
                        or xy_error <= args.success_xy_m
                    )
                    and z_error <= args.success_z_m
                    and node.pad_z_span_m is not None
                    and node.pad_z_span_m <= args.flat_pad_z_span_m
                    and grip >= 0.95
                    and retract_completed_sim is not None
                )
                if gates_pass:
                    if stable_success_since is None:
                        stable_success_since = node.sim_time
                    elif node.sim_time - stable_success_since >= 0.5:
                        success = True
                        reason = (
                            "stable_target_place_release_and_retract"
                            if args.placement_contract == "nominal"
                            else "stable_flat_place_release_and_retract"
                        )
                        break
                else:
                    stable_success_since = None
                    if (
                        retract_completed_sim is not None
                        and node.sim_time - retract_completed_sim >= 1.0
                    ):
                        reason = "post_release_placement_unstable"
                        break
    finally:
        if not success and reason == "host_watchdog_timeout" and node.pad_centroid:
            reason = "place_or_release_gate_timeout"
        result = {
            "success": success,
            "reason": reason,
            "schema_version": 1,
            "placement_contract": args.placement_contract,
            "source_episode": 19,
            "source_frames": [item[0] for item in JOINT_LANDMARKS],
            "relative_to_live_joint_state": not args.absolute_dataset_joints,
            "publish_count": node.publish_count,
            "baseline_pad_centroid_z_m": baseline_height,
            "final_preposition_max_joint_error_rad": final_preposition_error,
            "final_left_preposition_max_joint_error_rad": (
                final_left_preposition_error if args.pregrasp_only else None
            ),
            "final_right_preposition_max_joint_error_rad": (
                final_right_preposition_error
            ),
            "final_pad_centroid_m": node.pad_centroid,
            "final_pad_z_span_m": node.pad_z_span_m,
            "flat_pad_z_span_threshold_m": args.flat_pad_z_span_m,
            "final_right_ee_world_xyzw": node.right_ee,
            "final_ee_world": {"right": node.right_ee},
            "final_base_xyyaw": node.base,
            "final_joints": {
                **(
                    dict(zip(LEFT_JOINTS, node.left_joints, strict=True))
                    if node.left_joints is not None
                    else {}
                ),
                **(
                    dict(zip(RIGHT_JOINTS, node.joints, strict=True))
                    if node.joints is not None
                    else {}
                ),
                SPINE_JOINT: 0.4857,
            },
            "handoff_sim_time": node.sim_time,
            "provenance": {
                "controller": "formal_phase1_ground_truth_joint_lift",
                "dataset_episode": 19,
                "dataset_frame": 399,
                "absolute_dataset_joints": args.absolute_dataset_joints,
                "guessed_ik_used": False,
                "staged_groups": (
                    [
                        "base",
                        "spine",
                        "left_arm",
                        "right_arm",
                        "left_gripper",
                        "right_gripper",
                    ]
                    if args.pregrasp_only
                    else ["base", "spine", "right_arm", "right_gripper"]
                ),
            },
            "final_board_target_pose_wxyz": node.objects.get("board_target"),
            "grasp_aligned_base_xyyaw": grasp_base,
            "safe_preposition_base_xyyaw": preposition_base,
            "base_preposition_mode": (
                "legacy_direct_live_grasp"
                if args.preposition_at_live_grasp_base
                else "measured_odom_bounded_alignment"
            ),
            "base_preposition_completed_sim": base_preposition_completed_sim,
            "joint_preposition_completed_sim": joint_preposition_completed_sim,
            "grasp_dwell_duration_sim_s": (
                grasp_dwell_completed_sim - grasp_dwell_started_sim
                if grasp_dwell_completed_sim is not None
                and grasp_dwell_started_sim is not None
                else None
            ),
            "target_aligned_base_xyyaw": target_base,
            "placement_alignment_base_xyyaw": placement_base,
            "placement_alignment_duration_sim_s": (
                alignment_completed_sim - alignment_started_sim
                if alignment_completed_sim is not None
                and alignment_started_sim is not None
                else None
            ),
            "release_started_sim": release_started_sim,
            "release_pad_height_above_target_m": release_pad_height_m,
            "release_pad_target_xy_error_m": release_xy_error_m,
            "release_ee_z_m": release_ee_z_m,
            "release_orientation_error_deg": release_orientation_error_deg,
            "cartesian_phase": cartesian_phase,
            "precontact_goal_z_m": precontact_goal_z_m,
            "precontact_goal_xy_m": precontact_goal_xy_m,
            "contact_started_sim": contact_started_sim,
            "contact_ee_z_m": contact_ee_z_m,
            "contact_ee_target_z_m": contact_ee_target_z_m,
            "minimum_contact_ee_z_m": args.minimum_contact_ee_z_m,
            "contact_ee_tracking_margin_m": (
                args.contact_ee_tracking_margin_m
            ),
            "contact_ee_clearance_tolerance_m": (
                args.contact_ee_clearance_tolerance_m
            ),
            "minimum_observed_contact_ee_z_m": (
                minimum_observed_contact_ee_z_m
                if math.isfinite(minimum_observed_contact_ee_z_m)
                else None
            ),
            "contact_rotation_completed_sim": contact_rotation_completed_sim,
            "contact_pad_centroid_m": contact_pad_centroid_m,
            "retract_goal_z_m": retract_goal_z_m,
            "retract_started_sim": retract_started_sim,
            "retract_completed_sim": retract_completed_sim,
            "cartesian_handoff_started": cartesian_handoff_started_wall is not None,
            "transport_checkpoint_passed": transport_checkpoint_passed,
            "transport_alignment_started_sim": transport_alignment_started_sim,
            "transport_alignment_completed_sim": transport_alignment_completed_sim,
            "cartesian_target_world_xyzw": cartesian_target,
            "trajectory_samples": trajectory_samples,
            "maximum_pad_height_above_target_m": (
                maximum_pad_height if math.isfinite(maximum_pad_height) else None
            ),
            "final_pad_target_xy_error_m": (
                math.hypot(
                    node.pad_centroid[0] - node.objects["board_target"][0],
                    node.pad_centroid[1] - node.objects["board_target"][1],
                )
                if node.pad_centroid and "board_target" in node.objects
                else None
            ),
            "final_pad_target_z_error_m": (
                abs(node.pad_centroid[2] - node.objects["board_target"][2])
                if node.pad_centroid and "board_target" in node.objects
                else None
            ),
            "pad_height_above_target_m": (
                node.pad_centroid[2] - node.objects["board_target"][2]
                if node.pad_centroid and "board_target" in node.objects
                else None
            ),
            "elapsed_wall_s": time.monotonic() - started_wall,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
