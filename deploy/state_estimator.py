#!/usr/bin/env python3
"""Estimate Go2 world-frame velocity from a hardware deployment log.

The estimator combines IMU propagation with force-weighted stance-foot
kinematics.  Its world frame starts at the robot's initial yaw, so +x is the
robot's initial forward direction.  The input log is never modified.
"""

from __future__ import annotations

import argparse
import csv
import io
import pickle
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = ("hip", "thigh", "calf")
GRAVITY = 9.81


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclass
class Estimate:
    time: np.ndarray
    world_velocity: np.ndarray
    body_velocity: np.ndarray
    world_position: np.ndarray
    leg_world_velocity: np.ndarray
    yaw: np.ndarray
    yaw_rate: np.ndarray
    yaw_rate_from_rpy: np.ndarray
    contact_force: np.ndarray
    contact_weight: np.ndarray
    contact_count: np.ndarray
    contact_confidence: np.ndarray
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    command_velocity: np.ndarray | None
    command_yaw_rate: np.ndarray | None
    old_kf_forward_velocity: np.ndarray | None
    calibration_samples: int


def _parse_vector(text: str | None, length: int) -> np.ndarray:
    if not text:
        return np.zeros(length, dtype=float)
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.size != length:
        raise ValueError(f"expected {length} values, got {values.size}: {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class Go2Kinematics:
    """Minimal URDF kinematics for the four base-to-foot chains."""

    def __init__(self, urdf_path: Path):
        root = ET.parse(urdf_path).getroot()
        joints: dict[str, Joint] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            origin = element.find("origin")
            axis = element.find("axis")
            joints[name] = Joint(
                name=name,
                kind=element.attrib["type"],
                origin_xyz=_parse_vector(
                    None if origin is None else origin.attrib.get("xyz"), 3
                ),
                origin_rpy=_parse_vector(
                    None if origin is None else origin.attrib.get("rpy"), 3
                ),
                axis=_parse_vector(
                    "1 0 0" if axis is None else axis.attrib.get("xyz"), 3
                ),
            )

        self.chains: dict[str, tuple[Joint, ...]] = {}
        for leg in LEG_NAMES:
            names = (
                f"{leg}_hip_joint",
                f"{leg}_thigh_joint",
                f"{leg}_calf_joint",
                f"{leg}_foot_joint",
            )
            missing = [name for name in names if name not in joints]
            if missing:
                raise ValueError(f"URDF is missing joints: {', '.join(missing)}")
            self.chains[leg] = tuple(joints[name] for name in names)

    def foot_position_and_jacobian(
        self, leg: str, joint_position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.eye(3)
        position = np.zeros(3)
        axes: list[np.ndarray] = []
        pivots: list[np.ndarray] = []
        moving_joint_index = 0

        for joint in self.chains[leg]:
            position = position + rotation @ joint.origin_xyz
            rotation = rotation @ _rpy_matrix(joint.origin_rpy)
            if joint.kind in ("revolute", "continuous"):
                axis = rotation @ joint.axis
                axes.append(axis)
                pivots.append(position.copy())
                rotation = rotation @ _axis_angle_matrix(
                    joint.axis, float(joint_position[moving_joint_index])
                )
                moving_joint_index += 1

        if moving_joint_index != 3:
            raise ValueError(f"{leg} chain has {moving_joint_index} moving joints, expected 3")
        jacobian = np.column_stack(
            [np.cross(axis, position - pivot) for axis, pivot in zip(axes, pivots)]
        )
        return position, jacobian


def _load_pickle_on_cpu(path: Path) -> object:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to read this deployment pickle") from exc

    original_loader = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda data: torch.load(  # type: ignore[assignment]
        io.BytesIO(data), map_location="cpu", weights_only=False
    )
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    finally:
        torch.storage._load_from_bytes = original_loader


def _as_vector(value: object, name: str, length: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    result = np.asarray(value, dtype=float).reshape(-1)
    if length is not None and result.size != length:
        raise ValueError(f"field {name!r} has {result.size} values; expected {length}")
    return result


def _stack_required(samples: list[dict], name: str, length: int) -> np.ndarray:
    missing = [index for index, sample in enumerate(samples) if name not in sample]
    if missing:
        raise ValueError(f"field {name!r} is missing at sample {missing[0]}")
    return np.vstack([_as_vector(sample[name], name, length) for sample in samples])


def _stack_optional(samples: list[dict], name: str, length: int) -> np.ndarray | None:
    if not all(name in sample and sample[name] is not None for sample in samples):
        return None
    try:
        return np.vstack([_as_vector(sample[name], name, length) for sample in samples])
    except ValueError:
        return None


def _extract_log(path: Path, agent_key: str) -> tuple[dict[str, np.ndarray | None], list[dict]]:
    loaded = _load_pickle_on_cpu(path)
    if not isinstance(loaded, dict):
        raise ValueError("pickle root must be a dictionary")
    if agent_key not in loaded:
        available = ", ".join(map(str, loaded.keys()))
        raise ValueError(f"agent key {agent_key!r} not found; available keys: {available}")
    payload = loaded[agent_key]
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(f"{agent_key!r} must contain [config, samples]")
    samples = payload[1]
    if not isinstance(samples, list) or not samples:
        raise ValueError("log contains no samples")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("every log sample must be a dictionary")

    count = len(samples)
    if all("time" in sample for sample in samples):
        time = np.array([_as_vector(sample["time"], "time", 1)[0] for sample in samples])
    else:
        time = np.arange(count, dtype=float) * 0.02
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("sample times must be finite and strictly increasing")

    data: dict[str, np.ndarray | None] = {
        "time": time,
        "joint_pos": _stack_required(samples, "joint_pos", 12),
        "joint_vel": _stack_required(samples, "joint_vel", 12),
        "rpy": _stack_required(samples, "rpy", 3),
        "accel": _stack_required(samples, "aBody", 3),
        "gyro": _stack_required(samples, "omegaBody", 3),
        "force": _stack_required(samples, "contact_estimate", 4),
        "command_velocity": _stack_optional(samples, "body_linear_vel_cmd", 2),
        "command_yaw_rate": _stack_optional(samples, "body_angular_vel_cmd", 1),
    }
    if all("fwd_linear_vel_by_kf_se" in sample for sample in samples):
        data["old_kf_forward_velocity"] = np.array(
            [float(_as_vector(sample["fwd_linear_vel_by_kf_se"], "old_kf", 1)[0]) for sample in samples]
        )
    else:
        data["old_kf_forward_velocity"] = None
    return data, samples


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("contact median window must be odd")
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    return np.median(windows, axis=-1)


def _leading_stationary_samples(
    time: np.ndarray,
    joint_velocity: np.ndarray,
    gyro: np.ndarray,
    requested_seconds: float,
    maximum_auto_seconds: float,
) -> int:
    if requested_seconds > 0.0:
        return max(1, int(np.searchsorted(time - time[0], requested_seconds, side="right")))

    maximum = max(1, int(np.searchsorted(time - time[0], maximum_auto_seconds, side="right")))
    count = 0
    for index in range(maximum):
        joint_rms = float(np.sqrt(np.mean(joint_velocity[index] ** 2)))
        if np.linalg.norm(gyro[index]) > 0.25 or joint_rms > 0.25:
            break
        count += 1
    return max(1, count)


def _zero_phase_lowpass(values: np.ndarray, time: np.ndarray, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0.0 or len(values) < 3:
        return values.copy()
    dt = float(np.median(np.diff(time)))
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz * dt)

    def pass_once(source: np.ndarray) -> np.ndarray:
        result = source.copy()
        for index in range(1, len(source)):
            result[index] = result[index - 1] + alpha * (source[index] - result[index - 1])
        return result

    return pass_once(pass_once(values)[::-1])[::-1]


def _robust_leg_measurement(
    candidates: np.ndarray,
    weights: np.ndarray,
    huber_delta: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    effective = weights.copy()
    center = np.average(candidates, axis=0, weights=effective)
    for _ in range(4):
        residual = np.linalg.norm(candidates - center, axis=1)
        robust = np.minimum(1.0, huber_delta / np.maximum(residual, 1e-9))
        effective = weights * robust
        if np.sum(effective) <= 1e-9:
            effective = weights.copy()
        center = np.average(candidates, axis=0, weights=effective)
    residual = np.linalg.norm(candidates - center, axis=1)
    dispersion = float(np.sqrt(np.average(residual**2, weights=effective)))
    return center, dispersion, effective


def _contact_leg_odometry(
    kinematics: Go2Kinematics,
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    gyro: np.ndarray,
    rotation_world_body: np.ndarray,
    force: np.ndarray,
    force_threshold: float,
    full_contact_force: float,
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray, int, float, np.ndarray, float]:
    base_weights = np.clip(
        (force - force_threshold) / max(full_contact_force - force_threshold, 1e-6),
        0.0,
        1.0,
    )
    active = np.flatnonzero(base_weights > 0.0)
    if active.size == 0:
        return (
            np.full(3, np.nan),
            base_weights,
            0,
            0.0,
            np.full((4, 3), np.nan),
            np.inf,
        )

    candidates = []
    all_candidates = np.full((4, 3), np.nan)
    for leg_index in active:
        leg_slice = slice(3 * leg_index, 3 * leg_index + 3)
        foot_position, jacobian = kinematics.foot_position_and_jacobian(
            LEG_NAMES[leg_index], joint_position[leg_slice]
        )
        foot_relative_velocity = jacobian @ joint_velocity[leg_slice]
        body_candidate = -(
            np.cross(gyro, foot_position) + foot_relative_velocity
        )
        world_candidate = rotation_world_body @ body_candidate
        candidates.append(world_candidate)
        all_candidates[leg_index] = world_candidate

    candidate_array = np.asarray(candidates)
    center, dispersion, effective = _robust_leg_measurement(
        candidate_array, base_weights[active], huber_delta
    )
    final_weights = np.zeros(4)
    final_weights[active] = effective
    contact_strength = min(1.0, float(np.sum(final_weights)) / 2.0)
    agreement = np.exp(-((dispersion / max(huber_delta, 1e-6)) ** 2))
    confidence = float(contact_strength * agreement)
    return center, final_weights, int(active.size), confidence, all_candidates, dispersion


def _kalman_filter(
    time: np.ndarray,
    accel: np.ndarray,
    rotations: np.ndarray,
    leg_velocity: np.ndarray,
    contact_confidence: np.ndarray,
    leg_dispersion: np.ndarray,
    initial_accel_bias: np.ndarray,
    accel_noise: float,
    accel_bias_random_walk: float,
    leg_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(time)
    filtered_state = np.zeros((count, 6))
    filtered_covariance = np.zeros((count, 6, 6))
    predicted_state = np.zeros((count, 6))
    predicted_covariance = np.zeros((count, 6, 6))
    transitions = np.repeat(np.eye(6)[None, :, :], count, axis=0)

    state = np.concatenate((np.zeros(3), initial_accel_bias))
    covariance = np.diag([1e-8, 1e-8, 1e-8, 0.35**2, 0.35**2, 0.35**2])
    gravity_world = np.array([0.0, 0.0, -GRAVITY])
    observation = np.zeros((3, 6))
    observation[:, :3] = np.eye(3)

    predicted_state[0] = state
    predicted_covariance[0] = covariance
    filtered_state[0] = state
    filtered_covariance[0] = covariance

    for index in range(1, count):
        dt = float(time[index] - time[index - 1])
        rotation = rotations[index]
        transition = np.eye(6)
        transition[:3, 3:] = -rotation * dt
        acceleration_world = rotation @ (accel[index] - state[3:]) + gravity_world
        prediction = state.copy()
        prediction[:3] += acceleration_world * dt

        process_covariance = np.zeros((6, 6))
        process_covariance[:3, :3] = np.eye(3) * (accel_noise * dt) ** 2
        process_covariance[3:, 3:] = np.eye(3) * accel_bias_random_walk**2 * dt
        prediction_covariance = transition @ covariance @ transition.T + process_covariance

        predicted_state[index] = prediction
        predicted_covariance[index] = prediction_covariance
        transitions[index] = transition

        if np.all(np.isfinite(leg_velocity[index])) and contact_confidence[index] > 1e-4:
            confidence = max(float(contact_confidence[index]), 0.05)
            horizontal_sigma = (leg_sigma + min(float(leg_dispersion[index]), 1.0)) / np.sqrt(confidence)
            vertical_sigma = 1.75 * horizontal_sigma
            measurement_covariance = np.diag(
                [horizontal_sigma**2, horizontal_sigma**2, vertical_sigma**2]
            )
            innovation = leg_velocity[index] - observation @ prediction
            innovation_covariance = (
                observation @ prediction_covariance @ observation.T
                + measurement_covariance
            )
            gain = np.linalg.solve(
                innovation_covariance,
                observation @ prediction_covariance,
            ).T
            state = prediction + gain @ innovation
            identity_correction = np.eye(6) - gain @ observation
            covariance = (
                identity_correction @ prediction_covariance @ identity_correction.T
                + gain @ measurement_covariance @ gain.T
            )
        else:
            state = prediction
            covariance = prediction_covariance

        filtered_state[index] = state
        filtered_covariance[index] = covariance

    smoothed_state = filtered_state.copy()
    smoothed_covariance = filtered_covariance.copy()
    for index in range(count - 2, -1, -1):
        smoother_gain = np.linalg.solve(
            predicted_covariance[index + 1],
            transitions[index + 1] @ filtered_covariance[index],
        ).T
        smoothed_state[index] += smoother_gain @ (
            smoothed_state[index + 1] - predicted_state[index + 1]
        )
        smoothed_covariance[index] += smoother_gain @ (
            smoothed_covariance[index + 1] - predicted_covariance[index + 1]
        ) @ smoother_gain.T

    smoothed_state[0, :3] = 0.0
    return smoothed_state, filtered_state


def estimate(args: argparse.Namespace) -> Estimate:
    data, _ = _extract_log(args.log, args.agent_key)
    time = data["time"]
    joint_position = data["joint_pos"]
    joint_velocity = data["joint_vel"]
    rpy = data["rpy"]
    accel = data["accel"]
    gyro = data["gyro"]
    force = data["force"]
    assert all(value is not None for value in (time, joint_position, joint_velocity, rpy, accel, gyro, force))

    calibration_samples = _leading_stationary_samples(
        time,
        joint_velocity,
        gyro,
        args.stationary_seconds,
        args.max_auto_stationary_seconds,
    )
    gyro_bias = np.median(gyro[:calibration_samples], axis=0)

    yaw = np.unwrap(rpy[:, 2])
    yaw_relative = yaw - yaw[0]
    rotations = np.stack(
        [_rpy_matrix(np.array([angles[0], angles[1], relative_yaw]))
         for angles, relative_yaw in zip(rpy, yaw_relative)]
    )
    expected_specific_force = np.stack(
        [rotation.T @ np.array([0.0, 0.0, GRAVITY]) for rotation in rotations]
    )
    accel_bias = np.median(
        accel[:calibration_samples] - expected_specific_force[:calibration_samples], axis=0
    )

    corrected_gyro = gyro - gyro_bias
    kinematic_joint_velocity = _zero_phase_lowpass(
        joint_velocity, time, args.kinematics_cutoff
    )
    kinematic_gyro = _zero_phase_lowpass(
        corrected_gyro, time, args.kinematics_cutoff
    )
    force_filtered = _median_filter(force, args.contact_median_window)
    kinematics = Go2Kinematics(args.urdf)
    count = len(time)
    leg_velocity = np.full((count, 3), np.nan)
    per_foot_velocity = np.full((count, 4, 3), np.nan)
    contact_weight = np.zeros((count, 4))
    contact_count = np.zeros(count, dtype=int)
    contact_confidence = np.zeros(count)
    leg_dispersion = np.full(count, np.inf)

    for index in range(count):
        (
            leg_velocity[index],
            contact_weight[index],
            contact_count[index],
            contact_confidence[index],
            per_foot_velocity[index],
            leg_dispersion[index],
        ) = _contact_leg_odometry(
            kinematics,
            joint_position[index],
            kinematic_joint_velocity[index],
            kinematic_gyro[index],
            rotations[index],
            force_filtered[index],
            args.contact_threshold,
            args.full_contact_force,
            args.huber_delta,
        )

    smoothed_state, filtered_state = _kalman_filter(
        time,
        accel,
        rotations,
        leg_velocity,
        contact_confidence,
        leg_dispersion,
        accel_bias,
        args.accel_noise,
        args.accel_bias_random_walk,
        args.leg_sigma,
    )
    state = filtered_state if args.forward_only else smoothed_state
    world_velocity = state[:, :3]
    body_velocity = np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), world_velocity)

    world_position = np.zeros_like(world_velocity)
    dt = np.diff(time)
    world_position[1:] = np.cumsum(
        0.5 * (world_velocity[:-1] + world_velocity[1:]) * dt[:, None], axis=0
    )

    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    denominator = np.cos(pitch)
    denominator = np.where(np.abs(denominator) < 0.1, np.sign(denominator) * 0.1, denominator)
    yaw_rate_raw = (
        corrected_gyro[:, 1] * np.sin(roll)
        + corrected_gyro[:, 2] * np.cos(roll)
    ) / denominator
    yaw_rate = _zero_phase_lowpass(yaw_rate_raw, time, args.yaw_cutoff)
    yaw_rate_from_rpy = _zero_phase_lowpass(np.gradient(yaw, time), time, args.yaw_cutoff)

    return Estimate(
        time=time,
        world_velocity=world_velocity,
        body_velocity=body_velocity,
        world_position=world_position,
        leg_world_velocity=leg_velocity,
        yaw=yaw_relative,
        yaw_rate=yaw_rate,
        yaw_rate_from_rpy=yaw_rate_from_rpy,
        contact_force=force,
        contact_weight=contact_weight,
        contact_count=contact_count,
        contact_confidence=contact_confidence,
        accel_bias=state[:, 3:],
        gyro_bias=gyro_bias,
        command_velocity=data["command_velocity"],
        command_yaw_rate=data["command_yaw_rate"],
        old_kf_forward_velocity=data["old_kf_forward_velocity"],
        calibration_samples=calibration_samples,
    )


def _write_csv(path: Path, result: Estimate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "time_s",
        "world_x_m",
        "world_y_m",
        "world_z_m",
        "world_vx_mps",
        "world_vy_mps",
        "world_vz_mps",
        "body_vx_mps",
        "body_vy_mps",
        "body_vz_mps",
        "yaw_rad",
        "yaw_rate_radps",
        "yaw_rate_from_rpy_radps",
        "leg_world_vx_mps",
        "leg_world_vy_mps",
        "leg_world_vz_mps",
        "contact_count",
        "contact_confidence",
        "accel_bias_x_mps2",
        "accel_bias_y_mps2",
        "accel_bias_z_mps2",
    ]
    fields += [f"contact_force_{leg}" for leg in LEG_NAMES]
    fields += [f"contact_weight_{leg}" for leg in LEG_NAMES]
    if result.command_velocity is not None:
        fields += ["command_body_vx_mps", "command_body_vy_mps"]
    if result.command_yaw_rate is not None:
        fields += ["command_yaw_rate_radps"]
    if result.old_kf_forward_velocity is not None:
        fields += ["old_kf_forward_velocity_mps"]

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(result.time):
            row: dict[str, float | int] = {
                "time_s": timestamp,
                "world_x_m": result.world_position[index, 0],
                "world_y_m": result.world_position[index, 1],
                "world_z_m": result.world_position[index, 2],
                "world_vx_mps": result.world_velocity[index, 0],
                "world_vy_mps": result.world_velocity[index, 1],
                "world_vz_mps": result.world_velocity[index, 2],
                "body_vx_mps": result.body_velocity[index, 0],
                "body_vy_mps": result.body_velocity[index, 1],
                "body_vz_mps": result.body_velocity[index, 2],
                "yaw_rad": result.yaw[index],
                "yaw_rate_radps": result.yaw_rate[index],
                "yaw_rate_from_rpy_radps": result.yaw_rate_from_rpy[index],
                "leg_world_vx_mps": result.leg_world_velocity[index, 0],
                "leg_world_vy_mps": result.leg_world_velocity[index, 1],
                "leg_world_vz_mps": result.leg_world_velocity[index, 2],
                "contact_count": int(result.contact_count[index]),
                "contact_confidence": result.contact_confidence[index],
                "accel_bias_x_mps2": result.accel_bias[index, 0],
                "accel_bias_y_mps2": result.accel_bias[index, 1],
                "accel_bias_z_mps2": result.accel_bias[index, 2],
            }
            for leg_index, leg in enumerate(LEG_NAMES):
                row[f"contact_force_{leg}"] = result.contact_force[index, leg_index]
                row[f"contact_weight_{leg}"] = result.contact_weight[index, leg_index]
            if result.command_velocity is not None:
                row["command_body_vx_mps"] = result.command_velocity[index, 0]
                row["command_body_vy_mps"] = result.command_velocity[index, 1]
            if result.command_yaw_rate is not None:
                row["command_yaw_rate_radps"] = result.command_yaw_rate[index, 0]
            if result.old_kf_forward_velocity is not None:
                row["old_kf_forward_velocity_mps"] = result.old_kf_forward_velocity[index]
            writer.writerow(row)


def _write_npz(path: Path, result: Estimate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: value
        for key, value in vars(result).items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(path, **arrays)


def _default_urdf() -> Path:
    return Path(__file__).resolve().parent.parent / "resources/robots/go2/urdf/go2.urdf"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate Go2 x/y/yaw velocity from a hardware log.pkl file."
    )
    parser.add_argument("log", type=Path, help="input log.pkl")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output CSV (default: velocity_estimate.csv beside the input log)",
    )
    parser.add_argument("--npz", action="store_true", help="also write a compressed NPZ")
    parser.add_argument("--agent-key", default="hardware_closed_loop")
    parser.add_argument("--urdf", type=Path, default=_default_urdf())
    parser.add_argument("--contact-threshold", type=float, default=30.0)
    parser.add_argument("--full-contact-force", type=float, default=90.0)
    parser.add_argument("--contact-median-window", type=int, default=5)
    parser.add_argument("--huber-delta", type=float, default=0.35, help="leg disagreement scale in m/s")
    parser.add_argument("--leg-sigma", type=float, default=0.20, help="nominal leg-odometry noise in m/s")
    parser.add_argument("--accel-noise", type=float, default=3.0, help="IMU acceleration noise in m/s^2")
    parser.add_argument("--accel-bias-random-walk", type=float, default=0.10, help="bias drift in m/s^2/sqrt(s)")
    parser.add_argument(
        "--kinematics-cutoff", type=float, default=8.0,
        help="joint-rate and gyro cutoff used by leg odometry in Hz",
    )
    parser.add_argument("--yaw-cutoff", type=float, default=6.0, help="yaw-rate low-pass cutoff in Hz")
    parser.add_argument(
        "--stationary-seconds", type=float, default=0.0,
        help="known stationary prefix used for IMU bias (0: auto-detect)",
    )
    parser.add_argument("--max-auto-stationary-seconds", type=float, default=0.5)
    parser.add_argument(
        "--forward-only", action="store_true",
        help="disable the offline backward Kalman smoothing pass",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = args.log.with_name("velocity_estimate.csv")
    if args.contact_threshold >= args.full_contact_force:
        parser.error("--full-contact-force must exceed --contact-threshold")
    if args.contact_median_window < 1 or args.contact_median_window % 2 == 0:
        parser.error("--contact-median-window must be a positive odd number")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = estimate(args)
        _write_csv(args.output, result)
        if args.npz:
            _write_npz(args.output.with_suffix(".npz"), result)
    except (OSError, ValueError, RuntimeError, ET.ParseError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    duration = result.time[-1] - result.time[0]
    coverage = 100.0 * np.mean(result.contact_count > 0)
    print(f"Processed {len(result.time)} samples over {duration:.2f} s")
    print(
        f"IMU bias calibration: {result.calibration_samples} sample(s); "
        f"gyro bias {np.array2string(result.gyro_bias, precision=4)} rad/s"
    )
    print(f"Leg-odometry contact coverage: {coverage:.1f}%")
    print(
        "Estimated body velocity mean/std: "
        f"vx={np.mean(result.body_velocity[:, 0]):.3f}/{np.std(result.body_velocity[:, 0]):.3f} m/s, "
        f"vy={np.mean(result.body_velocity[:, 1]):.3f}/{np.std(result.body_velocity[:, 1]):.3f} m/s"
    )
    print(
        "Integrated final world position: "
        f"x={result.world_position[-1, 0]:.3f} m, y={result.world_position[-1, 1]:.3f} m"
    )
    print(f"Wrote {args.output}")
    if args.npz:
        print(f"Wrote {args.output.with_suffix('.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
