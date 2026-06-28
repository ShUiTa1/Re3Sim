#!/usr/bin/env python3
"""Stage 2 validation: Robotics Toolbox FK/IK vs PyBullet."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pybullet as p

sys.dont_write_bytecode = True

REAL_DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(REAL_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOYMENT_DIR))

from viperx_model import DEFAULT_URDF_PATH, load_viperx_model
from viperx_model import (
    DEFAULT_IK_POSITION_TOLERANCE_M,
    DEFAULT_IK_ROTATION_TOLERANCE_RAD,
)


POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-6
IK_POSITION_TOLERANCE_M = DEFAULT_IK_POSITION_TOLERANCE_M
IK_ROTATION_TOLERANCE_RAD = DEFAULT_IK_ROTATION_TOLERANCE_RAD
DEFAULT_MPLCONFIGDIR = Path("/tmp/matplotlib")
IK_SAMPLE_LABELS = (
    "zero",
    "reach_forward",
    "left_pose",
    "right_pose",
    "mid_limits",
)


def configure_runtime_environment() -> None:
    """Keep validation-only cache/config writes out of the model module."""

    mplconfigdir = Path(
        os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
    )
    mplconfigdir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--base-link", default="vx300s/base_link")
    parser.add_argument("--end-link", default="vx300s/ee_gripper_link")
    return parser.parse_args()


def rotation_error_rad(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.T @ right
    cos_theta = (np.trace(delta) - 1.0) / 2.0
    return float(math.acos(np.clip(cos_theta, -1.0, 1.0)))


def pybullet_transform(robot_id: int, link_index: int) -> np.ndarray:
    state = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    position = np.asarray(state[4], dtype=np.float64)
    quaternion = state[5]
    rotation = np.asarray(
        p.getMatrixFromQuaternion(quaternion),
        dtype=np.float64,
    ).reshape(3, 3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def build_pybullet_joint_maps(robot_id: int) -> tuple[dict[str, int], dict[str, int]]:
    joint_indices: dict[str, int] = {}
    child_link_indices: dict[str, int] = {}
    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        joint_name = info[1].decode("utf-8")
        child_link = info[12].decode("utf-8")
        joint_indices[joint_name] = joint_index
        child_link_indices[child_link] = joint_index
    return joint_indices, child_link_indices


def apply_q_full_to_pybullet(
    robot_id: int,
    joint_indices: dict[str, int],
    joint_names: tuple[str, ...],
    q_full: np.ndarray,
) -> None:
    for joint_name, joint_value in zip(joint_names, q_full):
        if joint_name not in joint_indices:
            raise ValueError(f"PyBullet model has no joint {joint_name!r}.")
        p.resetJointState(robot_id, joint_indices[joint_name], float(joint_value))


def sample_configurations(qlim: np.ndarray) -> dict[str, np.ndarray]:
    lower, upper = qlim
    mid = (lower + upper) / 2.0
    span = upper - lower
    return {
        "zero": np.zeros(6, dtype=np.float64),
        "waist_only": np.array([0.35, 0, 0, 0, 0, 0], dtype=np.float64),
        "shoulder_only": np.array([0, -0.45, 0, 0, 0, 0], dtype=np.float64),
        "elbow_only": np.array([0, 0, 0.55, 0, 0, 0], dtype=np.float64),
        "forearm_roll_only": np.array([0, 0, 0, 0.65, 0, 0], dtype=np.float64),
        "wrist_angle_only": np.array([0, 0, 0, 0, -0.55, 0], dtype=np.float64),
        "wrist_rotate_only": np.array([0, 0, 0, 0, 0, 0.8], dtype=np.float64),
        "reach_forward": np.array([0.0, -0.65, 0.9, 0.0, -0.35, 0.0], dtype=np.float64),
        "left_pose": np.array([0.45, -0.7, 1.0, 0.35, -0.45, 0.4], dtype=np.float64),
        "right_pose": np.array([-0.45, -0.55, 0.85, -0.35, 0.25, -0.4], dtype=np.float64),
        "near_lower_limits": lower + 0.10 * span,
        "near_upper_limits": upper - 0.10 * span,
        "mid_limits": mid,
    }
def validate_sample_configurations(
    samples: dict[str, np.ndarray],
    qlim: np.ndarray,
    *,
    atol: float = 1e-12,
) -> None:
    lower, upper = np.asarray(qlim, dtype=np.float64)

    for label, q_arm in samples.items():
        q_arm = np.asarray(q_arm, dtype=np.float64)

        if q_arm.shape != lower.shape:
            raise AssertionError(
                f"{label} has shape {q_arm.shape}, expected {lower.shape}."
            )

        if not np.all(np.isfinite(q_arm)):
            raise AssertionError(f"{label} contains NaN or infinity: {q_arm}.")

        below = q_arm < lower - atol
        above = q_arm > upper + atol
        if np.any(below) or np.any(above):
            raise AssertionError(
                f"{label} outside qlim: q={q_arm}, lower={lower}, upper={upper}."
            )
 


def make_ik_seed(q_arm: np.ndarray, qlim: np.ndarray) -> np.ndarray:
    perturbation = np.array(
        [0.16, -0.18, 0.20, -0.15, 0.16, -0.18],
        dtype=np.float64,
    )
    lower, upper = qlim
    return np.clip(q_arm + perturbation, lower + 1e-6, upper - 1e-6)


def main() -> None:
    configure_runtime_environment()
    args = parse_args()
    model = load_viperx_model(
        urdf_path=args.urdf,
        base_link=args.base_link,
        end_link=args.end_link,
    )

    print(f"urdf={Path(args.urdf).resolve()}")
    print(f"rtb_full_n={model.full_n}")
    print(f"wrapper_arm_n={model.n}")
    print(f"base_link={model.base_link}")
    print(f"end_link={model.end_link}")
    print(f"rtb_joint_names={model.joint_names}")
    print(f"arm_joint_names={model.arm_joint_names}")
    print(f"passive_defaults={model.passive_joint_values}")

    cid = p.connect(p.DIRECT)
    if cid < 0:
        raise RuntimeError("Failed to connect to PyBullet DIRECT.")
    try:
        robot_id = p.loadURDF(str(Path(args.urdf).resolve()), useFixedBase=True)
        joint_indices, child_link_indices = build_pybullet_joint_maps(robot_id)
        if model.end_link not in child_link_indices:
            raise ValueError(f"PyBullet model has no child link {model.end_link!r}.")
        end_link_index = child_link_indices[model.end_link]

        max_fk_position_error = 0.0
        max_fk_rotation_error = 0.0
        pybullet_targets: dict[str, np.ndarray] = {}
        samples = sample_configurations(model.qlim)
        validate_sample_configurations(samples, model.qlim)
        for label, q_arm in samples.items():

            q_full = model.expand_q_arm(q_arm)
            T_rtb = model.fk(q_arm)
            apply_q_full_to_pybullet(robot_id, joint_indices, model.joint_names, q_full)
            T_pb = pybullet_transform(robot_id, end_link_index)
            pybullet_targets[label] = T_pb

            position_error = float(np.linalg.norm(T_rtb[:3, 3] - T_pb[:3, 3]))
            rot_error = rotation_error_rad(T_pb[:3, :3], T_rtb[:3, :3])
            max_fk_position_error = max(max_fk_position_error, position_error)
            max_fk_rotation_error = max(max_fk_rotation_error, rot_error)

            print(
                f"{label}: pos_err={position_error:.3e}m "
                f"rot_err={rot_error:.3e}rad "
                f"rtb_xyz=({T_rtb[0,3]:.6f},{T_rtb[1,3]:.6f},{T_rtb[2,3]:.6f}) "
                f"pb_xyz=({T_pb[0,3]:.6f},{T_pb[1,3]:.6f},{T_pb[2,3]:.6f})"
            )

            if position_error > POSITION_TOLERANCE_M:
                raise AssertionError(f"{label}: position error too large.")
            if rot_error > ROTATION_TOLERANCE_RAD:
                raise AssertionError(f"{label}: rotation error too large.")

        max_ik_position_error = 0.0
        max_ik_rotation_error = 0.0
        for label in IK_SAMPLE_LABELS:
            q_target = samples[label]
            target_transform = pybullet_targets[label]
            q0_arm = make_ik_seed(q_target, model.qlim)
            result = model.ik(target_transform, q0_arm=q0_arm)
            if not result.success:
                raise AssertionError(f"{label}: IK failed: {result.reason}")

            q_solution_full = model.expand_q_arm(result.q_arm)
            apply_q_full_to_pybullet(
                robot_id,
                joint_indices,
                model.joint_names,
                q_solution_full,
            )
            T_solution_pb = pybullet_transform(robot_id, end_link_index)
            position_error, rot_error = model.transform_error(
                target_transform,
                T_solution_pb,
            )
            max_ik_position_error = max(max_ik_position_error, position_error)
            max_ik_rotation_error = max(max_ik_rotation_error, rot_error)

            print(
                f"ik_{label}: pos_err={position_error:.3e}m "
                f"rot_err={rot_error:.3e}rad "
                f"residual={result.residual:.3e} "
                f"iterations={result.iterations} "
                f"searches={result.searches}"
            )

            if position_error > IK_POSITION_TOLERANCE_M:
                raise AssertionError(f"{label}: IK position error too large.")
            if rot_error > IK_ROTATION_TOLERANCE_RAD:
                raise AssertionError(f"{label}: IK rotation error too large.")

        unreachable_transform = np.eye(4, dtype=np.float64)
        unreachable_transform[:3, 3] = [3.0, 0.0, 3.0]
        unreachable_result = model.ik(
            unreachable_transform,
            q0_arm=np.zeros(model.n, dtype=np.float64),
        )
        if np.isfinite(unreachable_result.position_error) and np.isfinite(
            unreachable_result.rotation_error
        ):
            error_summary = (
                f"pos_err={unreachable_result.position_error:.3e}m "
                f"rot_err={unreachable_result.rotation_error:.3e}rad "
            )
        else:
            error_summary = ""
        print(
            "unreachable_target: "
            f"success={unreachable_result.success} "
            f"{error_summary}"
            f"reason={unreachable_result.reason}"
        )
        if unreachable_result.success:
            raise AssertionError("Unreachable target unexpectedly succeeded.")
        print("unreachable_target_check=PASS")
    finally:
        p.disconnect()

    print(f"max_fk_position_error={max_fk_position_error:.3e}m")
    print(f"max_fk_rotation_error={max_fk_rotation_error:.3e}rad")
    print(f"max_ik_position_error={max_ik_position_error:.3e}m")
    print(f"max_ik_rotation_error={max_ik_rotation_error:.3e}rad")
    print("validate_viperx_model=PASS")


if __name__ == "__main__":
    main()
