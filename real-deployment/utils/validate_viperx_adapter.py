#!/usr/bin/env python3
"""Validate the ViperX adapter through one bounded live motion."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence
from scipy.optimize import least_squares

import numpy as np

sys.dont_write_bytecode = True

REAL_DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(REAL_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOYMENT_DIR))

from viperx_adapter import (
    STARTUP_STAGING_Q_RAD,
    ViperXAdapter,
    ViperXUrdfMapping,
)

DEFAULT_MAPPING = REAL_DEPLOYMENT_DIR / "configs" / "viperx_urdf_mapping.json"
DEFAULT_MPLCONFIGDIR = Path("/data/yuzzhu/Re3Sim_ViperX/cache/matplotlib")


def configure_runtime_environment() -> None:
    mplconfigdir = Path(os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR)))
    mplconfigdir.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--port")
    parser.add_argument("--robot-id")
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--heart-scale-m", type=float, default=0.005)
    parser.add_argument("--heart-samples", type=int, default=50)
    parser.add_argument("--max-waypoint-step-rad", type=float, default=0.40)
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def mapping_metadata(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        return json.load(file)


# def build_heart_waypoints(
#     model: Any,
#     q_start: Sequence[float],
#     *,
#     scale_m: float,
#     samples: int,
#     max_waypoint_step_rad: float,
# ) -> list[np.ndarray]:
#     if not np.isfinite(scale_m) or scale_m <= 0.0:
#         raise ValueError("heart_scale_m must be finite and positive.")
#     if samples < 12:
#         raise ValueError("heart_samples must be at least 12.")
#     if not np.isfinite(max_waypoint_step_rad) or max_waypoint_step_rad <= 0.0:
#         raise ValueError("max_waypoint_step_rad must be finite and positive.")

#     q_seed = np.asarray(model.validate_joints(q_start), dtype=np.float64)
#     start_pose = np.asarray(model.fk(q_seed), dtype=np.float64)
#     waypoints = [q_seed.copy()]
#     angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
#     for angle in angles[1:]:
#         x = 16.0 * np.sin(angle) ** 3
#         z = (
#             13.0 * np.cos(angle)
#             - 5.0 * np.cos(2.0 * angle)
#             - 2.0 * np.cos(3.0 * angle)
#             - np.cos(4.0 * angle)
#             - 5.0
#         )
#         target_pose = start_pose.copy()
#         target_pose[0, 3] += scale_m * x
#         target_pose[2, 3] += scale_m * z
#         result = model.ik(target_pose, q0_arm=q_seed)
#         if not bool(result.success):
#             raise RuntimeError(
#                 f"Heart IK failed at angle={angle:.3f}: "
#                 f"reason={result.reason!r}, residual={result.residual!r}"
#             )
#         q_next = np.asarray(model.validate_joints(result.q_arm), dtype=np.float64)
#         joint_delta = q_next - q_seed
#         abs_joint_delta = np.abs(joint_delta)
#         step_index = int(np.argmax(abs_joint_delta))
#         step = float(abs_joint_delta[step_index])

#         if step > max_waypoint_step_rad:
#             joint_details = "\n".join(
#                 (
#                     f"  {name}: "
#                     f"previous={q_seed[index]:.6f}, "
#                     f"next={q_next[index]:.6f}, "
#                     f"delta={joint_delta[index]:+.6f} rad "
#                     f"({np.rad2deg(joint_delta[index]):+.3f} deg)"
#                 )
#                 for index, name in enumerate(model.arm_joint_names)
#             )

#             raise RuntimeError(
#                 f"Heart IK path is discontinuous at angle={angle:.3f}: "
#                 f"max joint step={step:.4f} rad.\n"
#                 f"largest_joint={model.arm_joint_names[step_index]}\n"
#                 f"joint_deltas:\n{joint_details}"
#             )
#         waypoints.append(q_next)
#         q_seed = q_next

#     return waypoints
def build_heart_waypoints(
    model: Any,
    q_start: Sequence[float],
    *,
    scale_m: float,
    samples: int,
    max_waypoint_step_rad: float,
) -> list[np.ndarray]:
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("heart_scale_m must be finite and positive.")
    if samples < 12:
        raise ValueError("heart_samples must be at least 12.")
    if not np.isfinite(max_waypoint_step_rad) or max_waypoint_step_rad <= 0.0:
        raise ValueError("max_waypoint_step_rad must be finite and positive.")

    q = np.asarray(model.validate_joints(q_start), dtype=np.float64)
    start_xyz = np.asarray(model.fk(q), dtype=np.float64)[:3, 3]

    names = tuple(model.arm_joint_names)
    locked = [
        names.index("forearm_roll"),
        names.index("wrist_rotate"),
    ]
    free = [index for index in range(len(names)) if index not in locked]
    locked_values = q[locked].copy()
    lower, upper = model.qlim[:, free]

    waypoints = [q.copy()]
    angles = np.linspace(0.0, 2.0 * np.pi, samples)

    for angle in angles[1:]:
        x = 16.0 * np.sin(angle) ** 3
        z = (
            13.0 * np.cos(angle)
            - 5.0 * np.cos(2.0 * angle)
            - 2.0 * np.cos(3.0 * angle)
            - np.cos(4.0 * angle)
            - 5.0
        )
        target_xyz = start_xyz + scale_m * np.array([x, 0.0, z])

        def position_error(q_free: np.ndarray) -> np.ndarray:
            candidate = q.copy()
            candidate[free] = q_free
            candidate[locked] = locked_values
            return np.asarray(model.fk(candidate))[:3, 3] - target_xyz

        result = least_squares(
            position_error,
            q[free],
            bounds=(lower, upper),
            max_nfev=300,
        )

        q_next = q.copy()
        q_next[free] = result.x
        q_next[locked] = locked_values
        q_next = np.asarray(model.validate_joints(q_next), dtype=np.float64)

        error_m = float(np.linalg.norm(position_error(q_next[free])))
        if not result.success or error_m > 1e-5:
            raise RuntimeError(
                f"Heart position IK failed at angle={angle:.3f}: "
                f"position_error={error_m:.6e} m."
            )

        delta = np.abs(q_next - q)
        largest = int(np.argmax(delta))
        if delta[largest] > max_waypoint_step_rad:
            raise RuntimeError(
                f"Heart path is discontinuous at angle={angle:.3f}: "
                f"{names[largest]} step={delta[largest]:.4f} rad."
            )

        waypoints.append(q_next)
        q = q_next

    return waypoints

def run_live(args: argparse.Namespace, mapping: ViperXUrdfMapping) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Live validation requires an interactive terminal for the safety controls.")

    from lerobot.robots.utils import make_robot_from_config
    from lerobot.robots.viperx.config_viperx import ViperXConfig
    from viperx_model import load_viperx_model

    metadata = mapping_metadata(args.mapping)
    lerobot = metadata["lerobot"]
    port = args.port or str(lerobot["port"])
    robot_id = args.robot_id or str(lerobot["robot_id"])
    calibration_dir = args.calibration_dir or Path(str(lerobot["calibration_dir"]))
    model = load_viperx_model(
        urdf_path=metadata["urdf_path"],
        base_link=mapping.base_link,
        end_link=mapping.end_link,
    )

    q_staging = np.asarray(
        model.validate_joints(mapping.validate_q(STARTUP_STAGING_Q_RAD)), dtype=np.float64
    )
    waist_index = mapping.joint_order.index("waist")
    q_prepose = mapping.q_home_urdf.copy()
    q_prepose[waist_index] = np.deg2rad(40.0)
    q_prepose = np.asarray(
        model.validate_joints(mapping.validate_q(q_prepose)), dtype=np.float64
    )
    heart_waypoints = build_heart_waypoints(
        model,
        q_prepose,
        scale_m=args.heart_scale_m,
        samples=args.heart_samples,
        max_waypoint_step_rad=args.max_waypoint_step_rad,
    )
    planned_joint_targets = [
        q_staging,
        mapping.q_home_urdf,
        *heart_waypoints,
        mapping.q_home_urdf,
    ]
    for q_target in planned_joint_targets:
        mapping.rad_to_actuator_raw(model.validate_joints(q_target))
    print("raw_plan_preflight=PASS")
    print(
        "live_plan=startup_staging -> home -> waist_plus_40deg "
        "-> base_XZ_heart -> home"
    )
    print(f"heart_waypoints={len(heart_waypoints)}")

    if not args.yes:
        answer = input("Type MOVE to execute this live test: ").strip()
        if answer != "MOVE":
            raise SystemExit("Live validation aborted before hardware connection.")

    robot = make_robot_from_config(
        ViperXConfig(port=port, id=robot_id, calibration_dir=calibration_dir)
    )
    adapter = ViperXAdapter(robot, model, mapping, interactive_safety=True)
    try:
        adapter.connect()
        adapter.prepare_for_motion()
        adapter.go_home()
        adapter.move_joints(q_prepose)
        for waypoint in heart_waypoints[1:]:
            adapter.move_joints(waypoint)
        adapter.go_home()
        adapter.stop_motion()
        adapter.await_release()
    except BaseException:
        if adapter.is_connected:
            try:
                adapter.stop_motion()
                adapter.await_release(prepare_raw=adapter._session_start_raw)
            except Exception:
                pass
        raise

    print("live_staging_home_waist_heart_home=PASS")
    print("validate_viperx_adapter=PASS")


def main() -> None:
    configure_runtime_environment()
    args = parse_args()
    mapping = ViperXUrdfMapping.load(args.mapping)
    run_live(args, mapping)


if __name__ == "__main__":
    main()
