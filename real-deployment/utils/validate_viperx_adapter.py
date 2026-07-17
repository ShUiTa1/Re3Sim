#!/usr/bin/env python3
"""Validate the ViperX adapter through one bounded live motion."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.dont_write_bytecode = True

REAL_DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
if str(REAL_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOYMENT_DIR))

from viperx_adapter import (
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
    parser.add_argument("--heart-scale-m", type=float, default=0.0025)
    parser.add_argument("--heart-samples", type=int, default=48)
    parser.add_argument("--max-waypoint-step-rad", type=float, default=0.20)
    parser.add_argument("--shadow-step-rad", type=float, default=0.05)
    parser.add_argument("--full-plan", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def mapping_metadata(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        return json.load(file)


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

    q_seed = np.asarray(model.validate_joints(q_start), dtype=np.float64)
    start_pose = np.asarray(model.fk(q_seed), dtype=np.float64)
    waypoints = [q_seed.copy()]
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=True)
    for angle in angles[1:]:
        x = 16.0 * np.sin(angle) ** 3
        z = (
            13.0 * np.cos(angle)
            - 5.0 * np.cos(2.0 * angle)
            - 2.0 * np.cos(3.0 * angle)
            - np.cos(4.0 * angle)
            - 5.0
        )
        target_pose = start_pose.copy()
        target_pose[0, 3] += scale_m * x
        target_pose[2, 3] += scale_m * z
        result = model.ik(target_pose, q0_arm=q_seed)
        if not bool(result.success):
            raise RuntimeError(
                f"Heart IK failed at angle={angle:.3f}: "
                f"reason={result.reason!r}, residual={result.residual!r}"
            )
        q_next = np.asarray(model.validate_joints(result.q_arm), dtype=np.float64)
        step = float(np.max(np.abs(q_next - q_seed)))
        if step > max_waypoint_step_rad:
            raise RuntimeError(
                f"Heart IK path is discontinuous at angle={angle:.3f}: "
                f"max joint step={step:.4f} rad."
            )
        waypoints.append(q_next)
        q_seed = q_next

    return waypoints


def build_shadow_test_targets(
    mapping: ViperXUrdfMapping,
    q_start: Sequence[float],
    *,
    step_rad: float,
) -> list[tuple[str, np.ndarray]]:
    if not np.isfinite(step_rad) or step_rad <= 0.0:
        raise ValueError("shadow_step_rad must be finite and positive.")
    q_start = mapping.validate_q(q_start)
    targets: list[tuple[str, np.ndarray]] = []
    for joint in ("shoulder", "elbow"):
        index = mapping.joint_order.index(joint)
        for direction in (1.0, -1.0):
            target = q_start.copy()
            target[index] += direction * step_rad
            try:
                mapping.rad_to_actuator_raw(target)
            except ValueError:
                continue
            targets.append((joint, target))
            break
        else:
            raise ValueError(f"No safe {step_rad:.6f} rad test step is available for {joint}.")
    return targets


def validate_shadow_goal_equivalence(
    mapping: ViperXUrdfMapping,
    raw_goal: dict[str, int],
) -> dict[str, float]:
    raw_goal = mapping.validate_actuator_raw(raw_goal)
    mismatch_rad: dict[str, float] = {}
    for shadow_index, shadow in enumerate(mapping.shadow_names):
        joint_index = int(mapping.shadow_joint_indices[shadow_index])
        joint = mapping.joint_order[joint_index]
        main_delta = (
            mapping.sign[joint_index]
            * (raw_goal[joint] - mapping.raw_home[joint_index])
            * mapping.scale_rad_per_tick[joint_index]
        )
        shadow_delta = (
            mapping.shadow_sign[shadow_index]
            * (raw_goal[shadow] - mapping.shadow_raw_home[shadow_index])
            * mapping.shadow_scale_rad_per_tick[shadow_index]
        )
        mismatch = float(abs(main_delta - shadow_delta))
        tolerance = float(
            0.5
            * (
                mapping.scale_rad_per_tick[joint_index]
                + mapping.shadow_scale_rad_per_tick[shadow_index]
            )
            + 1e-12
        )
        if mismatch > tolerance:
            raise RuntimeError(
                f"{joint}/{shadow} goal increments disagree by {mismatch:.9f} rad."
            )
        mismatch_rad[shadow] = mismatch
    return mismatch_rad


def read_shadow_diagnostics(robot: Any, mapping: ViperXUrdfMapping) -> dict[str, dict[str, int]]:
    diagnostics: dict[str, dict[str, int]] = {}
    for register in (
        "Goal_Position",
        "Present_Position",
        "Present_Current",
        "Present_PWM",
        "Hardware_Error_Status",
    ):
        values = robot.bus.sync_read(register, list(mapping.actuator_names), normalize=False)
        diagnostics[register] = {name: int(values[name]) for name in mapping.actuator_names}
    return diagnostics


def run_shadow_live_check(
    adapter: ViperXAdapter,
    mapping: ViperXUrdfMapping,
    *,
    step_rad: float,
) -> None:
    if step_rad <= adapter.position_tolerance_rad:
        raise ValueError("shadow_step_rad must exceed the adapter position tolerance.")
    initial_goal = adapter.robot.bus.sync_read(
        "Goal_Position", list(mapping.actuator_names), normalize=False
    )
    expected_initial = adapter._session_start_raw
    if expected_initial is None or any(
        int(initial_goal[name]) != expected_initial[name] for name in mapping.actuator_names
    ):
        raise RuntimeError("The eight-actuator initial hold goal was not applied exactly.")
    print("shadow_initial_hold=PASS")

    q_start = adapter.read_joints()
    for joint, target in build_shadow_test_targets(mapping, q_start, step_rad=step_rad):
        adapter.move_joints(target)
        diagnostics = read_shadow_diagnostics(adapter.robot, mapping)
        mismatch = validate_shadow_goal_equivalence(mapping, diagnostics["Goal_Position"])
        shadow_index = ("shoulder", "elbow").index(joint)
        shadow = mapping.shadow_names[shadow_index]
        pair = (joint, shadow)
        pair_diagnostics = {
            register: {name: values[name] for name in pair}
            for register, values in diagnostics.items()
        }
        print(f"shadow_pair={joint} diagnostics={json.dumps(pair_diagnostics, sort_keys=True)}")
        print(f"shadow_pair={joint} goal_mismatch_rad={mismatch[shadow]:.9f}")
        hardware_errors = pair_diagnostics["Hardware_Error_Status"]
        if any(value != 0 for value in hardware_errors.values()):
            raise RuntimeError(f"Hardware error during {joint} shadow check: {hardware_errors}")
        adapter.move_joints(q_start)

    print("shadow_load_observation=REVIEW_PRESENT_CURRENT_PWM_AND_MECHANICAL_BEHAVIOR")
    print("shadow_pair_live_check=PASS")


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

    heart_waypoints: list[np.ndarray] = []
    q_prepose: np.ndarray | None = None
    if args.full_plan:
        shoulder_index = mapping.joint_order.index("shoulder")
        q_prepose = mapping.q_home_urdf.copy()
        q_prepose[shoulder_index] = -np.pi / 2.0
        q_prepose = model.validate_joints(mapping.validate_q(q_prepose))
        heart_waypoints = build_heart_waypoints(
            model,
            q_prepose,
            scale_m=args.heart_scale_m,
            samples=args.heart_samples,
            max_waypoint_step_rad=args.max_waypoint_step_rad,
        )
        planned_joint_targets = [
            mapping.q_home_urdf,
            *heart_waypoints,
            mapping.q_home_urdf,
        ]
        for q_target in planned_joint_targets:
            mapping.rad_to_actuator_raw(model.validate_joints(q_target))
        print("raw_plan_preflight=PASS")
        print("live_plan=home -> shoulder_left_-90deg -> base_XZ_heart -> home")
        print(f"heart_waypoints={len(heart_waypoints)}")
        confirmation = "MOVE"
    else:
        shadow_targets = build_shadow_test_targets(
            mapping, mapping.q_home_urdf, step_rad=args.shadow_step_rad
        )
        for _, q_target in shadow_targets:
            validate_shadow_goal_equivalence(mapping, mapping.rad_to_actuator_raw(q_target))
        print("shadow_raw_plan_preflight=PASS")
        print("live_plan=current_pose -> small_shoulder_check -> small_elbow_check -> current_pose")
        confirmation = "SHADOW"

    if not args.yes:
        answer = input(f"Type {confirmation} to execute this live test: ").strip()
        if answer != confirmation:
            raise SystemExit("Live validation aborted before hardware connection.")

    robot = make_robot_from_config(
        ViperXConfig(port=port, id=robot_id, calibration_dir=calibration_dir)
    )
    adapter = ViperXAdapter(robot, model, mapping, interactive_safety=True)
    if not args.full_plan and args.shadow_step_rad <= adapter.position_tolerance_rad:
        raise ValueError("shadow_step_rad must exceed the adapter position tolerance.")
    try:
        adapter.connect()
        if args.full_plan:
            if q_prepose is None:
                raise RuntimeError("Full-plan prepose was not created.")
            adapter.go_home()
            adapter.move_joints(q_prepose)
            for waypoint in heart_waypoints[1:]:
                adapter.move_joints(waypoint)
            adapter.go_home()
        else:
            run_shadow_live_check(adapter, mapping, step_rad=args.shadow_step_rad)
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

    if args.full_plan:
        print("live_home_shoulder_heart_home=PASS")
    else:
        print("live_shadow_pair_check=PASS")
    print("validate_viperx_adapter=PASS")


def main() -> None:
    configure_runtime_environment()
    args = parse_args()
    mapping = ViperXUrdfMapping.load(args.mapping)
    run_live(args, mapping)


if __name__ == "__main__":
    main()
