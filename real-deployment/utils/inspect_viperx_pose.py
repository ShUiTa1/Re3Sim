#!/usr/bin/env python3
"""Inspect a torque-off ViperX pose and capture one wrist-camera frame pair.

Each snapshot reports the Stage-4 raw encoder values, mapped URDF joint
radians, and the Stage-2 FK end-effector position, then saves one RGB/depth
pair. The script never writes a position target or enables torque.
"""

import argparse
from datetime import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

sys.dont_write_bytecode = True

REAL_DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING_PATH = REAL_DEPLOYMENT_DIR / "configs" / "viperx_urdf_mapping.json"
DEFAULT_MPLCONFIGDIR = Path("/data/yuzzhu/Re3Sim_ViperX/cache/matplotlib")
DEFAULT_ASSETS_DIR = UTILS_DIR / "assets"
RGB_RESOLUTION = (640, 480)
DEPTH_RESOLUTION = (640, 480)
CAMERA_WARMUP_S = 5.0

if str(REAL_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOYMENT_DIR))

from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.viperx.config_viperx import ViperXConfig

from calibration.realsense.realsense import Camera, get_devices
from viperx_adapter import ViperXUrdfMapping
from viperx_model import load_viperx_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    parser.add_argument("--camera-serial", required=True)
    return parser.parse_args()


def configure_runtime_environment() -> None:
    cache_dir = Path(os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR)))
    cache_dir.mkdir(parents=True, exist_ok=True)


def load_mapping_metadata(path: Path) -> dict[str, Any]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    missing = {"urdf_path", "lerobot"} - set(metadata)
    if missing:
        raise ValueError(f"Mapping metadata is missing keys: {sorted(missing)}")
    missing_lerobot = {"port", "robot_id", "calibration_dir"} - set(metadata["lerobot"])
    if missing_lerobot:
        raise ValueError(
            f"Mapping lerobot metadata is missing keys: {sorted(missing_lerobot)}"
        )
    return metadata


def make_viperx_robot(mapping: ViperXUrdfMapping, metadata: dict[str, Any]) -> Any:
    lerobot = metadata["lerobot"]
    if str(lerobot["robot_id"]) != mapping.robot_id:
        raise ValueError("Mapping robot_id does not match its LeRobot metadata.")
    return make_robot_from_config(
        ViperXConfig(
            port=str(lerobot["port"]),
            id=str(lerobot["robot_id"]),
            calibration_dir=Path(str(lerobot["calibration_dir"])),
        )
    )


def validate_bus_contract(robot: Any, mapping: ViperXUrdfMapping) -> None:
    arm_actuators = {name for name in robot.bus.motors if name != "gripper"}
    if arm_actuators != set(mapping.actuator_names):
        raise ValueError(
            "LeRobot arm actuators do not match the Stage-4 mapping: "
            f"bus={sorted(arm_actuators)}, mapping={sorted(mapping.actuator_names)}"
        )


def ensure_torque_disabled(robot: Any) -> None:
    motor_names = list(robot.bus.motors)
    states = robot.bus.sync_read(
        "Torque_Enable", motor_names, normalize=False, num_retry=2
    )
    enabled = [name for name in motor_names if int(states[name]) != 0]
    if enabled:
        robot.bus.disable_torque(num_retry=5)
        raise RuntimeError(f"Torque became enabled on motors: {enabled}")


def read_raw_actuators(
    robot: Any, actuator_names: Sequence[str]
) -> dict[str, int]:
    values = robot.bus.sync_read(
        "Present_Position", list(actuator_names), normalize=False, num_retry=2
    )
    return {name: int(values[name]) for name in actuator_names}


def raw_to_measured_q(
    mapping: ViperXUrdfMapping, raw_positions: dict[str, int]
) -> np.ndarray:
    raw = np.asarray(
        [raw_positions[name] for name in mapping.joint_order], dtype=np.float64
    )
    return (
        mapping.q_home_urdf
        + mapping.sign * (raw - mapping.raw_home) * mapping.scale_rad_per_tick
    )


def measured_fk(model: Any, q_rad: Sequence[float]) -> np.ndarray:
    """Compute diagnostic FK without rejecting a measured out-of-limit pose."""

    q = np.asarray(q_rad, dtype=np.float64)
    if q.shape != (len(model.arm_joint_indices),) or not np.all(np.isfinite(q)):
        raise ValueError("Measured q must contain six finite joint radians.")

    q_full = model.full_joint_defaults.copy()
    for source_index, target_index in enumerate(model.arm_joint_indices):
        q_full[target_index] = q[source_index]
    pose = model.robot_model.fkine(q_full, start=model.base_link, end=model.end_link)
    return model.validate_transform(getattr(pose, "A", pose))


def range_violations(
    names: Sequence[str],
    values: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> list[str]:
    violations = []
    for name, value, minimum, maximum in zip(names, values, lower, upper):
        if value < minimum or value > maximum:
            violations.append(
                f"{name}={value:.6f} outside [{minimum:.6f}, {maximum:.6f}]"
            )
    return violations


def capture_frame_pair(camera: Camera, output_dir: Path) -> tuple[Path, Path]:
    rgb_image, depth_image = camera.shoot()
    if rgb_image is None or depth_image is None:
        raise RuntimeError("RealSense returned an incomplete RGB/depth frame pair.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / f"rgb_{timestamp}.png"
    depth_path = output_dir / f"depth_{timestamp}.npy"

    bgr_image = cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(rgb_path), bgr_image):
        raise IOError(f"Failed to save RGB image: {rgb_path}")
    np.save(depth_path, np.asarray(depth_image))

    print(f"rgb_saved={rgb_path}")
    print(f"depth_saved={depth_path}")
    return rgb_path, depth_path


def print_snapshot(
    sample_index: int,
    robot: Any,
    model: Any,
    mapping: ViperXUrdfMapping,
    camera: Camera,
    output_dir: Path,
) -> None:
    ensure_torque_disabled(robot)
    raw = read_raw_actuators(robot, mapping.actuator_names)
    q_rad = raw_to_measured_q(mapping, raw)
    ee_pose = measured_fk(model, q_rad)

    main_raw = [raw[name] for name in mapping.joint_order]
    shadow_raw = [raw[name] for name in mapping.shadow_names]
    raw_warnings = range_violations(
        mapping.joint_order, main_raw, mapping.raw_min, mapping.raw_max
    )
    raw_warnings += range_violations(
        mapping.shadow_names,
        shadow_raw,
        mapping.shadow_raw_min,
        mapping.shadow_raw_max,
    )
    q_warnings = range_violations(
        mapping.joint_order, q_rad, mapping.q_lower, mapping.q_upper
    )

    print(f"\nsnapshot={sample_index}")
    print("raw_ticks:")
    print("  " + "  ".join(f"{name}={raw[name]}" for name in mapping.joint_order))
    print(
        "  "
        + "  ".join(f"{name}={raw[name]}" for name in mapping.shadow_names)
    )
    print("q_urdf_rad:")
    print(
        "  "
        + "  ".join(
            f"{name}={q_rad[index]:.6f}"
            for index, name in enumerate(mapping.joint_order)
        )
    )
    xyz = ee_pose[:3, 3]
    print(f"ee_frame: {mapping.base_link} -> {mapping.end_link}")
    print(f"ee_xyz_m: x={xyz[0]:.6f}  y={xyz[1]:.6f}  z={xyz[2]:.6f}")
    print("raw_range_check=" + ("PASS" if not raw_warnings else "WARNING"))
    for warning in raw_warnings:
        print(f"  {warning}")
    print("urdf_limit_check=" + ("PASS" if not q_warnings else "WARNING"))
    for warning in q_warnings:
        print(f"  {warning}")
    capture_frame_pair(camera, output_dir)


def wait_for_start() -> bool:
    print("Support the ViperX before continuing; the script will disable torque.")
    while True:
        answer = input("Place the arm, then press Enter (or E + Enter to exit): ").strip()
        if not answer:
            return True
        if answer.casefold() == "e":
            return False
        print("Use Enter to continue or E + Enter to exit.")


def inspect_loop(
    robot: Any,
    model: Any,
    mapping: ViperXUrdfMapping,
    camera: Camera,
    output_dir: Path,
) -> None:
    sample_index = 1
    print_snapshot(sample_index, robot, model, mapping, camera, output_dir)
    while True:
        answer = input(
            "Press Enter for another pose snapshot and camera capture, "
            "or E + Enter to exit: "
        ).strip()
        if answer.casefold() == "e":
            return
        if answer:
            print("Use Enter for a snapshot/capture or E + Enter to exit.")
            continue
        sample_index += 1
        print_snapshot(sample_index, robot, model, mapping, camera, output_dir)


def disconnect_torque_off(robot: Any) -> None:
    if not robot.bus.is_connected:
        return
    try:
        robot.bus.disable_torque(num_retry=5)
    finally:
        robot.bus.disconnect(disable_torque=False)


def main() -> None:
    args = parse_args()
    configure_runtime_environment()

    mapping = ViperXUrdfMapping.load(args.mapping)
    metadata = load_mapping_metadata(args.mapping)
    model = load_viperx_model(
        urdf_path=metadata["urdf_path"],
        base_link=mapping.base_link,
        end_link=mapping.end_link,
    )
    robot = make_viperx_robot(mapping, metadata)
    validate_bus_contract(robot, mapping)
    device_serials = get_devices()
    if args.camera_serial not in device_serials:
        raise RuntimeError(
            f"RealSense {args.camera_serial!r} was not found; detected {device_serials}."
        )
    camera = Camera(args.camera_serial, RGB_RESOLUTION, DEPTH_RESOLUTION)

    print(f"mapping={mapping.source_path}")
    print(f"port={robot.bus.port}")
    print(f"camera_serial={args.camera_serial}")
    print(f"capture_dir={DEFAULT_ASSETS_DIR}")
    print("mode=read-only raw encoder inspection; Goal_Position is never written")
    if not wait_for_start():
        print("Exited before hardware connection.")
        return

    camera_started = False
    try:
        robot.bus.connect()
        robot.bus.disable_torque(num_retry=5)
        ensure_torque_disabled(robot)
        print("torque_disabled=PASS")
        camera.start()
        camera_started = True
        camera.shoot()
        time.sleep(CAMERA_WARMUP_S)
        print("camera_warmup=PASS")
        inspect_loop(robot, model, mapping, camera, DEFAULT_ASSETS_DIR)
    except (KeyboardInterrupt, EOFError):
        print("\nExit requested; stopping camera, disabling torque, and disconnecting.")
    finally:
        try:
            if camera_started:
                camera.stop()
        finally:
            disconnect_torque_off(robot)

    print("viperx_pose_inspection=EXITED_CLEANLY")


if __name__ == "__main__":
    main()
