#!/usr/bin/env python3
"""Create the stage-4 ViperX raw-encoder to URDF-radian mapping.

Interactive flow:
1. Parse a LeRobot-style dataclass config with ``draccus``.
2. Open the validated full ViperX URDF in PyBullet GUI.
3. Open the LeRobot Dynamixel bus directly and disable torque by default.
4. Let the user match the PyBullet home pose and the real arm pose.
5. Read ``q_home_urdf`` from PyBullet sliders and ``raw_home`` from raw encoders.
6. Infer per-joint signs by manual URDF-positive joint motion.
7. Save an adapter-layer mapping JSON.

Safety boundary:
- sends no ``Goal_Position``;
- does not rewrite Dynamixel ``Homing_Offset`` or ``Drive_Mode``;
- does not edit LeRobot calibration files;
- disables torque by default so the arm can be moved by hand.
"""

from __future__ import annotations

import json
import select
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.dont_write_bytecode = True

REAL_DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING_PATH = (
    REAL_DEPLOYMENT_DIR / "configs" / "viperx_urdf_mapping.json"
)

if str(REAL_DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(REAL_DEPLOYMENT_DIR))

import draccus
from lerobot.robots.config import RobotConfig
from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.viperx.config_viperx import ViperXConfig

from viperx_model import DEFAULT_ARM_JOINT_NAMES, DEFAULT_URDF_PATH, load_viperx_model


NON_KINEMATIC_MOTORS = frozenset({"shoulder_shadow", "elbow_shadow", "gripper"})


@dataclass
class CreateViperXUrdfMappingConfig:
    """CLI config for creating the adapter-layer ViperX URDF mapping."""

    robot: RobotConfig
    urdf: Path = DEFAULT_URDF_PATH
    base_link: str = "vx300s/base_link"
    end_link: str = "vx300s/ee_gripper_link"
    output: Path = DEFAULT_MAPPING_PATH
    overwrite: bool = False
    disable_torque: bool = True
    enable_torque_on_exit: bool = False
    signs: str | None = None
    home_source: str = "manual PyBullet GUI home-anchor"
    note: str = ""
    pybullet_width: int = 1280
    pybullet_height: int = 800
    yes: bool = False


@dataclass
class PyBulletHomeSelector:
    pybullet: Any
    client_id: int
    robot_id: int
    sliders: dict[str, tuple[int, int]]


def parse_six_float_list(value: str, *, field_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be comma-separated floats.") from exc
    if len(values) != len(DEFAULT_ARM_JOINT_NAMES):
        raise ValueError(
            f"{field_name} must have {len(DEFAULT_ARM_JOINT_NAMES)} values."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} contains NaN or infinity.")
    return values


def parse_six_sign_list(value: str) -> tuple[int, ...]:
    values = parse_six_float_list(value, field_name="signs")
    signs = tuple(int(value) for value in values)
    if any(value not in (-1, 1) for value in signs):
        raise ValueError("signs must contain only +1 or -1.")
    return signs


def require_interactive_confirmation(cfg: CreateViperXUrdfMappingConfig) -> None:
    if cfg.yes:
        return
    print("This script will create the adapter-layer ViperX URDF mapping.")
    print("It will open PyBullet GUI and the LeRobot Dynamixel bus.")
    if cfg.disable_torque:
        print("It will disable torque so the real arm can be moved by hand.")
    print("It will not send Goal_Position.")
    print("It will not edit Homing_Offset, Drive_Mode, or LeRobot calibration.")
    answer = input("Type MAPPING to continue: ").strip()
    if answer != "MAPPING":
        raise SystemExit("Aborted by user.")


def make_lerobot_viperx(cfg: CreateViperXUrdfMappingConfig) -> Any:
    if not isinstance(cfg.robot, ViperXConfig):
        raise TypeError(
            f"Expected --robot.type=viperx, got {cfg.robot.type!r}."
        )
    return make_robot_from_config(cfg.robot)


def discover_arm_joint_names(robot: Any) -> tuple[str, ...]:
    motor_names = tuple(robot.bus.motors)
    arm_joint_names = tuple(
        name for name in motor_names if name not in NON_KINEMATIC_MOTORS
    )
    expected = tuple(DEFAULT_ARM_JOINT_NAMES)
    if arm_joint_names != expected:
        raise ValueError(
            f"Unexpected arm joint order {arm_joint_names}; expected {expected}."
        )
    return arm_joint_names


def calibration_value(calibration: Any, field_name: str) -> Any:
    if isinstance(calibration, Mapping):
        return calibration[field_name]
    return getattr(calibration, field_name)


def read_calibration_ranges(
    robot: Any,
    joint_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    calibration = getattr(robot, "calibration", None)
    if not calibration:
        raise RuntimeError(
            "LeRobot calibration was not loaded. Check --robot.id and "
            f"--robot.calibration_dir. Expected file: {robot.calibration_fpath}"
        )

    ranges: dict[str, dict[str, int]] = {}
    for joint in joint_names:
        if joint not in calibration:
            raise KeyError(f"LeRobot calibration is missing {joint!r}.")
        item = calibration[joint]
        ranges[joint] = {
            "min": int(calibration_value(item, "range_min")),
            "max": int(calibration_value(item, "range_max")),
        }
    return ranges


def read_encoder_scales(
    robot: Any,
    joint_names: Sequence[str],
) -> tuple[dict[str, int], dict[str, float]]:
    resolutions: dict[str, int] = {}
    scales: dict[str, float] = {}
    for joint in joint_names:
        motor = robot.bus.motors[joint]
        resolution = int(robot.bus.model_resolution_table[motor.model])
        if resolution <= 0:
            raise ValueError(f"Invalid encoder resolution for {joint}: {resolution}")
        resolutions[joint] = resolution
        scales[joint] = float(2.0 * np.pi / resolution)
    return resolutions, scales


class RawEncoderSession:
    """Direct raw-encoder bus session for mapping."""

    def __init__(
        self,
        robot: Any,
        *,
        disable_torque: bool,
        enable_torque_on_exit: bool,
    ) -> None:
        self.robot = robot
        self.disable_torque = disable_torque
        self.enable_torque_on_exit = enable_torque_on_exit

    def __enter__(self) -> "RawEncoderSession":
        self.robot.bus.connect()
        if self.disable_torque:
            print("Disabling torque for manual arm movement...")
            self.robot.bus.disable_torque()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.robot.bus.is_connected:
            try:
                if self.enable_torque_on_exit:
                    print("Re-enabling torque before disconnect...")
                    self.robot.bus.enable_torque()
            finally:
                self.robot.bus.disconnect(disable_torque=False)

    def read_raw_joints(self, joint_names: Sequence[str]) -> dict[str, int]:
        values = self.robot.bus.sync_read(
            "Present_Position",
            list(joint_names),
            normalize=False,
        )
        return {joint: int(values[joint]) for joint in joint_names}


def validate_raw_ranges(
    raw: Mapping[str, int],
    safe_raw_range: Mapping[str, Mapping[str, int]],
) -> None:
    violations = {}
    for joint, value in raw.items():
        lower = int(safe_raw_range[joint]["min"])
        upper = int(safe_raw_range[joint]["max"])
        if value < lower or value > upper:
            violations[joint] = {"raw": value, "min": lower, "max": upper}
    if violations:
        raise ValueError(f"raw_home violates LeRobot calibration ranges: {violations}")


def open_pybullet_home_selector(
    cfg: CreateViperXUrdfMappingConfig,
    joint_names: Sequence[str],
) -> PyBulletHomeSelector:
    import pybullet as p
    import pybullet_data

    urdf = cfg.urdf.expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(f"Missing URDF: {urdf}")

    options = f"--width={cfg.pybullet_width} --height={cfg.pybullet_height}"
    client_id = p.connect(p.GUI, options=options)
    if client_id < 0:
        raise RuntimeError("PyBullet GUI connection failed.")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1,
        cameraYaw=135,
        cameraPitch=-25,
        cameraTargetPosition=[0.2, 0.0, 0.25],
    )
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(
        str(urdf),
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
    )

    joint_info: dict[str, dict[str, float | int]] = {}
    for joint_index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, joint_index)
        name = info[1].decode("utf-8")
        joint_info[name] = {
            "index": joint_index,
            "type": info[2],
            "lower": info[8],
            "upper": info[9],
        }

    sliders: dict[str, tuple[int, int]] = {}
    for name in joint_names:
        if name not in joint_info:
            raise ValueError(f"URDF is missing arm joint {name!r}.")
        info = joint_info[name]
        if info["type"] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            raise ValueError(f"URDF joint {name!r} is not movable.")
        lower = float(info["lower"])
        upper = float(info["upper"])
        if lower >= upper:
            lower, upper = -np.pi, np.pi
        slider = p.addUserDebugParameter(name, lower, upper, 0.0)
        sliders[name] = (int(info["index"]), slider)

    print(f"Opened PyBullet GUI with URDF: {urdf}")
    print("Move the six PyBullet sliders to the chosen URDF home pose.")
    return PyBulletHomeSelector(
        pybullet=p,
        client_id=client_id,
        robot_id=robot_id,
        sliders=sliders,
    )


def close_pybullet_home_selector(selector: PyBulletHomeSelector | None) -> None:
    if selector is None:
        return
    selector.pybullet.disconnect(selector.client_id)


def read_pybullet_q_home(
    selector: PyBulletHomeSelector,
    joint_names: Sequence[str],
) -> tuple[float, ...]:
    q_values: list[float] = []
    for joint in joint_names:
        joint_index, slider = selector.sliders[joint]
        value = float(selector.pybullet.readUserDebugParameter(slider))
        selector.pybullet.resetJointState(selector.robot_id, joint_index, value)
        q_values.append(value)
    selector.pybullet.stepSimulation()
    return tuple(q_values)


def wait_for_home_anchor(
    selector: PyBulletHomeSelector,
    joint_names: Sequence[str],
) -> tuple[float, ...]:
    print("\nHome-anchor selection:")
    print("1. Move the PyBullet sliders to the URDF home pose.")
    print("2. Move the real arm to the matching physical pose.")
    print("3. Return to this terminal and press ENTER to record both poses.")
    print("Waiting for ENTER while keeping the PyBullet robot updated...")

    while True:
        q_home = read_pybullet_q_home(selector, joint_names)
        readable, _, _ = select.select([sys.stdin], [], [], 1.0 / 60.0)
        if readable:
            sys.stdin.readline()
            q_home = read_pybullet_q_home(selector, joint_names)
            print(
                "q_home_urdf="
                + ", ".join(f"{joint}={value:.9f}" for joint, value in zip(joint_names, q_home))
            )
            return q_home
        time.sleep(1.0 / 240.0)


def infer_signs_interactively(
    session: RawEncoderSession,
    joint_names: Sequence[str],
) -> dict[str, int]:
    signs: dict[str, int] = {}
    print("\nSign inference:")
    print("Move only the prompted joint in the URDF-positive direction.")
    print("In PyBullet, increasing the corresponding slider value is URDF-positive.")
    for joint in joint_names:
        input(f"\nHold still, then press ENTER to read baseline for {joint}: ")
        before = session.read_raw_joints(joint_names)
        input(f"Move {joint} slightly in URDF-positive direction, then press ENTER: ")
        after = session.read_raw_joints(joint_names)
        delta = after[joint] - before[joint]
        if delta == 0:
            raise RuntimeError(
                f"No raw encoder change detected for {joint}. "
                "Move a little more and rerun sign inference."
            )
        sign = 1 if delta > 0 else -1
        print(
            f"{joint}: raw_before={before[joint]} raw_after={after[joint]} "
            f"delta={delta} => sign={sign:+d}"
        )
        answer = input("Accept this sign? [y/N]: ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted by user.")
        signs[joint] = sign
    return signs


def named_float_dict(
    joint_names: Sequence[str],
    values: Sequence[float],
) -> dict[str, float]:
    return {joint: float(value) for joint, value in zip(joint_names, values)}


def named_int_dict(
    joint_names: Sequence[str],
    values: Sequence[int],
) -> dict[str, int]:
    return {joint: int(value) for joint, value in zip(joint_names, values)}


def build_mapping(
    *,
    cfg: CreateViperXUrdfMappingConfig,
    model: Any,
    robot: Any,
    joint_names: tuple[str, ...],
    raw_home: dict[str, int],
    q_home_urdf: tuple[float, ...],
    signs: dict[str, int],
    resolutions: dict[str, int],
    scales: dict[str, float],
    safe_raw_range: dict[str, dict[str, int]],
) -> dict[str, Any]:
    lower, upper = np.asarray(model.qlim, dtype=np.float64)
    urdf_limit = {
        joint: {"lower": float(lo), "upper": float(hi)}
        for joint, lo, hi in zip(joint_names, lower, upper)
    }

    robot_config = cfg.robot
    if not isinstance(robot_config, ViperXConfig):
        raise TypeError("Mapping can only be built for ViperXConfig.")

    return {
        "schema_version": 1,
        "mapping_mode": "home_anchor",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "urdf_path": str(cfg.urdf.expanduser().resolve()),
        "base_link": model.base_link,
        "end_link": model.end_link,
        "joint_order": list(joint_names),
        "raw_home": raw_home,
        "q_home_urdf": named_float_dict(joint_names, q_home_urdf),
        "sign": signs,
        "encoder_resolution": resolutions,
        "scale_rad_per_tick": scales,
        "safe_raw_range": safe_raw_range,
        "urdf_limit": urdf_limit,
        "lerobot": {
            "robot_id": robot_config.id,
            "port": robot_config.port,
            "calibration_dir": str(robot_config.calibration_dir),
            "calibration_file": str(robot.calibration_fpath),
        },
        "source": {
            "home": cfg.home_source,
            "sign": "manual per-joint URDF-positive movement",
            "note": cfg.note,
        },
        "safety": {
            "disables_torque": cfg.disable_torque,
            "reenables_torque_on_exit": cfg.enable_torque_on_exit,
            "writes_goal_position": False,
            "writes_dynamixel_homing_offset": False,
            "writes_dynamixel_drive_mode": False,
            "writes_lerobot_calibration": False,
        },
    }


def save_mapping(mapping: Mapping[str, Any], path: Path, *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite=true to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    print(f"saved_mapping={path}")


@draccus.wrap()
def main(cfg: CreateViperXUrdfMappingConfig) -> None:
    require_interactive_confirmation(cfg)

    model = load_viperx_model(
        urdf_path=cfg.urdf,
        base_link=cfg.base_link,
        end_link=cfg.end_link,
    )
    robot = make_lerobot_viperx(cfg)
    joint_names = discover_arm_joint_names(robot)
    safe_raw_range = read_calibration_ranges(robot, joint_names)
    resolutions, scales = read_encoder_scales(robot, joint_names)

    selector: PyBulletHomeSelector | None = None
    try:
        selector = open_pybullet_home_selector(cfg, joint_names)
        with RawEncoderSession(
            robot,
            disable_torque=cfg.disable_torque,
            enable_torque_on_exit=cfg.enable_torque_on_exit,
        ) as session:
            q_home_urdf = wait_for_home_anchor(selector, joint_names)
            model.validate_joints(q_home_urdf)

            raw_home = session.read_raw_joints(joint_names)
            validate_raw_ranges(raw_home, safe_raw_range)
            print(f"raw_home={raw_home}")

            if cfg.signs:
                sign_values = parse_six_sign_list(cfg.signs)
                signs = named_int_dict(joint_names, sign_values)
            else:
                signs = infer_signs_interactively(session, joint_names)
    finally:
        close_pybullet_home_selector(selector)

    mapping = build_mapping(
        cfg=cfg,
        model=model,
        robot=robot,
        joint_names=joint_names,
        raw_home=raw_home,
        q_home_urdf=q_home_urdf,
        signs=signs,
        resolutions=resolutions,
        scales=scales,
        safe_raw_range=safe_raw_range,
    )
    save_mapping(mapping, cfg.output, overwrite=cfg.overwrite)


if __name__ == "__main__":
    main()
