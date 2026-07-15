#!/usr/bin/env python3
"""Validate the stage-4 ViperX raw-encoder to URDF-radian mapping.

Default mode is offline and never touches hardware. Set ``live=true`` to open
the LeRobot Dynamixel bus and validate the current raw encoder state. Add
``verify_signs=true`` to interactively check URDF-positive joint directions and
optionally write corrected signs back to the mapping JSON. Add ``gui=true`` to
capture a manually held live pose and show the converted URDF pose in PyBullet.
"""

import json
import sys
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

from viperx_model import DEFAULT_ARM_JOINT_NAMES, load_viperx_model


NON_KINEMATIC_MOTORS = frozenset({"shoulder_shadow", "elbow_shadow", "gripper"})


@dataclass
class ValidateViperXUrdfMappingConfig:
    """CLI config for validating the adapter-layer ViperX URDF mapping."""

    mapping: Path = DEFAULT_MAPPING_PATH
    urdf: Path | None = None
    base_link: str | None = None
    end_link: str | None = None
    live: bool = False
    gui: bool = False
    verify_signs: bool = False
    disable_torque: bool = True
    enable_torque_on_exit: bool = False
    robot: RobotConfig | None = None
    yes: bool = False


def load_mapping(path: Path) -> dict[str, Any]:
    """Load and type-check the adapter-layer mapping JSON."""

    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict):
        raise ValueError("Mapping file must contain a JSON object.")
    return mapping


def save_mapping(mapping: Mapping[str, Any], path: Path) -> None:
    """Write an updated mapping JSON after interactive sign correction."""

    path = path.expanduser().resolve()
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    print(f"updated_mapping={path}")


def require_mapping_keys(mapping: Mapping[str, Any]) -> None:
    """Ensure the mapping file has the fields required by this validator."""

    required = {
        "schema_version",
        "mapping_mode",
        "urdf_path",
        "base_link",
        "end_link",
        "joint_order",
        "raw_home",
        "q_home_urdf",
        "sign",
        "scale_rad_per_tick",
        "safe_raw_range",
        "urdf_limit",
    }
    missing = required - set(mapping)
    if missing:
        raise ValueError(f"Mapping file missing keys: {sorted(missing)}")
    if mapping["mapping_mode"] != "home_anchor":
        raise ValueError(f"Unsupported mapping_mode={mapping['mapping_mode']!r}")


def validate_joint_order(mapping: Mapping[str, Any]) -> tuple[str, ...]:
    """Return mapping joint order and require the expected six ViperX arm joints."""

    joint_order = tuple(str(name) for name in mapping["joint_order"])
    expected = tuple(DEFAULT_ARM_JOINT_NAMES)
    if joint_order != expected:
        raise ValueError(f"Unexpected joint_order={joint_order}; expected={expected}.")
    return joint_order


def ordered_array(
    mapping: Mapping[str, Any],
    key: str,
    joint_order: Sequence[str],
) -> np.ndarray:
    """Read one named mapping section as a finite array in joint order."""

    values = mapping[key]
    if not isinstance(values, Mapping):
        raise ValueError(f"{key} must be a mapping.")
    missing = set(joint_order) - set(values)
    extra = set(values) - set(joint_order)
    if missing or extra:
        raise ValueError(f"{key} keys mismatch: missing={missing}, extra={extra}")
    array = np.asarray([values[joint] for joint in joint_order], dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key} contains NaN or infinity.")
    return array


def raw_to_q_urdf(
    raw: Mapping[str, int | float],
    mapping: Mapping[str, Any],
    joint_order: Sequence[str],
) -> np.ndarray:
    """Apply q_urdf = q_home + sign * (raw - raw_home) * scale."""

    raw_array = np.asarray([raw[joint] for joint in joint_order], dtype=np.float64)
    raw_home = ordered_array(mapping, "raw_home", joint_order)
    q_home = ordered_array(mapping, "q_home_urdf", joint_order)
    signs = ordered_array(mapping, "sign", joint_order)
    scales = ordered_array(mapping, "scale_rad_per_tick", joint_order)

    if np.any(scales <= 0.0):
        raise ValueError(f"scale_rad_per_tick must be positive: {scales}")
    if np.any(~np.isin(signs, [-1.0, 1.0])):
        raise ValueError(f"sign values must be +1 or -1: {signs}")
    return q_home + signs * (raw_array - raw_home) * scales


def validate_safe_raw_range(
    raw: Mapping[str, int | float],
    mapping: Mapping[str, Any],
    joint_order: Sequence[str],
) -> None:
    """Check raw encoder values against the saved safe raw ranges."""

    ranges = mapping["safe_raw_range"]
    violations = {}
    for joint in joint_order:
        lower = float(ranges[joint]["min"])
        upper = float(ranges[joint]["max"])
        value = float(raw[joint])
        if value < lower or value > upper:
            violations[joint] = {"raw": value, "min": lower, "max": upper}
    if violations:
        raise ValueError(f"Raw encoder values violate safe ranges: {violations}")


def validate_urdf_limits_from_mapping(
    q: np.ndarray,
    mapping: Mapping[str, Any],
    joint_order: Sequence[str],
) -> None:
    """Check converted URDF radians against the saved URDF joint limits."""

    limits = mapping["urdf_limit"]
    violations = {}
    for index, joint in enumerate(joint_order):
        lower = float(limits[joint]["lower"])
        upper = float(limits[joint]["upper"])
        value = float(q[index])
        if value < lower or value > upper:
            violations[joint] = {"q": value, "lower": lower, "upper": upper}
    if violations:
        raise ValueError(f"q_urdf violates mapped URDF limits: {violations}")


def resolve_model_config(
    cfg: ValidateViperXUrdfMappingConfig,
    mapping: Mapping[str, Any],
) -> tuple[Path, str, str]:
    """Choose URDF/base/end links from CLI overrides or the mapping file."""

    urdf = cfg.urdf if cfg.urdf else Path(str(mapping["urdf_path"]))
    base_link = cfg.base_link if cfg.base_link else str(mapping["base_link"])
    end_link = cfg.end_link if cfg.end_link else str(mapping["end_link"])
    return urdf, base_link, end_link


def print_pose(label: str, q: np.ndarray, transform: np.ndarray) -> None:
    """Print q_urdf and the FK pose in a readable terminal format."""

    xyz = transform[:3, 3]
    print(f"{label}_q_urdf=({','.join(f'{value:.6f}' for value in q)})")
    print(f"{label}_ee_xyz=({xyz[0]:.6f},{xyz[1]:.6f},{xyz[2]:.6f})")
    print(f"{label}_T_base_ee=")
    print(np.array2string(transform, precision=6, suppress_small=False))


def require_live_config(cfg: ValidateViperXUrdfMappingConfig) -> None:
    """Validate live-mode config and require an explicit read-only confirmation."""

    if cfg.gui and not cfg.live:
        raise ValueError("gui=true requires live=true.")
    if cfg.verify_signs and not cfg.live:
        raise ValueError("verify_signs=true requires live=true.")
    if not cfg.live:
        return
    if cfg.robot is None:
        raise ValueError(
            "live=true requires a robot config, e.g. "
            "--robot.type=viperx --robot.port=/dev/ttyUSB0 --robot.id=..."
        )
    if not isinstance(cfg.robot, ViperXConfig):
        raise TypeError(f"Expected --robot.type=viperx, got {cfg.robot.type!r}.")
    if cfg.yes:
        return
    print("Live validation will open the LeRobot Dynamixel bus and read raw encoders.")
    print("It will not send Goal_Position and will not edit LeRobot calibration.")
    if cfg.gui:
        print("gui=true will wait for you to hold a live pose before reading raw encoders.")
        if cfg.disable_torque:
            print("It will disable torque so you can move and hold the arm by hand.")
    if cfg.verify_signs:
        print("verify_signs=true may update the adapter mapping JSON after confirmation.")
        if cfg.disable_torque:
            print("It will disable torque so you can move joints by hand.")
    print("It disconnects the bus with disable_torque=False.")
    answer = input("Type READONLY to continue: ").strip()
    if answer != "READONLY":
        raise SystemExit("Aborted by user.")


def make_lerobot_viperx(cfg: ValidateViperXUrdfMappingConfig) -> Any:
    """Build the LeRobot ViperX instance for live validation."""

    if not isinstance(cfg.robot, ViperXConfig):
        raise TypeError("Live validation requires a ViperXConfig.")
    return make_robot_from_config(cfg.robot)


def discover_arm_joint_names(robot: Any) -> tuple[str, ...]:
    """Return the six kinematic arm joints from the LeRobot motor list."""

    motor_names = tuple(robot.bus.motors)
    arm_joint_names = tuple(
        name for name in motor_names if name not in NON_KINEMATIC_MOTORS
    )
    expected = tuple(DEFAULT_ARM_JOINT_NAMES)
    if arm_joint_names != expected:
        raise ValueError(
            f"Unexpected LeRobot arm joints {arm_joint_names}; expected {expected}."
        )
    return arm_joint_names


def read_raw_joints(robot: Any, joint_names: Sequence[str]) -> dict[str, int]:
    """Read raw Dynamixel Present_Position ticks for the requested joints."""

    values = robot.bus.sync_read(
        "Present_Position",
        list(joint_names),
        normalize=False,
    )
    return {joint: int(values[joint]) for joint in joint_names}


def validate_live_scales(
    robot: Any,
    mapping: Mapping[str, Any],
    joint_order: Sequence[str],
    *,
    atol: float = 1e-12,
) -> None:
    """Check that live motor resolutions match mapping scale_rad_per_tick."""

    mapped_scales = mapping["scale_rad_per_tick"]
    for joint in joint_order:
        motor = robot.bus.motors[joint]
        resolution = int(robot.bus.model_resolution_table[motor.model])
        live_scale = float(2.0 * np.pi / resolution)
        mapped_scale = float(mapped_scales[joint])
        if not np.isclose(live_scale, mapped_scale, atol=atol):
            raise ValueError(
                f"{joint} scale mismatch: mapping={mapped_scale}, live={live_scale}."
            )


def should_disable_torque_for_live(cfg: ValidateViperXUrdfMappingConfig) -> bool:
    """Return whether live validation should disable torque for manual movement."""

    return cfg.disable_torque and (cfg.gui or cfg.verify_signs)


def wait_for_manual_live_pose() -> None:
    """Pause until the user is holding the real arm at the validation pose."""

    print("\nManual live-pose validation:")
    print("Support the ViperX before continuing.")
    print("Move the real arm to the pose you want to compare with PyBullet.")
    print("Keep holding the arm steady; raw encoders are read after ENTER.")
    input("Press ENTER when the real arm is stable: ")


def show_pybullet_pose(
    urdf: Path,
    joint_order: Sequence[str],
    q_urdf: Sequence[float],
) -> None:
    """Open PyBullet GUI and display the converted live q_urdf pose."""

    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise RuntimeError("gui=true requires pybullet in the current environment.") from exc

    urdf = urdf.expanduser().resolve()
    if not urdf.exists():
        raise FileNotFoundError(urdf)

    client_id = p.connect(p.GUI, options="--window_title=ViperX URDF mapping validation")
    if client_id < 0:
        raise RuntimeError("Failed to open PyBullet GUI.")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.1,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=[0.25, 0.0, 0.35],
            physicsClientId=client_id,
        )
        p.loadURDF("plane.urdf", physicsClientId=client_id)
        robot_id = p.loadURDF(
            str(urdf),
            useFixedBase=True,
            physicsClientId=client_id,
        )

        joint_index_by_name: dict[str, int] = {}
        for joint_index in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
            info = p.getJointInfo(robot_id, joint_index, physicsClientId=client_id)
            joint_index_by_name[info[1].decode("utf-8")] = joint_index

        missing = set(joint_order) - set(joint_index_by_name)
        if missing:
            raise ValueError(f"PyBullet URDF missing joints: {sorted(missing)}")

        for joint, value in zip(joint_order, q_urdf, strict=True):
            p.resetJointState(
                robot_id,
                joint_index_by_name[joint],
                float(value),
                physicsClientId=client_id,
            )
        p.stepSimulation(physicsClientId=client_id)

        print("\nPyBullet GUI is showing live raw -> q_urdf.")
        print("Compare the displayed pose with the real ViperX pose you just held.")
        input("Press ENTER to close PyBullet GUI: ")
    finally:
        p.disconnect(physicsClientId=client_id)


def infer_signs_interactively(robot: Any, joint_order: Sequence[str]) -> dict[str, int]:
    """Infer signs by asking the user to move each joint URDF-positive."""

    signs: dict[str, int] = {}
    print("\nInteractive sign verification:")
    print("For each prompted joint, move only that joint in the URDF-positive direction.")
    print("URDF-positive means increasing that joint value in the validated PyBullet/URDF model.")
    for joint in joint_order:
        input(f"\nHold still, then press ENTER to read baseline for {joint}: ")
        before = read_raw_joints(robot, joint_order)
        input(f"Move {joint} slightly in URDF-positive direction, then press ENTER: ")
        after = read_raw_joints(robot, joint_order)
        delta = after[joint] - before[joint]
        if delta == 0:
            raise RuntimeError(
                f"No raw encoder change detected for {joint}. "
                "Move a little more and rerun sign verification."
            )
        sign = 1 if delta > 0 else -1
        signs[joint] = sign
        print(
            f"{joint}: raw_before={before[joint]} raw_after={after[joint]} "
            f"delta={delta} inferred_sign={sign:+d}"
        )
    return signs


def verify_and_fix_signs(
    mapping: dict[str, Any],
    inferred_signs: Mapping[str, int],
    joint_order: Sequence[str],
    mapping_path: Path,
) -> list[str]:
    """Compare inferred signs with mapping signs and optionally write fixes."""

    corrected: list[str] = []
    mapped_signs = mapping["sign"]
    for joint in joint_order:
        current = int(mapped_signs[joint])
        inferred = int(inferred_signs[joint])
        if current not in (-1, 1):
            raise ValueError(f"Invalid mapped sign for {joint}: {current}")
        if inferred not in (-1, 1):
            raise ValueError(f"Invalid inferred sign for {joint}: {inferred}")
        if current == inferred:
            print(f"{joint}: sign OK ({current:+d})")
            continue

        print(
            f"{joint}: sign mismatch, mapping={current:+d}, "
            f"observed={inferred:+d}."
        )
        answer = input(f"Flip {joint} sign in mapping to {inferred:+d}? [y/N]: ")
        if answer.strip().lower() != "y":
            raise SystemExit("Sign correction declined; mapping was not changed.")
        mapped_signs[joint] = inferred
        corrected.append(joint)

    if not corrected:
        print("sign_verification=PASS")
        return corrected

    mapping["updated_at"] = datetime.now(timezone.utc).isoformat()
    source = mapping.setdefault("source", {})
    if isinstance(source, dict):
        source["sign_validation"] = (
            "validate_viperx_urdf_mapping.py interactive URDF-positive motion"
        )
    save_mapping(mapping, mapping_path)
    print(f"corrected_signs={corrected}")
    return corrected


@draccus.wrap()
def main(cfg: ValidateViperXUrdfMappingConfig) -> None:
    """Run offline checks, then optional live/GUIsign validation."""

    require_live_config(cfg)

    mapping = load_mapping(cfg.mapping)
    require_mapping_keys(mapping)
    joint_order = validate_joint_order(mapping)

    urdf, base_link, end_link = resolve_model_config(cfg, mapping)
    model = load_viperx_model(
        urdf_path=urdf,
        base_link=base_link,
        end_link=end_link,
    )

    raw_home = {
        joint: int(mapping["raw_home"][joint])
        for joint in joint_order
    }
    q_home_from_mapping = raw_to_q_urdf(raw_home, mapping, joint_order)
    q_home_expected = ordered_array(mapping, "q_home_urdf", joint_order)
    if not np.allclose(q_home_from_mapping, q_home_expected, atol=1e-12):
        raise ValueError("raw_home does not map back to q_home_urdf.")

    model.validate_joints(q_home_expected)
    validate_safe_raw_range(raw_home, mapping, joint_order)
    validate_urdf_limits_from_mapping(q_home_expected, mapping, joint_order)
    print_pose("home", q_home_expected, model.fk(q_home_expected))

    if cfg.live:
        robot = make_lerobot_viperx(cfg)
        live_joint_order = discover_arm_joint_names(robot)
        if live_joint_order != joint_order:
            raise ValueError(
                f"Live joint order {live_joint_order} != mapping {joint_order}."
            )
        robot.bus.connect()
        try:
            if should_disable_torque_for_live(cfg):
                print("Disabling torque for manual validation...")
                robot.bus.disable_torque()
            validate_live_scales(robot, mapping, joint_order)
            if cfg.verify_signs:
                inferred_signs = infer_signs_interactively(robot, joint_order)
                verify_and_fix_signs(mapping, inferred_signs, joint_order, cfg.mapping)
            if cfg.gui:
                wait_for_manual_live_pose()
            raw_live = read_raw_joints(robot, joint_order)
        finally:
            if robot.bus.is_connected:
                if (cfg.verify_signs or cfg.gui) and cfg.enable_torque_on_exit:
                    print("Re-enabling torque before disconnect...")
                    robot.bus.enable_torque()
                robot.bus.disconnect(disable_torque=False)

        validate_safe_raw_range(raw_live, mapping, joint_order)
        q_live = raw_to_q_urdf(raw_live, mapping, joint_order)
        model.validate_joints(q_live)
        validate_urdf_limits_from_mapping(q_live, mapping, joint_order)
        print(f"live_raw={raw_live}")
        print_pose("live", q_live, model.fk(q_live))
        if cfg.gui:
            show_pybullet_pose(urdf, joint_order, q_live)

    print("validate_viperx_urdf_mapping=PASS")


if __name__ == "__main__":
    main()
