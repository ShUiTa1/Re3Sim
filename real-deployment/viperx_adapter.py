"""Re3Sim-facing adapter for a LeRobot-controlled ViperX300s.

LeRobot owns the hardware connection and raw Dynamixel IO. ``ViperXModel``
owns FK/IK. The accepted stage-4 mapping is the only boundary between raw
encoder ticks and URDF joint radians.
"""

# from __future__ import annotations

import json
import select
import signal
import sys
import threading
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
# make noting clear

ARM_JOINT_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
SHADOW_TO_JOINT = {
    "shoulder_shadow": "shoulder",
    "elbow_shadow": "elbow",
}
NON_ARM_MOTORS = frozenset({"shoulder_shadow", "elbow_shadow", "gripper"})
RANGE_M100_100 = "range_m100_100"
NORMALIZED_POSITION_SPAN = 200.0
EXPECTED_MAX_RELATIVE_TARGET = 5.0
TORQUE_ON_BY_GOAL_UPDATE_BIT = 0b1000
STARTUP_STAGING_Q_RAD = (
    0.018408,
    -1.293146,
    1.201107,
    0.012272,
    0.493942,
    0.006136,
)


class ViperXMotionInterrupted(RuntimeError):
    """Raised after a requested or fault-triggered motion halt."""


def _calibration_value(calibration: Any, field: str) -> Any:
    if isinstance(calibration, Mapping):
        return calibration[field]
    return getattr(calibration, field)


@dataclass(frozen=True)
class ViperXUrdfMapping:
    """Stage-4 home-anchor mapping in the fixed six-joint order."""

    source_path: Path
    joint_order: tuple[str, ...]
    robot_id: str
    base_link: str
    end_link: str
    encoder_resolution: IntArray
    raw_home: IntArray
    q_home_urdf: FloatArray
    sign: IntArray
    scale_rad_per_tick: FloatArray
    raw_min: IntArray
    raw_max: IntArray
    q_lower: FloatArray
    q_upper: FloatArray
    shadow_names: tuple[str, ...]
    shadow_joint_indices: IntArray
    shadow_encoder_resolution: IntArray
    shadow_raw_home: IntArray
    shadow_sign: IntArray
    shadow_scale_rad_per_tick: FloatArray
    shadow_raw_min: IntArray
    shadow_raw_max: IntArray

    @classmethod
    def load(cls, path: str | Path) -> "ViperXUrdfMapping":
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        required = {
            "schema_version",
            "mapping_mode",
            "joint_order",
            "lerobot",
            "base_link",
            "end_link",
            "encoder_resolution",
            "raw_home",
            "q_home_urdf",
            "sign",
            "scale_rad_per_tick",
            "safe_raw_range",
            "shadow_mapping",
            "urdf_limit",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"Mapping is missing keys: {sorted(missing)}")
        if data["schema_version"] != 1 or data["mapping_mode"] != "home_anchor":
            raise ValueError("Expected schema_version=1 and mapping_mode='home_anchor'.")

        joint_order = tuple(data["joint_order"])
        if joint_order != ARM_JOINT_NAMES:
            raise ValueError(
                f"Unexpected mapping joint order {joint_order}; expected {ARM_JOINT_NAMES}."
            )

        def ordered(section: str, dtype: Any) -> NDArray[Any]:
            values = data[section]
            if set(values) != set(joint_order):
                raise ValueError(f"{section} keys must match the six arm joints.")
            array = np.asarray([values[name] for name in joint_order], dtype=dtype)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{section} contains NaN or infinity.")
            return array

        resolution = ordered("encoder_resolution", np.int64)
        raw_home = ordered("raw_home", np.int64)
        q_home = ordered("q_home_urdf", np.float64)
        sign = ordered("sign", np.int64)
        scale = ordered("scale_rad_per_tick", np.float64)

        raw_ranges = data["safe_raw_range"]
        urdf_limits = data["urdf_limit"]
        if set(raw_ranges) != set(joint_order) or set(urdf_limits) != set(joint_order):
            raise ValueError("safe_raw_range and urdf_limit must match the arm joints.")
        raw_min = np.asarray([raw_ranges[name]["min"] for name in joint_order], dtype=np.int64)
        raw_max = np.asarray([raw_ranges[name]["max"] for name in joint_order], dtype=np.int64)
        q_lower = np.asarray([urdf_limits[name]["lower"] for name in joint_order], dtype=np.float64)
        q_upper = np.asarray([urdf_limits[name]["upper"] for name in joint_order], dtype=np.float64)

        shadow_names = tuple(SHADOW_TO_JOINT)
        shadow_data = data["shadow_mapping"]
        if set(shadow_data) != set(shadow_names):
            raise ValueError(f"shadow_mapping keys must be {list(shadow_names)}.")
        shadow_required = {
            "joint",
            "raw_home",
            "sign",
            "encoder_resolution",
            "scale_rad_per_tick",
            "safe_raw_range",
        }
        for shadow, joint in SHADOW_TO_JOINT.items():
            if set(shadow_data[shadow]) != shadow_required:
                raise ValueError(f"{shadow} mapping keys are invalid.")
            if shadow_data[shadow]["joint"] != joint:
                raise ValueError(f"{shadow} must map to {joint}.")

        shadow_joint_indices = np.asarray(
            [joint_order.index(SHADOW_TO_JOINT[name]) for name in shadow_names], dtype=np.int64
        )
        shadow_resolution = np.asarray(
            [shadow_data[name]["encoder_resolution"] for name in shadow_names], dtype=np.int64
        )
        shadow_raw_home = np.asarray(
            [shadow_data[name]["raw_home"] for name in shadow_names], dtype=np.int64
        )
        shadow_sign = np.asarray(
            [shadow_data[name]["sign"] for name in shadow_names], dtype=np.int64
        )
        shadow_scale = np.asarray(
            [shadow_data[name]["scale_rad_per_tick"] for name in shadow_names], dtype=np.float64
        )
        shadow_raw_min = np.asarray(
            [shadow_data[name]["safe_raw_range"]["min"] for name in shadow_names], dtype=np.int64
        )
        shadow_raw_max = np.asarray(
            [shadow_data[name]["safe_raw_range"]["max"] for name in shadow_names], dtype=np.int64
        )

        if np.any(resolution <= 0):
            raise ValueError("Encoder resolutions must be positive.")
        if np.any(~np.isin(sign, (-1, 1))):
            raise ValueError("Mapping signs must be +1 or -1.")
        if np.any(scale <= 0.0) or not np.allclose(
            scale, 2.0 * np.pi / resolution, rtol=1e-9, atol=1e-12
        ):
            raise ValueError("Mapping scales must equal 2*pi/encoder_resolution.")
        if np.any(raw_min >= raw_max) or np.any(raw_home < raw_min) or np.any(raw_home > raw_max):
            raise ValueError("Mapping raw home/ranges are invalid.")
        if np.any(q_lower >= q_upper) or np.any(q_home < q_lower) or np.any(q_home > q_upper):
            raise ValueError("Mapping URDF home/limits are invalid.")
        if np.any(shadow_resolution <= 0):
            raise ValueError("Shadow encoder resolutions must be positive.")
        if np.any(~np.isin(shadow_sign, (-1, 1))):
            raise ValueError("Shadow mapping signs must be +1 or -1.")
        if np.any(shadow_scale <= 0.0) or not np.allclose(
            shadow_scale, 2.0 * np.pi / shadow_resolution, rtol=1e-9, atol=1e-12
        ):
            raise ValueError("Shadow mapping scales must equal 2*pi/encoder_resolution.")
        if (
            np.any(shadow_raw_min >= shadow_raw_max)
            or np.any(shadow_raw_home < shadow_raw_min)
            or np.any(shadow_raw_home > shadow_raw_max)
        ):
            raise ValueError("Shadow mapping raw home/ranges are invalid.")

        robot_id = str(data["lerobot"].get("robot_id", ""))
        base_link = str(data["base_link"])
        end_link = str(data["end_link"])
        if not robot_id or not base_link or not end_link or base_link == end_link:
            raise ValueError("Mapping robot_id/base_link/end_link are invalid.")

        return cls(
            source_path=path,
            joint_order=joint_order,
            robot_id=robot_id,
            base_link=base_link,
            end_link=end_link,
            encoder_resolution=resolution,
            raw_home=raw_home,
            q_home_urdf=q_home,
            sign=sign,
            scale_rad_per_tick=scale,
            raw_min=raw_min,
            raw_max=raw_max,
            q_lower=q_lower,
            q_upper=q_upper,
            shadow_names=shadow_names,
            shadow_joint_indices=shadow_joint_indices,
            shadow_encoder_resolution=shadow_resolution,
            shadow_raw_home=shadow_raw_home,
            shadow_sign=shadow_sign,
            shadow_scale_rad_per_tick=shadow_scale,
            shadow_raw_min=shadow_raw_min,
            shadow_raw_max=shadow_raw_max,
        )

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return self.joint_order + self.shadow_names

    def validate_raw(self, raw_positions: Mapping[str, Any]) -> dict[str, int]:
        if set(raw_positions) != set(self.joint_order):
            raise ValueError("Raw position keys must match the six arm joints.")
        values = np.asarray([raw_positions[name] for name in self.joint_order], dtype=np.float64)
        if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
            raise ValueError("Raw positions must be finite integer encoder ticks.")
        values = np.rint(values).astype(np.int64)
        invalid = (values < self.raw_min) | (values > self.raw_max)
        if np.any(invalid):
            details = {
                name: {
                    "raw": int(values[index]),
                    "min": int(self.raw_min[index]),
                    "max": int(self.raw_max[index]),
                }
                for index, name in enumerate(self.joint_order)
                if invalid[index]
            }
            raise ValueError(f"Raw values violate mapping safe raw range: {details}")
        return {name: int(values[index]) for index, name in enumerate(self.joint_order)}

    def validate_q(self, q_rad: Sequence[float]) -> FloatArray:
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape != (len(self.joint_order),) or not np.all(np.isfinite(q)):
            raise ValueError("Expected six finite URDF joint radians.")
        if np.any(q < self.q_lower) or np.any(q > self.q_upper):
            raise ValueError("Joint target violates mapping URDF limits.")
        return q

    def validate_actuator_raw(self, raw_positions: Mapping[str, Any]) -> dict[str, int]:
        if set(raw_positions) != set(self.actuator_names):
            raise ValueError("Raw actuator keys must match the six arm joints and two shadow motors.")
        validated = self.validate_raw({name: raw_positions[name] for name in self.joint_order})
        values = np.asarray([raw_positions[name] for name in self.shadow_names], dtype=np.float64)
        if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
            raise ValueError("Shadow raw positions must be finite integer encoder ticks.")
        values = np.rint(values).astype(np.int64)
        invalid = (values < self.shadow_raw_min) | (values > self.shadow_raw_max)
        if np.any(invalid):
            details = {
                name: {
                    "raw": int(values[index]),
                    "min": int(self.shadow_raw_min[index]),
                    "max": int(self.shadow_raw_max[index]),
                }
                for index, name in enumerate(self.shadow_names)
                if invalid[index]
            }
            raise ValueError(f"Shadow raw values violate mapping safe raw range: {details}")
        validated.update(
            {name: int(values[index]) for index, name in enumerate(self.shadow_names)}
        )
        return {name: validated[name] for name in self.actuator_names}

    def raw_to_rad(self, raw_positions: Mapping[str, Any]) -> FloatArray:
        raw = self.validate_raw(raw_positions)
        raw_array = np.asarray([raw[name] for name in self.joint_order], dtype=np.float64)
        return self.q_home_urdf + self.sign * (raw_array - self.raw_home) * self.scale_rad_per_tick

    def rad_to_raw(self, q_rad: Sequence[float]) -> dict[str, int]:
        q = self.validate_q(q_rad)
        raw = np.rint(
            self.raw_home
            + self.sign * (q - self.q_home_urdf) / self.scale_rad_per_tick
        ).astype(np.int64)
        return self.validate_raw(
            {name: int(raw[index]) for index, name in enumerate(self.joint_order)}
        )

    def _rad_to_actuator_raw_unchecked(self, q_rad: Sequence[float]) -> dict[str, int]:
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape != (len(self.joint_order),) or not np.all(np.isfinite(q)):
            raise ValueError("Expected six finite URDF joint radians.")
        raw_values = np.rint(
            self.raw_home
            + self.sign * (q - self.q_home_urdf) / self.scale_rad_per_tick
        ).astype(np.int64)
        raw = {
            name: int(raw_values[index]) for index, name in enumerate(self.joint_order)
        }
        shadow_q = q[self.shadow_joint_indices]
        shadow_q_home = self.q_home_urdf[self.shadow_joint_indices]
        shadow_raw = np.rint(
            self.shadow_raw_home
            + self.shadow_sign
            * (shadow_q - shadow_q_home)
            / self.shadow_scale_rad_per_tick
        ).astype(np.int64)
        raw.update(
            {name: int(shadow_raw[index]) for index, name in enumerate(self.shadow_names)}
        )
        return self.validate_actuator_raw(raw)

    def rad_to_actuator_raw(self, q_rad: Sequence[float]) -> dict[str, int]:
        return self._rad_to_actuator_raw_unchecked(self.validate_q(q_rad))


def load_viperx_urdf_mapping(path: str | Path) -> ViperXUrdfMapping:
    return ViperXUrdfMapping.load(path)


class ViperXAdapter:
    """Adapt LeRobot raw ViperX IO to URDF-radian and FK/IK interfaces."""

    def __init__(
        self,
        robot: Any,
        model: Any,
        mapping: ViperXUrdfMapping | str | Path,
        *,
        command_period_s: float = 0.05,
        motion_timeout_s: float = 10.0,
        position_tolerance_rad: float = 0.03,
        settle_samples: int = 3,
        interactive_safety: bool = True,
    ) -> None:
        self.robot = robot
        self.model = model
        self.mapping = (
            mapping if isinstance(mapping, ViperXUrdfMapping) else ViperXUrdfMapping.load(mapping)
        )
        self.arm_joint_names = self._arm_joint_names()
        self.arm_actuator_names = self.mapping.actuator_names
        self._calibration = self._loaded_calibration()
        self._validate_contract()

        self.command_period_s = float(command_period_s) # frequency of getting robot states and sending commands
        self.motion_timeout_s = float(motion_timeout_s) # maximum time allowed for a motion to complete
        self.position_tolerance_rad = float(position_tolerance_rad) 
        self.settle_samples = int(settle_samples) # number of consecutive samples within tolerance to consider motion settled
        numeric_settings = (
            self.command_period_s,
            self.motion_timeout_s,
            self.position_tolerance_rad,
        )
        if not np.all(np.isfinite(numeric_settings)) or any(value <= 0 for value in numeric_settings):
            raise ValueError("Motion periods, timeout, and tolerance must be finite and positive.")
        if self.settle_samples <= 0:
            raise ValueError("settle_samples must be positive.")

        self.home_q_rad = self._validate_q(self.mapping.q_home_urdf)
        self.max_joint_step_raw, self.max_joint_step_rad = self._motion_step_limits()

        self.interactive_safety = bool(interactive_safety)
        self._session_start_raw: dict[str, int] | None = None
        self._held_raw: dict[str, int] | None = None
        self._motion_state = "disconnected"
        self._stop_requested = threading.Event()
        self._release_requested = threading.Event()
        self._terminal_monitor_stop = threading.Event()
        self._terminal_monitor: threading.Thread | None = None
        self._previous_sigint_handler: Any = None

    def _arm_joint_names(self) -> tuple[str, ...]:
        try:
            names = tuple(name for name in self.robot.bus.motors if name not in NON_ARM_MOTORS)
        except AttributeError as exc:
            raise TypeError("robot must provide bus.motors.") from exc
        if names != self.mapping.joint_order:
            raise ValueError("LeRobot arm joint order does not match the mapping.")
        return names

    def _loaded_calibration(self) -> Mapping[str, Any]:
        calibration = getattr(self.robot, "calibration", None) or getattr(
            self.robot.bus, "calibration", None
        )
        if not calibration or any(name not in calibration for name in self.arm_actuator_names):
            raise RuntimeError("LeRobot arm and shadow calibration is not loaded.")
        return calibration

    def _validate_contract(self) -> None:
        if any(name not in self.robot.bus.motors for name in self.arm_actuator_names):
            raise ValueError("LeRobot motors do not contain all mapped arm actuators.")
        if str(getattr(self.robot, "id", "")) != self.mapping.robot_id:
            raise ValueError("LeRobot robot id does not match the mapping.")
        if tuple(getattr(self.model, "arm_joint_names", ())) != self.mapping.joint_order:
            raise ValueError("ViperXModel joint order does not match the mapping.")
        if getattr(self.model, "base_link", None) != self.mapping.base_link:
            raise ValueError("ViperXModel base_link does not match the mapping.")
        if getattr(self.model, "end_link", None) != self.mapping.end_link:
            raise ValueError("ViperXModel end_link does not match the mapping.")

        model_qlim = np.asarray(getattr(self.model, "qlim", None), dtype=np.float64)
        mapping_qlim = np.vstack((self.mapping.q_lower, self.mapping.q_upper))
        if model_qlim.shape != mapping_qlim.shape or not np.allclose(
            model_qlim, mapping_qlim, rtol=1e-9, atol=1e-9
        ):
            raise ValueError("ViperXModel joint limits do not match the mapping.")

        resolutions = []
        calibration_min = []
        calibration_max = []
        for name in self.arm_joint_names:
            motor = self.robot.bus.motors[name]
            resolutions.append(self.robot.bus.model_resolution_table[motor.model])
            calibration_min.append(_calibration_value(self._calibration[name], "range_min"))
            calibration_max.append(_calibration_value(self._calibration[name], "range_max"))
            norm_mode = getattr(motor.norm_mode, "value", motor.norm_mode)
            if norm_mode != RANGE_M100_100:
                raise ValueError(f"{name} must use LeRobot RANGE_M100_100 normalization.")

        if not np.array_equal(resolutions, self.mapping.encoder_resolution):
            raise ValueError("LeRobot encoder resolutions do not match the mapping.")
        if not np.array_equal(calibration_min, self.mapping.raw_min) or not np.array_equal(
            calibration_max, self.mapping.raw_max
        ):
            raise ValueError("LeRobot calibration ranges do not match the mapping.")

        for index, name in enumerate(self.mapping.shadow_names):
            motor = self.robot.bus.motors[name]
            resolution = self.robot.bus.model_resolution_table[motor.model]
            if resolution != self.mapping.shadow_encoder_resolution[index]:
                raise ValueError(f"LeRobot {name} encoder resolution does not match the mapping.")
            range_min = _calibration_value(self._calibration[name], "range_min")
            range_max = _calibration_value(self._calibration[name], "range_max")
            if (
                range_min != self.mapping.shadow_raw_min[index]
                or range_max != self.mapping.shadow_raw_max[index]
            ):
                raise ValueError(f"LeRobot {name} calibration range does not match the mapping.")
            norm_mode = getattr(motor.norm_mode, "value", motor.norm_mode)
            if norm_mode != RANGE_M100_100:
                raise ValueError(f"{name} must use LeRobot RANGE_M100_100 normalization.")

    def _motion_step_limits(self) -> tuple[IntArray, FloatArray]:
        configured = getattr(self.robot.config, "max_relative_target", None)
        if isinstance(configured, Mapping):
            if any(name not in configured for name in self.arm_actuator_names):
                raise ValueError("max_relative_target is missing arm or shadow actuators.")
            normalized = np.asarray(
                [configured[name] for name in self.arm_joint_names], dtype=np.float64
            )
            shadow_normalized = np.asarray(
                [configured[name] for name in self.mapping.shadow_names], dtype=np.float64
            )
        elif isinstance(configured, Real) and not isinstance(configured, bool):
            normalized = np.full(len(self.arm_joint_names), float(configured))
            shadow_normalized = np.full(len(self.mapping.shadow_names), float(configured))
        else:
            raise ValueError("max_relative_target must be a scalar or arm-joint mapping.")
        if not np.allclose(normalized, EXPECTED_MAX_RELATIVE_TARGET, rtol=0.0, atol=1e-12) or not np.allclose(
            shadow_normalized, EXPECTED_MAX_RELATIVE_TARGET, rtol=0.0, atol=1e-12
        ):
            raise ValueError("ViperX max_relative_target must remain 5 normalized units.")

        raw_steps = np.floor(
            normalized / NORMALIZED_POSITION_SPAN * (self.mapping.raw_max - self.mapping.raw_min)
        ).astype(np.int64)
        if np.any(raw_steps < 1):
            raise ValueError("max_relative_target converts to less than one encoder tick.")
        max_step_rad = raw_steps * self.mapping.scale_rad_per_tick
        shadow_raw_steps = np.floor(
            shadow_normalized
            / NORMALIZED_POSITION_SPAN
            * (self.mapping.shadow_raw_max - self.mapping.shadow_raw_min)
        ).astype(np.int64)
        if np.any(shadow_raw_steps < 1):
            raise ValueError("Shadow max_relative_target converts to less than one encoder tick.")
        shadow_step_rad = shadow_raw_steps * self.mapping.shadow_scale_rad_per_tick
        for shadow_index, joint_index in enumerate(self.mapping.shadow_joint_indices):
            max_step_rad[joint_index] = min(
                max_step_rad[joint_index], shadow_step_rad[shadow_index]
            )
        return raw_steps, max_step_rad

    def _validate_q(self, q_rad: Sequence[float]) -> FloatArray:
        return np.asarray(
            self.model.validate_joints(self.mapping.validate_q(q_rad)), dtype=np.float64
        )

    @property
    def is_connected(self) -> bool:
        return bool(self.robot.is_connected)

    def _cleanup_connections(self, *, disable_torque: bool) -> None:
        bus = getattr(self.robot, "bus", None)
        if getattr(bus, "is_connected", False):
            try:
                bus.disconnect(disable_torque)
            except Exception:
                pass

        for camera in getattr(self.robot, "cameras", {}).values():
            if getattr(camera, "is_connected", False):
                try:
                    camera.disconnect()
                except Exception:
                    pass

    def _cleanup_failed_connect(self) -> None:
        disable_torque = getattr(
            getattr(self.robot, "config", None), "disable_torque_on_disconnect", True
        )
        self._cleanup_connections(disable_torque=bool(disable_torque))

    def _validate_goal_update_mode(self) -> None:
        drive_modes = self.robot.bus.sync_read(
            "Drive_Mode", list(self.arm_actuator_names), normalize=False
        )
        unsafe = {
            name: int(drive_modes[name])
            for name in self.arm_actuator_names
            if int(drive_modes[name]) & TORQUE_ON_BY_GOAL_UPDATE_BIT
        }
        if unsafe:
            raise RuntimeError(
                "Torque On by Goal Update is enabled for arm motors; "
                f"cannot safely prepare a next-session goal while torque is off: {unsafe}"
            )

    def _request_stop(self, source: str) -> None:
        if self._motion_state == "active":
            print(f"ViperX stop requested by {source}.")
            self._stop_requested.set()
        elif self._motion_state in {"holding", "unsafe"}:
            self._release_requested.set()

    def _handle_sigint(self, _signum: int, _frame: Any) -> None:
        self._request_stop("Ctrl-C")

    def _install_sigint_handler(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _restore_sigint_handler(self) -> None:
        if self._previous_sigint_handler is None:
            return
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
        self._previous_sigint_handler = None

    def _monitor_terminal(self) -> None:
        while not self._terminal_monitor_stop.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            except (OSError, ValueError):
                return
            if not readable:
                continue
            if sys.stdin.readline() == "":
                return
            self._request_stop("Enter")

    def _start_terminal_monitor(self) -> None:
        if not self.interactive_safety or not sys.stdin.isatty():
            return
        print("Safety: Enter or Ctrl-C halts at the current pose. After you support the arm, Enter releases torque and disconnects.")
        self._terminal_monitor_stop.clear()
        self._terminal_monitor = threading.Thread(
            target=self._monitor_terminal,
            name="viperx-safety-input",
            daemon=True,
        )
        self._terminal_monitor.start()

    def _stop_terminal_monitor(self) -> None:
        self._terminal_monitor_stop.set()
        monitor = self._terminal_monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=0.2)
        self._terminal_monitor = None

    def _reset_safety_state(self) -> None:
        self._stop_terminal_monitor()
        self._restore_sigint_handler()
        self._session_start_raw = None
        self._held_raw = None
        self._motion_state = "disconnected"
        self._stop_requested.clear()
        self._release_requested.clear()

    def connect(self) -> None:
        if self.is_connected:
            raise RuntimeError("ViperX adapter is already connected.")
        try:
            self.robot.connect(calibrate=False)
            if not self.robot.bus.is_calibrated:
                raise RuntimeError("LeRobot calibration was not applied to the hardware.")
            self._validate_goal_update_mode()
            self._session_start_raw, _ = self._read_raw_actuator_snapshot()
            self._write_raw_actuators(self._session_start_raw)
            self._motion_state = "active"
            self._install_sigint_handler()
            self._start_terminal_monitor()
        except Exception:
            self._cleanup_failed_connect()
            self._reset_safety_state()
            raise

    def disconnect(self) -> None:
        if self._motion_state in {"holding", "unsafe"}:
            raise RuntimeError("Support the arm and call release() before disconnecting.")
        disable_torque = getattr(
            getattr(self.robot, "config", None), "disable_torque_on_disconnect", True
        )
        self._cleanup_connections(disable_torque=bool(disable_torque))
        self._reset_safety_state()

    def _read_raw_actuator_snapshot(self) -> tuple[dict[str, int], float]:
        if not self.is_connected:
            raise RuntimeError("ViperX adapter is not connected.")
        raw = self.robot.bus.sync_read(
            "Present_Position", list(self.arm_actuator_names), normalize=False
        )
        return self.mapping.validate_actuator_raw(raw), time.time()

    def _read_raw_snapshot(self) -> tuple[dict[str, int], float]:
        actuator_raw, timestamp_s = self._read_raw_actuator_snapshot()
        return (
            {name: actuator_raw[name] for name in self.arm_joint_names},
            timestamp_s,
        )

    def read_raw_joints(self) -> dict[str, int]:
        raw, _ = self._read_raw_snapshot()
        return raw

    def raw_to_rad(self, raw_positions: Mapping[str, Any]) -> FloatArray:
        return self._validate_q(self.mapping.raw_to_rad(raw_positions))

    def rad_to_raw(self, q_rad: Sequence[float]) -> dict[str, int]:
        return self.mapping.rad_to_raw(self._validate_q(q_rad))

    def read_joints(self) -> FloatArray:
        return self.raw_to_rad(self.read_raw_joints())

    def _read_joints_unchecked(self) -> FloatArray:
        return np.asarray(self.mapping.raw_to_rad(self.read_raw_joints()), dtype=np.float64)

    def get_ee_pose(self) -> FloatArray:
        return np.asarray(self.model.fk(self.read_joints()), dtype=np.float64)

    def _write_raw_actuators(self, raw_targets: Mapping[str, Any]) -> None:
        if not self.is_connected:
            raise RuntimeError("ViperX adapter is not connected.")
        self.robot.bus.sync_write(
            "Goal_Position", self.mapping.validate_actuator_raw(raw_targets), normalize=False
        )

    def _raise_if_stop_requested(self) -> None:
        if self._motion_state != "active":
            raise ViperXMotionInterrupted(
                "ViperX motion is halted; support the arm and call release() before another command."
            )
        if self._stop_requested.is_set():
            raise ViperXMotionInterrupted("ViperX motion halt was requested.")

    def stop_motion(self) -> dict[str, int]:
        """Hold the current pose and block further motion commands."""

        if not self.is_connected:
            raise RuntimeError("ViperX adapter is not connected.")
        if self._motion_state not in {"active", "holding"}:
            raise RuntimeError(f"Cannot halt motion from state {self._motion_state!r}.")

        try:
            held_raw, _ = self._read_raw_actuator_snapshot()
            self._write_raw_actuators(held_raw)
        except Exception:
            self._motion_state = "unsafe"
            print("ViperX could not confirm a holding target; support the arm before releasing torque.")
            raise

        self._held_raw = held_raw
        self._motion_state = "holding"
        print("ViperX is holding its current pose. Support the arm, then press Enter or call release().")
        return {name: held_raw[name] for name in self.arm_joint_names}

    def release(self, *, prepare_raw: Mapping[str, Any] | None = None) -> None:
        """Disable torque, prepare the next-session goal, and close all connections."""

        if self._motion_state not in {"holding", "unsafe"}:
            raise RuntimeError("release() requires a halted ViperX motion.")
        if not self.is_connected:
            self._reset_safety_state()
            return

        next_goal = prepare_raw if prepare_raw is not None else self._held_raw
        if next_goal is None:
            raise RuntimeError("No safe raw goal is available for release().")

        release_error: Exception | None = None
        try:
            self.robot.bus.disable_torque()
            self.robot.bus.sync_write(
                "Goal_Position", self.mapping.validate_actuator_raw(next_goal), normalize=False
            )
        except Exception as exc:
            release_error = exc
        finally:
            self._cleanup_connections(disable_torque=False)
            self._reset_safety_state()

        if release_error is not None:
            raise RuntimeError("ViperX release did not complete cleanly.") from release_error

    def await_release(self, *, prepare_raw: Mapping[str, Any] | None = None) -> None:
        """Wait for explicit operator confirmation, then release the supported arm."""

        if self._motion_state not in {"holding", "unsafe"}:
            raise RuntimeError("await_release() requires a halted ViperX motion.")
        if self._terminal_monitor is not None:
            self._release_requested.wait()
        elif self.interactive_safety:
            input("Support the arm, then press Enter to release torque and disconnect: ")
        self.release(prepare_raw=prepare_raw)

    def _handle_motion_failure(self) -> None:
        if self._motion_state == "active":
            try:
                self.stop_motion()
            except Exception:
                pass
        if self._motion_state in {"holding", "unsafe"} and self.interactive_safety:
            print("After supporting the arm, press Enter to release torque and disconnect.")
            self.await_release(prepare_raw=self._session_start_raw)

    def move_joints(
        self,
        q_target_rad: Sequence[float],
        *,
        timeout_s: float | None = None,
        tolerance_rad: float | None = None,
        allow_out_of_limit_start: bool = False,
    ) -> FloatArray:
        if not self.is_connected:
            raise RuntimeError("ViperX adapter is not connected.")
        target = self._validate_q(q_target_rad)
        self.mapping.rad_to_actuator_raw(target)

        timeout = self.motion_timeout_s if timeout_s is None else float(timeout_s)
        tolerance = self.position_tolerance_rad if tolerance_rad is None else float(tolerance_rad)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_s must be finite and positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance_rad must be finite and positive.")

        read_measured = (
            self._read_joints_unchecked if allow_out_of_limit_start else self.read_joints
        )
        map_command = (
            self.mapping._rad_to_actuator_raw_unchecked
            if allow_out_of_limit_start
            else self.mapping.rad_to_actuator_raw
        )

        try:
            self._raise_if_stop_requested()
            deadline = time.monotonic() + timeout
            settled = 0
            measured = read_measured()
            while True:
                self._raise_if_stop_requested()
                error = target - measured
                arrived = bool(np.max(np.abs(error)) <= tolerance)
                if allow_out_of_limit_start and arrived:
                    try:
                        self._validate_q(measured)
                    except ValueError:
                        arrived = False
                if arrived:
                    settled += 1
                    if settled >= self.settle_samples:
                        return measured
                else:
                    settled = 0
                    next_q = measured + np.clip(
                        error, -self.max_joint_step_rad, self.max_joint_step_rad
                    )
                    self._write_raw_actuators(map_command(next_q))

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    max_error = float(np.max(np.abs(measured - target)))
                    raise TimeoutError(
                        f"ViperX did not settle within {timeout:.3f}s; "
                        f"maximum joint error is {max_error:.6f} rad."
                    )
                time.sleep(min(self.command_period_s, remaining))
                measured = read_measured()
        except (Exception, KeyboardInterrupt):
            self._handle_motion_failure()
            raise

    def prepare_for_motion(self) -> FloatArray:
        """Move from an unchecked startup state to the fixed in-limit staging pose."""

        return self.move_joints(
            STARTUP_STAGING_Q_RAD,
            allow_out_of_limit_start=True,
        )

    def move_ee(
        self,
        target_pose: Sequence[Sequence[float]],
        *,
        q_seed: Sequence[float] | None = None,
    ) -> FloatArray:
        seed = self.read_joints() if q_seed is None else self._validate_q(q_seed)
        result = self.model.ik(target_pose, q0_arm=seed)
        if not bool(result.success):
            raise RuntimeError(
                f"ViperX IK failed: reason={getattr(result, 'reason', None)!r}, "
                f"residual={getattr(result, 'residual', None)!r}"
            )
        return self.move_joints(result.q_arm)

    def go_home(self) -> FloatArray:
        return self.move_joints(self.home_q_rad)

    def get_obs(self) -> dict[str, Any]:
        raw, timestamp_s = self._read_raw_snapshot()
        q_rad = self.raw_to_rad(raw)
        return {
            "timestamp_s": timestamp_s,
            "joint_positions": q_rad,
            "ee_pose": np.asarray(self.model.fk(q_rad), dtype=np.float64),
            "raw_joint_positions": raw,
        }

    def apply_action(self, action: Any, *, action_type: str = "joint") -> FloatArray:
        if action_type == "joint":
            return self.move_joints(action)
        if action_type == "ee":
            if isinstance(action, tuple):
                raise NotImplementedError(
                    "ViperX gripper-width semantics are not defined for EE actions."
                )
            return self.move_ee(action)
        raise ValueError(f"Unsupported action_type: {action_type!r}")
