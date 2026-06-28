"""Re3Sim-facing robot adapter for a LeRobot-controlled ViperX300s.

Purpose
-------
This module exposes a small, stable robot interface to data collection,
calibration, and Re3Sim-side code. Implementation details stay below the
interface boundary:

1. LeRobot owns hardware connection, motor observations, and position commands.
2. The injected kinematics object owns forward/inverse kinematics.
3. Calibration code consumes robot-agnostic joint and end-effector interfaces.
4. Re3Sim consumes calibration products instead of talking to hardware.

The adapter receives an existing LeRobot ``ViperX`` instance. Motor names,
calibration ranges, raw encoder positions, and encoder resolutions are read
from that object instead of being configured a second time.

Coordinate and unit contract
----------------------------
- Arm joint order is discovered from the LeRobot driver's motor definitions.
- Public joint vectors use radians and have shape ``(6,)``.
- End-effector poses are 4x4 transforms ``T_base_ee``.
- Encoder IO uses LeRobot's motor bus with ``normalize=False``.
- Shadow motors and the gripper are not part of the arm joint vector.

The adapter must not be used on hardware until:
- encoder zero positions and signs have been aligned with the URDF;
- the configured model/end-effector frame matches the physical robot;
- ``home_q_rad`` and motion limits have been confirmed on the real arm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

EXPECTED_ARM_DOF = 6
NON_KINEMATIC_MOTORS = frozenset(
    {"shoulder_shadow", "elbow_shadow", "gripper"}
)


@dataclass(frozen=True)
class IKResult:
    """Robot-independent result returned by the kinematics adapter."""

    q_rad: FloatArray
    success: bool
    residual: float | None = None
    reason: str | None = None


class ViperXKinematics:
    """Small wrapper around an injected ViperX kinematics model.

    Construct/load the kinematics model outside this class, then inject it
    here. Injection avoids hard-coding an unverified URDF-loading procedure or
    end-effector frame in the hardware adapter.

    Expected model methods:
    - ``fkine(q, ...)``
    - ``ikine_LM(T, q0=..., joint_limits=True, ...)``
    """

    def __init__(self, robot_model: Any, *, ee_link: str | None = None):
        self.robot_model = robot_model
        self.ee_link = ee_link

    def _end_kwargs(self) -> dict[str, str]:
        """Pass an explicit end frame only when one has been configured."""

        return {} if self.ee_link is None else {"end": self.ee_link}

    def validate_joints(self, q_rad: Sequence[float]) -> FloatArray:
        q = np.asarray(q_rad, dtype=np.float64)
        if q.shape != (EXPECTED_ARM_DOF,):
            raise ValueError(
                f"Expected {EXPECTED_ARM_DOF} ViperX arm joints, got shape {q.shape}."
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint vector contains NaN or infinity.")

        qlim = getattr(self.robot_model, "qlim", None)
        if qlim is not None:
            limits = np.asarray(qlim, dtype=np.float64)
            if limits.shape == (2, q.size):
                lower, upper = limits
                if np.any(q < lower) or np.any(q > upper):
                    raise ValueError(
                        f"Joint target violates model limits: q={q}, "
                        f"lower={lower}, upper={upper}."
                    )
        return q

    @staticmethod
    def validate_transform(transform: Sequence[Sequence[float]]) -> FloatArray:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Transform contains NaN or infinity.")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ValueError("Transform has an invalid homogeneous bottom row.")

        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Transform rotation is not orthonormal.")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("Transform rotation determinant is not +1.")
        return matrix

    def fk(self, q_rad: Sequence[float]) -> FloatArray:
        """Return ``T_base_ee`` for a six-joint ViperX vector in radians."""

        q = self.validate_joints(q_rad)
        pose = self.robot_model.fkine(q, **self._end_kwargs())
        matrix = getattr(pose, "A", pose)
        return self.validate_transform(matrix)

    def ik(
        self,
        target_pose: Sequence[Sequence[float]],
        *,
        q_seed: Sequence[float],
    ) -> IKResult:
        """Solve ``T_base_ee -> q_rad`` using the current joints as the seed."""

        target = self.validate_transform(target_pose)
        seed = self.validate_joints(q_seed)
        solution = self.robot_model.ikine_LM(
            target,
            q0=seed,
            joint_limits=True,
            **self._end_kwargs(),
        )

        q = np.asarray(solution.q, dtype=np.float64)
        success = bool(solution.success)
        residual_value = getattr(solution, "residual", None)
        residual = None if residual_value is None else float(residual_value)
        reason_value = getattr(solution, "reason", None)
        reason = None if reason_value is None else str(reason_value)

        if success:
            q = self.validate_joints(q)

        return IKResult(
            q_rad=q,
            success=success,
            residual=residual,
            reason=reason,
        )


class ViperXAdapter:
    """Adapt LeRobot ViperX IO to a stable joint/EE robot interface.

    ``robot`` is an already configured LeRobot ViperX object. The adapter does
    not duplicate its port, ID, calibration directory, motor models, or motor
    calibration.

    Only the cross-library alignment remains explicit:
    - ``urdf_zero_raw[joint]``: raw encoder value representing URDF q=0.
    - ``urdf_joint_sign[joint]``: +1/-1 mapping encoder direction to URDF q.

    Camera capture and calibration algorithms stay outside this class.
    """

    def __init__(
        self,
        robot: Any,
        kinematics: ViperXKinematics,
        *,
        urdf_zero_raw: Mapping[str, float],
        urdf_joint_sign: Mapping[str, int],
        home_q_rad: Sequence[float] | None = None,
        max_joint_step_rad: float = 0.03,
        command_period_s: float = 0.05,
        motion_timeout_s: float = 10.0,
        position_tolerance_rad: float = 0.03,
    ):
        self.robot = robot
        self.kinematics = kinematics
        self.arm_joint_names = self._discover_arm_joint_names(robot)
        self._resolutions = self._read_encoder_resolutions()
        self._calibration = self._read_lerobot_calibration()
        self.urdf_zero_raw = self._validate_named_values(
            urdf_zero_raw, "urdf_zero_raw"
        )
        self.urdf_joint_sign = self._validate_joint_signs(urdf_joint_sign)
        self.home_q_rad = (
            None
            if home_q_rad is None
            else self.kinematics.validate_joints(home_q_rad)
        )

        self.max_joint_step_rad = float(max_joint_step_rad)
        self.command_period_s = float(command_period_s)
        self.motion_timeout_s = float(motion_timeout_s)
        self.position_tolerance_rad = float(position_tolerance_rad)
        if self.max_joint_step_rad <= 0.0:
            raise ValueError("max_joint_step_rad must be positive.")
        if self.command_period_s <= 0.0:
            raise ValueError("command_period_s must be positive.")
        if self.motion_timeout_s <= 0.0:
            raise ValueError("motion_timeout_s must be positive.")
        if self.position_tolerance_rad <= 0.0:
            raise ValueError("position_tolerance_rad must be positive.")

    @staticmethod
    def _discover_arm_joint_names(robot: Any) -> tuple[str, ...]:
        """Use the LeRobot driver's motor declaration as the source of truth."""

        try:
            motor_names = tuple(robot.bus.motors)
        except AttributeError as exc:
            raise TypeError(
                "robot must be a LeRobot ViperX-like object with bus.motors."
            ) from exc
        arm_joint_names = tuple(
            name for name in motor_names if name not in NON_KINEMATIC_MOTORS
        )
        if len(arm_joint_names) != EXPECTED_ARM_DOF:
            raise ValueError(
                "Expected six kinematic motors after excluding shadow motors "
                f"and gripper, found {arm_joint_names}."
            )
        return arm_joint_names

    def _read_encoder_resolutions(self) -> dict[str, int]:
        """Read each motor model's encoder resolution from LeRobot."""

        resolutions = {}
        for joint in self.arm_joint_names:
            motor = self.robot.bus.motors[joint]
            try:
                resolution = int(
                    self.robot.bus.model_resolution_table[motor.model]
                )
            except (AttributeError, KeyError) as exc:
                raise ValueError(
                    f"LeRobot has no encoder resolution for {joint}/{motor.model}."
                ) from exc
            if resolution <= 0:
                raise ValueError(f"Invalid encoder resolution for {joint}.")
            resolutions[joint] = resolution
        return resolutions

    def _read_lerobot_calibration(self) -> dict[str, Any]:
        """Reuse calibration loaded by LeRobot from ``<robot_id>.json``."""

        calibration = getattr(self.robot, "calibration", None)
        if not calibration:
            calibration = getattr(self.robot.bus, "calibration", None)
        if not calibration:
            calibration_path = getattr(self.robot, "calibration_fpath", None)
            raise RuntimeError(
                "LeRobot has no loaded motor calibration. Expected calibration "
                f"file: {calibration_path!s}"
            )
        missing = [name for name in self.arm_joint_names if name not in calibration]
        if missing:
            raise KeyError(f"LeRobot calibration is missing motors: {missing}")
        return dict(calibration)

    def _validate_named_values(
        self,
        values: Mapping[str, float],
        field_name: str,
    ) -> dict[str, float]:
        expected = set(self.arm_joint_names)
        actual = set(values)
        if actual != expected:
            raise ValueError(
                f"{field_name} keys must match LeRobot arm joints; "
                f"missing={expected - actual}, extra={actual - expected}."
            )
        result = {name: float(values[name]) for name in self.arm_joint_names}
        if not np.all(np.isfinite(list(result.values()))):
            raise ValueError(f"{field_name} contains NaN or infinity.")
        return result

    def _validate_joint_signs(
        self,
        signs: Mapping[str, int],
    ) -> dict[str, int]:
        values = self._validate_named_values(signs, "urdf_joint_sign")
        invalid = {name: value for name, value in values.items() if value not in (-1, 1)}
        if invalid:
            raise ValueError(f"Joint signs must be +1 or -1: {invalid}")
        return {name: int(value) for name, value in values.items()}

    @classmethod
    def assuming_calibration_neutral_is_urdf_zero(
        cls,
        robot: Any,
        kinematics: ViperXKinematics,
        *,
        urdf_joint_sign: Mapping[str, int],
        **kwargs: Any,
    ) -> "ViperXAdapter":
        """Use LeRobot's half-turn neutral as URDF q=0.

        ViperX calibration calls ``set_half_turn_homings()``. For a 4096-count
        motor, the pose held during that step becomes raw position 2047. Use
        this constructor only if that physical calibration pose was the URDF
        zero pose.
        """

        names = cls._discover_arm_joint_names(robot)
        zero_raw = {}
        for joint in names:
            motor = robot.bus.motors[joint]
            resolution = int(robot.bus.model_resolution_table[motor.model])
            zero_raw[joint] = int((resolution - 1) / 2)
        return cls(
            robot,
            kinematics,
            urdf_zero_raw=zero_raw,
            urdf_joint_sign=urdf_joint_sign,
            **kwargs,
        )

    @property
    def is_connected(self) -> bool:
        return bool(self.robot.is_connected)

    def connect(self) -> None:
        """Delegate connection to the injected LeRobot object."""

        if self.is_connected:
            raise RuntimeError("ViperX adapter is already connected.")
        self.robot.connect(calibrate=False)

    def _require_connected(self) -> Any:
        if not self.is_connected:
            raise RuntimeError("ViperX adapter is not connected.")
        return self.robot

    def read_raw_observation(self) -> dict[str, Any]:
        """Return LeRobot's normalized policy observation for diagnostics."""

        robot = self._require_connected()
        return robot.get_observation()

    def read_raw_joints(self) -> dict[str, int]:
        """Read raw encoder positions through LeRobot without normalization."""

        robot = self._require_connected()
        positions = robot.bus.sync_read(
            "Present_Position",
            list(self.arm_joint_names),
            normalize=False,
        )
        return {name: int(positions[name]) for name in self.arm_joint_names}

    def raw_to_rad(self, raw_positions: Mapping[str, int]) -> FloatArray:
        """Convert LeRobot raw encoder values to URDF joint radians."""

        missing = set(self.arm_joint_names) - set(raw_positions)
        if missing:
            raise KeyError(f"Raw joint positions are missing: {missing}")
        return np.asarray(
            [
                self.urdf_joint_sign[name]
                * (float(raw_positions[name]) - self.urdf_zero_raw[name])
                * (2.0 * np.pi / self._resolutions[name])
                for name in self.arm_joint_names
            ],
            dtype=np.float64,
        )

    def rad_to_raw(self, q_rad: Sequence[float]) -> dict[str, int]:
        """Convert URDF joint radians to raw encoder targets.

        Target limits come from LeRobot's loaded calibration, not a duplicated
        adapter configuration.
        """

        q = self.kinematics.validate_joints(q_rad)
        raw_targets = {}
        for index, name in enumerate(self.arm_joint_names):
            raw = round(
                self.urdf_zero_raw[name]
                + self.urdf_joint_sign[name]
                * q[index]
                * self._resolutions[name]
                / (2.0 * np.pi)
            )
            calibration = self._calibration[name]
            if raw < calibration.range_min or raw > calibration.range_max:
                raise ValueError(
                    f"{name} target {raw} violates LeRobot calibration range "
                    f"[{calibration.range_min}, {calibration.range_max}]."
                )
            raw_targets[name] = int(raw)
        return raw_targets

    def read_joints_rad(self) -> FloatArray:
        """Read measured six-DoF arm joints in radians."""

        return self.kinematics.validate_joints(
            self.raw_to_rad(self.read_raw_joints())
        )

    def read_joints(self) -> FloatArray:
        """Stable interface alias for measured arm joints in radians."""

        return self.read_joints_rad()

    def get_ee_pose(self) -> FloatArray:
        """Return the measured ``T_base_ee`` obtained by FK."""

        return self.kinematics.fk(self.read_joints_rad())

    def get_end_effector_pose(self) -> FloatArray:
        """Stable interface alias for the measured end-effector pose."""

        return self.get_ee_pose()

    def _write_raw_joints(self, raw_targets: Mapping[str, int]) -> None:
        """Write raw targets through LeRobot's public motor-bus interface."""

        robot = self._require_connected()
        robot.bus.sync_write(
            "Goal_Position",
            dict(raw_targets),
            normalize=False,
        )

    def move_joints(
        self,
        q_target_rad: Sequence[float],
        *,
        timeout_s: float | None = None,
        tolerance_rad: float | None = None,
    ) -> FloatArray:
        """Move by bounded radian steps while LeRobot performs raw motor IO."""

        self._require_connected()
        target = self.kinematics.validate_joints(q_target_rad)

        timeout = (
            self.motion_timeout_s if timeout_s is None else float(timeout_s)
        )
        tolerance = (
            self.position_tolerance_rad
            if tolerance_rad is None
            else float(tolerance_rad)
        )
        if timeout <= 0.0:
            raise ValueError("timeout_s must be positive.")
        if tolerance <= 0.0:
            raise ValueError("tolerance_rad must be positive.")

        deadline = time.monotonic() + timeout
        measured = self.read_joints_rad()
        while time.monotonic() < deadline:
            error = target - measured
            if np.max(np.abs(error)) <= tolerance:
                return measured
            next_q = measured + np.clip(
                error,
                -self.max_joint_step_rad,
                self.max_joint_step_rad,
            )
            self._write_raw_joints(self.rad_to_raw(next_q))
            time.sleep(self.command_period_s)
            measured = self.read_joints_rad()

        max_error = float(np.max(np.abs(measured - target)))
        raise TimeoutError(
            f"ViperX did not reach target within {timeout:.3f}s; "
            f"maximum joint error is {max_error:.6f} rad."
        )

    def move_to_joint_positions(
        self,
        q_target_rad: Sequence[float],
        *,
        timeout_s: float | None = None,
        tolerance_rad: float | None = None,
    ) -> FloatArray:
        """Stable interface alias for joint-space motion."""

        return self.move_joints(
            q_target_rad,
            timeout_s=timeout_s,
            tolerance_rad=tolerance_rad,
        )

    def move_ee(
        self,
        target_pose: Sequence[Sequence[float]],
        *,
        q_seed: Sequence[float] | None = None,
    ) -> FloatArray:
        """Solve IK for ``T_base_ee`` and execute the resulting joint target."""

        seed = self.read_joints_rad() if q_seed is None else np.asarray(q_seed)
        result = self.kinematics.ik(target_pose, q_seed=seed)
        if not result.success:
            raise RuntimeError(
                "ViperX IK failed: "
                f"reason={result.reason!r}, residual={result.residual!r}"
            )
        return self.move_joints(result.q_rad)

    def move_to_end_effector_pose(
        self,
        target_pose: Sequence[Sequence[float]],
        *,
        q_seed: Sequence[float] | None = None,
    ) -> FloatArray:
        """Stable interface alias for end-effector motion."""

        return self.move_ee(target_pose, q_seed=q_seed)

    def move_home(self) -> FloatArray:
        """Move to the explicitly configured and physically verified home pose."""

        if self.home_q_rad is None:
            raise RuntimeError(
                "home_q_rad is not configured. Refusing to assume a zero pose."
            )
        return self.move_joints(self.home_q_rad)

    def go_home(self) -> FloatArray:
        """Stable interface alias for returning to the verified home pose."""

        return self.move_home()

    def get_obs(self) -> dict[str, Any]:
        """Return a small Re3Sim-facing observation plus raw driver state."""

        raw_observation = self.read_raw_observation()
        raw_joints = self.read_raw_joints()
        q_rad = self.kinematics.validate_joints(self.raw_to_rad(raw_joints))
        return {
            "joint_positions": q_rad,
            "ee_pose": self.kinematics.fk(q_rad),
            "raw_joint_positions": raw_joints,
            "raw_lerobot": raw_observation,
        }

    def apply_action(self, action: Any, *, action_type: str = "joint") -> FloatArray:
        """Compatibility entry point for joint or end-effector actions.

        ``joint``:
            ``action`` is a six-element radian vector.

        ``ee``:
            ``action`` is a 4x4 ``T_base_ee`` matrix. Re3Sim's
            ``(gripper_width, transform)`` form is intentionally rejected until
            the ViperX gripper-width conversion has been specified.
        """

        if action_type == "joint":
            return self.move_joints(action)
        if action_type == "ee":
            if isinstance(action, tuple):
                raise NotImplementedError(
                    "Define ViperX gripper-width semantics before accepting "
                    "(gripper_width, transform) EE actions."
                )
            return self.move_ee(action)
        raise ValueError(f"Unsupported action_type: {action_type!r}")

    def disconnect(self) -> None:
        """Delegate disconnect to the injected LeRobot object."""

        if self.is_connected:
            self.robot.disconnect()

    def __enter__(self) -> "ViperXAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.disconnect()
