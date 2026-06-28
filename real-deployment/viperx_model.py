"""ViperX300s kinematics model loaded from the validated full URDF.

This module keeps Robotics Toolbox behind a small Re3Sim-facing interface:
callers provide the six arm joints only, while the full URDF and its gripper
and finger joints remain available internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF_PATH = PROJECT_ROOT / "viperx_asset" / "urdf" / "vx300s_full.urdf"

DEFAULT_ARM_JOINT_NAMES = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)

DEFAULT_PASSIVE_JOINT_VALUES = {
    "gripper": 0.0,
    "left_finger": 0.021,
    "right_finger": -0.021,
}

DEFAULT_IK_POSITION_TOLERANCE_M = 1e-5
DEFAULT_IK_ROTATION_TOLERANCE_RAD = 1e-5
DEFAULT_IK_SOLVER_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ViperXModelConfig:
    """Explicit configuration for the ViperX FK interface."""

    urdf_path: Path = DEFAULT_URDF_PATH
    base_link: str = "vx300s/base_link"
    end_link: str = "vx300s/ee_gripper_link"
    arm_joint_names: tuple[str, ...] = DEFAULT_ARM_JOINT_NAMES
    passive_joint_values: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PASSIVE_JOINT_VALUES)
    )
    # Avoid sharing a mutable dict between dataclass instances.


@dataclass(frozen=True)
class ViperXIKResult:
    """Numerical IK result for the six exposed ViperX arm joints."""

    q_arm: FloatArray
    success: bool
    position_error: float
    rotation_error: float
    residual: float
    iterations: int
    searches: int
    reason: str


class ViperXModel:
    """Six-DOF arm FK/IK wrapper around a full Robotics Toolbox URDF model."""

    def __init__(self, robot_model: object, config: ViperXModelConfig):
        self.robot_model = robot_model
        self.config = config
        self.base_link = self._resolve_link_name(config.base_link)
        self.end_link = self._resolve_link_name(config.end_link)
        self.joint_names = self._read_joint_names_by_index()
        self.joint_index_by_name = {
            name: index for index, name in enumerate(self.joint_names)
        }
        self.arm_joint_names = tuple(config.arm_joint_names)
        self.arm_joint_indices = self._resolve_joint_indices(self.arm_joint_names)
        self.passive_joint_values = dict(config.passive_joint_values)
        self.full_joint_defaults = self._build_full_joint_defaults()
        self.qlim = self._build_arm_qlim()
        self._validate_config()

    @property
    def n(self) -> int:
        """Expose the arm DOF expected by Re3Sim calibration code."""

        return len(self.arm_joint_names)

    @property
    def full_n(self) -> int:
        """Number of active joints in the full URDF model."""

        return len(self.joint_names)

    def _link_names(self) -> tuple[str, ...]:
        return tuple(link.name for link in self.robot_model.links)

    def _resolve_link_name(self, link_name: str) -> str:
        names = self._link_names()
        if link_name in names:
            return link_name

        suffix_matches = [name for name in names if name.endswith(f"/{link_name}")]
        if len(suffix_matches) == 1:
            return suffix_matches[0]

        raise ValueError(
            f"Cannot resolve link {link_name!r}. Available links: {names}"
        )

    def _read_joint_names_by_index(self) -> tuple[str, ...]:
        joint_names: list[str | None] = [None] * int(self.robot_model.n)
        for link in self.robot_model.links:
            if not getattr(link, "isjoint", False):
                continue
            joint_index = int(link.jindex)
            joint_name = getattr(link, "_joint_name", None)
            if joint_name is None:
                raise ValueError(f"RTB link {link.name!r} has no joint name.")
            joint_names[joint_index] = str(joint_name)

        missing = [index for index, name in enumerate(joint_names) if name is None]
        if missing:
            raise ValueError(f"Missing joint names for RTB indices {missing}.")
        return tuple(name for name in joint_names if name is not None)

    def _resolve_joint_indices(self, names: Sequence[str]) -> tuple[int, ...]:
        missing = [name for name in names if name not in self.joint_index_by_name]
        if missing:
            raise ValueError(f"Missing arm joints in RTB model: {missing}")
        return tuple(self.joint_index_by_name[name] for name in names)

    def _build_full_joint_defaults(self) -> FloatArray:
        q = np.zeros(self.full_n, dtype=np.float64)
        for name, value in self.passive_joint_values.items():
            if name in self.joint_index_by_name:
                q[self.joint_index_by_name[name]] = float(value)
        return q

    def _build_arm_qlim(self) -> FloatArray:
        qlim = np.asarray(self.robot_model.qlim, dtype=np.float64)
        if qlim.shape != (2, self.full_n):
            raise ValueError(
                f"Expected full model qlim shape {(2, self.full_n)}, got {qlim.shape}."
            )
        return qlim[:, self.arm_joint_indices]

    def _validate_config(self) -> None:
        expected = tuple(DEFAULT_ARM_JOINT_NAMES)
        if self.arm_joint_names != expected:
            raise ValueError(
                f"Unexpected arm joint order {self.arm_joint_names}; expected {expected}."
            )

        path, _, _ = self.robot_model.get_path(start=self.base_link, end=self.end_link)
        path_names = tuple(link.name for link in path)
        if self.base_link not in path_names:
            raise ValueError(f"Base link {self.base_link!r} not on FK path.")
        if self.end_link not in path_names:
            raise ValueError(f"End link {self.end_link!r} not on FK path.")

        for name, index in self.joint_index_by_name.items():
            value = self.full_joint_defaults[index]
            qlim = np.asarray(self.robot_model.qlim, dtype=object)[:, index]
            lower, upper = qlim
            if lower is not None and value < float(lower):
                raise ValueError(f"Default for {name}={value} is below {lower}.")
            if upper is not None and value > float(upper):
                raise ValueError(f"Default for {name}={value} is above {upper}.")

    def validate_joints(self, q_arm: Sequence[float]) -> FloatArray:
        q = np.asarray(q_arm, dtype=np.float64)
        if q.shape != (self.n,):
            raise ValueError(f"Expected q_arm shape {(self.n,)}, got {q.shape}.")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_arm contains NaN or infinity.")

        lower, upper = self.qlim
        if np.any(q < lower) or np.any(q > upper):
            raise ValueError(
                f"q_arm violates limits: q={q}, lower={lower}, upper={upper}."
            )
        return q

    def _default_q0_arm(self) -> FloatArray:
        zero = np.zeros(self.n, dtype=np.float64)
        lower, upper = self.qlim
        if np.all(zero >= lower) and np.all(zero <= upper):
            return zero
        return (lower + upper) / 2.0

    def expand_q_arm(self, q_arm: Sequence[float]) -> FloatArray:
        """Return full-model q with arm joints set and passive joints fixed."""

        q = self.full_joint_defaults.copy()
        q_arm_array = self.validate_joints(q_arm)
        for source_index, target_index in enumerate(self.arm_joint_indices):
            q[target_index] = q_arm_array[source_index]
        return q

    @staticmethod
    def validate_transform(transform: Sequence[Sequence[float]]) -> FloatArray:
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Transform contains NaN or infinity.")
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
            raise ValueError("Transform has an invalid homogeneous bottom row.")

        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("Transform rotation is not orthonormal.")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("Transform rotation determinant is not +1.")
        return matrix

    @staticmethod
    def rotation_error_rad(
        left: Sequence[Sequence[float]],
        right: Sequence[Sequence[float]],
    ) -> float:
        left_rotation = np.asarray(left, dtype=np.float64)
        right_rotation = np.asarray(right, dtype=np.float64)
        delta = left_rotation.T @ right_rotation
        cos_theta = (np.trace(delta) - 1.0) / 2.0
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    @classmethod
    def transform_error(
        cls,
        left: Sequence[Sequence[float]],
        right: Sequence[Sequence[float]],
    ) -> tuple[float, float]:
        left_matrix = cls.validate_transform(left)
        right_matrix = cls.validate_transform(right)
        position_error = float(
            np.linalg.norm(left_matrix[:3, 3] - right_matrix[:3, 3])
        )
        rotation_error = cls.rotation_error_rad(
            left_matrix[:3, :3],
            right_matrix[:3, :3],
        )
        return position_error, rotation_error

    def fk(self, q_arm: Sequence[float]) -> FloatArray:
        """Return T_base_hand for the six ViperX arm joints."""

        q_full = self.expand_q_arm(q_arm)
        pose = self.robot_model.fkine(q_full, start=self.base_link, end=self.end_link)
        return self.validate_transform(getattr(pose, "A", pose))

    def ik(
        self,
        target_transform: Sequence[Sequence[float]],
        *,
        q0_arm: Sequence[float] | None = None,
        position_tolerance: float = DEFAULT_IK_POSITION_TOLERANCE_M,
        rotation_tolerance: float = DEFAULT_IK_ROTATION_TOLERANCE_RAD,
        solver_tolerance: float = DEFAULT_IK_SOLVER_TOLERANCE,
        ilimit: int = 100,
        slimit: int = 20,
        seed: int | None = 0,
    ) -> ViperXIKResult:
        """Return a numerical IK solution for T_base_hand without hardware side effects."""

        target = self.validate_transform(target_transform)
        q0 = self.validate_joints(
            q0_arm if q0_arm is not None else self._default_q0_arm()
        )

        solution = self.robot_model.ikine_LM(
            target,
            start=self.base_link,
            end=self.end_link,
            q0=q0,
            joint_limits=True,
            tol=solver_tolerance,
            ilimit=ilimit,
            slimit=slimit,
            seed=seed,
        )

        q_solution = np.asarray(getattr(solution, "q", q0), dtype=np.float64)
        if q_solution.shape == (self.full_n,):
            q_arm = q_solution[list(self.arm_joint_indices)]
        elif q_solution.shape == (self.n,):
            q_arm = q_solution
        else:
            return ViperXIKResult(
                q_arm=q0,
                success=False,
                position_error=float("inf"),
                rotation_error=float("inf"),
                residual=float(getattr(solution, "residual", float("inf"))),
                iterations=int(getattr(solution, "iterations", 0)),
                searches=int(getattr(solution, "searches", 0)),
                reason=f"Unexpected IK q shape {q_solution.shape}.",
            )

        try:
            q_arm = self.validate_joints(q_arm)
            solved_transform = self.fk(q_arm)
            position_error, rotation_error = self.transform_error(target, solved_transform)
        except ValueError as exc:
            return ViperXIKResult(
                q_arm=np.asarray(q_arm, dtype=np.float64),
                success=False,
                position_error=float("inf"),
                rotation_error=float("inf"),
                residual=float(getattr(solution, "residual", float("inf"))),
                iterations=int(getattr(solution, "iterations", 0)),
                searches=int(getattr(solution, "searches", 0)),
                reason=str(exc),
            )

        solver_success = bool(getattr(solution, "success", False))
        within_tolerance = (
            position_error <= position_tolerance
            and rotation_error <= rotation_tolerance
        )
        reason = str(getattr(solution, "reason", ""))
        if solver_success and not within_tolerance:
            reason = (
                f"IK FK-backcheck exceeds tolerance: "
                f"position={position_error:.3e}m rotation={rotation_error:.3e}rad."
            )

        return ViperXIKResult(
            q_arm=q_arm,
            success=solver_success and within_tolerance,
            position_error=position_error,
            rotation_error=rotation_error,
            residual=float(getattr(solution, "residual", float("nan"))),
            iterations=int(getattr(solution, "iterations", 0)),
            searches=int(getattr(solution, "searches", 0)),
            reason=reason,
        )

    def fkine(
        self,
        q_arm: Sequence[float],
        *,
        start: str | None = None,
        end: str | None = None,
        **_: object,
    ) -> FloatArray:
        """Compatibility shim for the existing ViperXKinematics adapter."""

        if start is not None and self._resolve_link_name(start) != self.base_link:
            raise ValueError(f"Unexpected start link {start!r}.")
        if end is not None and self._resolve_link_name(end) != self.end_link:
            raise ValueError(f"Unexpected end link {end!r}.")
        return self.fk(q_arm)


def load_viperx_model(
    *,
    urdf_path: str | Path = DEFAULT_URDF_PATH,
    base_link: str = "vx300s/base_link",
    end_link: str = "vx300s/ee_gripper_link",
) -> ViperXModel:
    """Load the validated full ViperX URDF as a six-DOF FK interface."""

    import roboticstoolbox as rtb

    path = Path(urdf_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    config = ViperXModelConfig(
        urdf_path=path,
        base_link=base_link,
        end_link=end_link,
    )
    robot_model = rtb.ERobot.URDF(str(path))
    return ViperXModel(robot_model, config)
