"""MPLib expert for the seven-dimensional ViperX task contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class PickPlacePhase:
    name: str
    tcp_pose: np.ndarray | None
    gripper_scalar: float
    hold_policy_steps: int = 1
    joint_target_rad: np.ndarray | None = None


def _as_transform(position: np.ndarray, orientation_wxyz: Iterable[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
    if wxyz.shape != (4,):
        raise ValueError("tcp_orientation_wxyz must have shape (4,)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_quat(
        [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    ).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def tcp_target_to_hand_target(
    base_to_tcp_target: np.ndarray, hand_to_tcp: np.ndarray
) -> np.ndarray:
    """Convert a task-space TCP goal to MPLib's accepted hand-link goal."""
    base_to_tcp_target = np.asarray(base_to_tcp_target, dtype=np.float64)
    hand_to_tcp = np.asarray(hand_to_tcp, dtype=np.float64)
    if base_to_tcp_target.shape != (4, 4) or hand_to_tcp.shape != (4, 4):
        raise ValueError("TCP and hand transforms must both have shape (4, 4)")
    return base_to_tcp_target @ np.linalg.inv(hand_to_tcp)


def build_pick_place_sequence(
    item_pose: np.ndarray,
    expert_config: dict[str, Any],
    *,
    home_q_rad: Iterable[float],
) -> tuple[PickPlacePhase, ...]:
    """Build the fixed pick/place sequence ending at the configured home q."""
    item_pose = np.asarray(item_pose, dtype=np.float64)
    if item_pose.shape != (4, 4):
        raise ValueError("item_pose must have shape (4, 4)")
    item_position = item_pose[:3, 3]
    grasp_offset = np.asarray(expert_config["grasp_offset_m"], dtype=np.float64)
    pregrasp_offset = np.asarray(
        expert_config["pregrasp_offset_m"], dtype=np.float64
    )
    lift_offset = np.asarray(expert_config["lift_offset_m"], dtype=np.float64)
    place_position = np.asarray(
        expert_config["place_item_position_m"], dtype=np.float64
    )
    place_approach_offset = np.asarray(
        expert_config["place_approach_offset_m"], dtype=np.float64
    )
    retreat_offset = np.asarray(
        expert_config["retreat_offset_m"], dtype=np.float64
    )
    for name, value in {
        "grasp_offset_m": grasp_offset,
        "pregrasp_offset_m": pregrasp_offset,
        "lift_offset_m": lift_offset,
        "place_item_position_m": place_position,
        "place_approach_offset_m": place_approach_offset,
        "retreat_offset_m": retreat_offset,
    }.items():
        if value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,)")

    orientation = expert_config["tcp_orientation_wxyz"]
    grasp_position = item_position + grasp_offset
    place_tcp_position = place_position + grasp_offset
    close_hold = int(expert_config.get("gripper_settle_policy_steps", 5))
    if close_hold < 1:
        raise ValueError("gripper_settle_policy_steps must be positive")
    home_q_rad = np.asarray(home_q_rad, dtype=np.float64)
    if home_q_rad.shape != (6,) or not np.all(np.isfinite(home_q_rad)):
        raise ValueError("home_q_rad must contain six finite arm joint angles")
    return (
        PickPlacePhase(
            "pregrasp",
            _as_transform(grasp_position + pregrasp_offset, orientation),
            1.0,
        ),
        PickPlacePhase(
            "approach", _as_transform(grasp_position, orientation), 1.0
        ),
        PickPlacePhase("close", None, 0.0, close_hold),
        PickPlacePhase(
            "lift", _as_transform(grasp_position + lift_offset, orientation), 0.0
        ),
        PickPlacePhase(
            "place_approach",
            _as_transform(place_tcp_position + place_approach_offset, orientation),
            0.0,
        ),
        PickPlacePhase(
            "place", _as_transform(place_tcp_position, orientation), 0.0
        ),
        PickPlacePhase("open", None, 1.0, close_hold),
        PickPlacePhase(
            "retreat",
            _as_transform(place_tcp_position + retreat_offset, orientation),
            1.0,
        ),
        PickPlacePhase(
            "home",
            None,
            1.0,
            joint_target_rad=home_q_rad.copy(),
        ),
    )


def _matrix_to_mplib_pose(transform: np.ndarray, mplib_module: Any) -> Any:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return mplib_module.Pose(
        p=transform[:3, 3], q=[xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    )


class ViperXPickAndPlaceController:
    """Plan one TCP phase at a time and emit only seven-dimensional actions."""

    def __init__(
        self,
        *,
        phases: Iterable[PickPlacePhase],
        hand_to_tcp: np.ndarray,
        urdf_path: Path | str,
        srdf_path: Path | str | None,
        hand_link: str,
        policy_dt: float,
        joint_target_tolerance_rad: float = 0.02,
        base_pose_world: np.ndarray | None = None,
        joint_vel_limits: Iterable[float] | None = None,
        collision_points: np.ndarray | None = None,
        collision_resolution_m: float = 0.02,
    ) -> None:
        import mplib

        planner_args: dict[str, Any] = {
            "urdf": str(urdf_path),
            "move_group": str(hand_link),
        }
        if srdf_path not in (None, ""):
            planner_args["srdf"] = str(srdf_path)
        if joint_vel_limits is not None:
            planner_args["joint_vel_limits"] = np.asarray(
                joint_vel_limits, dtype=np.float64
            )
        self.planner = mplib.Planner(**planner_args)
        resolved_base_pose = (
            np.eye(4, dtype=np.float64)
            if base_pose_world is None
            else np.asarray(base_pose_world, dtype=np.float64)
        )
        if resolved_base_pose.shape != (4, 4) or not np.all(
            np.isfinite(resolved_base_pose)
        ):
            raise ValueError("base_pose_world must be a finite 4x4 matrix")
        self.planner.set_base_pose(
            _matrix_to_mplib_pose(resolved_base_pose, mplib)
        )
        if collision_points is not None and len(collision_points):
            self.planner.update_point_cloud(
                np.asarray(collision_points, dtype=np.float64),
                resolution=float(collision_resolution_m),
            )

        self._mplib = mplib
        self.phases = tuple(phases)
        self.hand_to_tcp = np.asarray(hand_to_tcp, dtype=np.float64)
        self.policy_dt = float(policy_dt)
        self.joint_target_tolerance_rad = float(joint_target_tolerance_rad)
        self.phase_index = 0
        self._plan: np.ndarray | None = None
        self._waypoint_index = 0
        self._last_arm_target: np.ndarray | None = None
        self._hold_remaining: int | None = None
        self.failed_reason: str | None = None

    @property
    def done(self) -> bool:
        return self.phase_index >= len(self.phases) or self.failed_reason is not None

    @property
    def phase_name(self) -> str:
        return "done" if self.phase_index >= len(self.phases) else self.phases[
            self.phase_index
        ].name

    def _advance_phase(self) -> None:
        self.phase_index += 1
        self._plan = None
        self._waypoint_index = 0
        self._last_arm_target = None
        self._hold_remaining = None

    def _plan_phase(self, phase: PickPlacePhase, current_arm: np.ndarray) -> bool:
        current_qpos = self.planner.pad_move_group_qpos(current_arm)
        try:
            if phase.joint_target_rad is not None:
                goal_qpos = self.planner.pad_move_group_qpos(
                    phase.joint_target_rad
                )
                result = self.planner.plan_qpos(
                    [goal_qpos], current_qpos, time_step=self.policy_dt
                )
            else:
                if phase.tcp_pose is None:
                    raise ValueError(f"{phase.name}: phase has no motion target")
                hand_target = tcp_target_to_hand_target(
                    phase.tcp_pose, self.hand_to_tcp
                )
                target_pose = _matrix_to_mplib_pose(hand_target, self._mplib)
                result = self.planner.plan_pose(
                    target_pose, current_qpos, time_step=self.policy_dt
                )
        except Exception as error:
            self.failed_reason = f"{phase.name}: {error}"
            return False
        if result.get("status") != "Success":
            self.failed_reason = f"{phase.name}: {result.get('status', 'planning failed')}"
            return False
        positions = np.asarray(result["position"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] < 6 or len(positions) == 0:
            self.failed_reason = f"{phase.name}: invalid MPLib trajectory shape"
            return False
        self._plan = positions[:, :6]
        return True

    def forward(
        self, current_semantic_state: Iterable[float]
    ) -> tuple[np.ndarray, bool, dict[str, Any]]:
        state = np.asarray(current_semantic_state, dtype=np.float64)
        if state.shape != (7,):
            raise ValueError("current_semantic_state must have shape (7,)")
        if self.failed_reason is not None:
            return state.copy(), True, {
                "status": "failed",
                "reason": self.failed_reason,
            }
        if self.phase_index >= len(self.phases):
            return state.copy(), True, {"status": "done", "phase": "done"}

        phase = self.phases[self.phase_index]
        if phase.tcp_pose is None and phase.joint_target_rad is None:
            if self._hold_remaining is None:
                self._hold_remaining = phase.hold_policy_steps
            action = np.concatenate([state[:6], [phase.gripper_scalar]])
            self._hold_remaining -= 1
            if self._hold_remaining <= 0:
                self._advance_phase()
            return action, False, {"status": "running", "phase": phase.name}

        if self._plan is None and not self._plan_phase(phase, state[:6]):
            return state.copy(), True, {
                "status": "failed",
                "reason": self.failed_reason,
                "phase": phase.name,
            }

        if self._last_arm_target is not None:
            error = np.max(np.abs(state[:6] - self._last_arm_target))
            if error <= self.joint_target_tolerance_rad:
                self._waypoint_index += 1
                self._last_arm_target = None

        assert self._plan is not None
        if self._waypoint_index >= len(self._plan):
            completed_phase = phase.name
            self._advance_phase()
            return np.concatenate([state[:6], [phase.gripper_scalar]]), False, {
                "status": "running",
                "phase": completed_phase,
            }

        if self._last_arm_target is None:
            self._last_arm_target = self._plan[self._waypoint_index].copy()
        action = np.concatenate(
            [self._last_arm_target, [phase.gripper_scalar]]
        )
        return action, False, {"status": "running", "phase": phase.name}
