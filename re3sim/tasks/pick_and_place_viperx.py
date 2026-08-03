"""ViperX task boundary shared by expert collection and replay.

The public state/action vector is always seven dimensional:
six URDF joint angles in radians followed by one gripper scalar.
Isaac's three gripper joints remain an internal implementation detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ARM_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
GRIPPER_JOINT_ORDER = ("gripper", "left_finger", "right_finger")
SEMANTIC_ACTION_ORDER = ARM_JOINT_ORDER + ("gripper",)
CONTRACT_VERSION = "viperx_joint_gripper_v1"
DYNAMIC_SEMANTIC_LABELS = frozenset({"robot", "item"})
RENDERABLE_PRIM_TYPES = frozenset(
    {"Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule"}
)


def contract_metadata() -> dict[str, Any]:
    return {
        "version": CONTRACT_VERSION,
        "order": list(SEMANTIC_ACTION_ORDER),
        "shape": [7],
        "arm_unit": "radian",
        "gripper": "0=closed,1=open",
    }


def validate_semantic_vector(value: Iterable[float], name: str = "value") -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    if not 0.0 <= vector[6] <= 1.0:
        raise ValueError(f"{name} gripper scalar must be in [0, 1]")
    return vector


def advance_arm_targets(
    current_targets: Iterable[float],
    final_targets: Iterable[float],
    *,
    max_joint_speed_rad_s: float,
    physics_dt: float,
) -> np.ndarray:
    current = np.asarray(current_targets, dtype=np.float64)
    final = np.asarray(final_targets, dtype=np.float64)
    if current.shape != (6,) or final.shape != (6,):
        raise ValueError("arm targets must both have shape (6,)")
    max_delta = float(max_joint_speed_rad_s) * float(physics_dt)
    if max_delta <= 0.0:
        raise ValueError("max_joint_speed_rad_s and physics_dt must be positive")
    return current + np.clip(final - current, -max_delta, max_delta)


def gripper_scalar_to_joint_targets(
    scalar: float,
    *,
    closed_finger_m: float,
    open_finger_m: float,
    prop_rad: float,
) -> np.ndarray:
    scalar = float(scalar)
    if not 0.0 <= scalar <= 1.0:
        raise ValueError("gripper scalar must be in [0, 1]")
    finger_m = float(closed_finger_m) + scalar * (
        float(open_finger_m) - float(closed_finger_m)
    )
    return np.asarray([prop_rad, finger_m, -finger_m], dtype=np.float64)


def gripper_joint_positions_to_scalar(
    positions: Iterable[float],
    *,
    closed_finger_m: float,
    open_finger_m: float,
) -> float:
    joints = np.asarray(positions, dtype=np.float64)
    if joints.shape != (3,):
        raise ValueError("gripper joint positions must have shape (3,)")
    span = float(open_finger_m) - float(closed_finger_m)
    if span <= 0.0:
        raise ValueError("open_finger_m must be greater than closed_finger_m")
    symmetric_finger_m = 0.5 * (joints[1] - joints[2])
    return float(np.clip((symmetric_finger_m - closed_finger_m) / span, 0.0, 1.0))


@dataclass(frozen=True)
class CameraSpec:
    name: str
    parent_frame: str
    parent_to_camera: np.ndarray
    camera_params: tuple[float, float, float, float, int, int]


def compose_gaussian_foreground(
    sim_image: np.ndarray,
    gaussian_image: np.ndarray,
    segmentation: np.ndarray,
    id_to_labels: dict[Any, dict[str, Any]],
    *,
    dynamic_labels: Iterable[str] = DYNAMIC_SEMANTIC_LABELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep visible robot/item pixels and replace everything else with GS."""
    sim = np.asarray(sim_image, dtype=np.uint8)
    gaussian = np.asarray(gaussian_image, dtype=np.uint8)
    segment_ids = np.asarray(segmentation)
    if sim.shape != gaussian.shape or sim.ndim != 3 or sim.shape[2] != 3:
        raise ValueError("sim and Gaussian images must have the same HxWx3 shape")
    if segment_ids.shape != sim.shape[:2]:
        raise ValueError("semantic segmentation must match the image height and width")
    dynamic_label_set = frozenset(dynamic_labels)
    foreground_ids = [
        int(segment_id)
        for segment_id, labels in id_to_labels.items()
        if labels.get("class") in dynamic_label_set
    ]
    foreground_mask = np.isin(segment_ids, foreground_ids).astype(np.uint8)
    composite = np.where(foreground_mask[..., None] != 0, sim, gaussian)
    return composite.astype(np.uint8), foreground_mask


def label_renderable_prims(root_prim: Any, label: str, add_label: Any) -> None:
    """Apply one class label to all renderable geometry below a USD prim."""
    if root_prim.GetTypeName() in RENDERABLE_PRIM_TYPES:
        add_label(root_prim, label)
    for child in root_prim.GetAllChildren():
        label_renderable_prims(child, label, add_label)


def normalize_id_to_labels(
    id_to_labels: dict[Any, dict[str, Any]],
    *,
    dynamic_labels: Iterable[str],
) -> dict[Any, dict[str, Any]]:
    """Use Re3Sim's BACKGROUND class for every non-dynamic semantic ID."""
    dynamic_label_set = frozenset(dynamic_labels)
    normalized: dict[Any, dict[str, Any]] = {}
    for segment_id, labels in id_to_labels.items():
        normalized_labels = dict(labels)
        if normalized_labels.get("class") not in dynamic_label_set:
            normalized_labels["class"] = "BACKGROUND"
        normalized[segment_id] = normalized_labels
    return normalized


def render_gaussian_background(
    background: Any,
    camera_spec: CameraSpec,
    camera_pose: np.ndarray,
    fixed_cache: dict[str, np.ndarray],
) -> np.ndarray:
    """Render moving cameras every frame and cache base-mounted cameras."""
    is_fixed = camera_spec.parent_frame == "base"
    if is_fixed and camera_spec.name in fixed_cache:
        return fixed_cache[camera_spec.name]
    fx, fy, _, _, width, height = camera_spec.camera_params
    image = np.asarray(
        background.render(
            cam_pose=np.asarray(camera_pose, dtype=np.float64),
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            camera_pose_frame="isaacsim",
        ),
        dtype=np.uint8,
    )
    if image.shape != (height, width, 3):
        raise ValueError(
            f"Gaussian render for {camera_spec.name!r} has shape {image.shape}, "
            f"expected {(height, width, 3)}"
        )
    if is_fixed:
        fixed_cache[camera_spec.name] = image
    return image


def _load_matrix(value: Any, config_dir: Path) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        matrix = np.load(path.resolve())
    else:
        matrix = value
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera extrinsic must be a finite 4x4 transform")
    return matrix


def load_enabled_camera_specs(
    camera_configs: Iterable[dict[str, Any]], *, config_dir: Path
) -> list[CameraSpec]:
    """Load configured cameras; an empty extrinsic keeps an optional slot disabled."""
    enabled: list[CameraSpec] = []
    for config in camera_configs:
        extrinsic = config.get("extrinsic")
        camera_params = config.get("camera_params")
        if extrinsic in (None, "") or camera_params in (None, ""):
            continue
        parent_frame = str(config.get("parent_frame", "base"))
        if parent_frame not in {"base", "hand"}:
            raise ValueError("camera parent_frame must be 'base' or 'hand'")
        params = tuple(float(value) for value in camera_params)
        if len(params) != 6:
            raise ValueError("camera_params must be [fx, fy, cx, cy, width, height]")
        enabled.append(
            CameraSpec(
                name=str(config["name"]),
                parent_frame=parent_frame,
                parent_to_camera=_load_matrix(extrinsic, Path(config_dir)),
                camera_params=(
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    int(params[4]),
                    int(params[5]),
                ),
            )
        )
    return enabled


def _matrix_pose_wxyz(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return transform[:3, 3].copy(), np.asarray(
        [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64
    )


def _find_link_prim_path(robot_prim_path: str, urdf_link_name: str) -> str:
    """Resolve Isaac's sanitized or nested prim path for one URDF link."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    full_suffix = urdf_link_name.strip("/")
    sanitized_suffix = full_suffix.replace("/", "_")
    leaf_suffix = full_suffix.rsplit("/", 1)[-1]
    candidates: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(robot_prim_path + "/"):
            continue
        relative = path[len(robot_prim_path) + 1 :]
        if relative in {full_suffix, sanitized_suffix} or relative.endswith(
            "/" + leaf_suffix
        ):
            candidates.append(path)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Isaac prim for URDF link {urdf_link_name!r}, "
            f"found {candidates}"
        )
    return candidates[0]


class ViperXPickAndPlaceTask:
    """Small Isaac-facing adapter; policy/controller code only sees the 7-D contract."""

    def __init__(
        self,
        *,
        robot: Any,
        item: Any,
        background: Any,
        root_path: str,
        robot_config: dict[str, Any],
        task_params: dict[str, Any],
        config_dir: Path,
    ) -> None:
        self.robot = robot
        self.item = item
        self.background = background
        self.root_path = root_path
        self.robot_config = robot_config
        self.task_params = task_params
        self.item_semantic_label = str(getattr(item, "name", "item"))
        self.dynamic_semantic_labels = frozenset(
            {"robot", self.item_semantic_label}
        )
        self.hand_to_tcp = np.asarray(
            robot_config["params"]["hand_to_tcp"], dtype=np.float64
        )
        if self.hand_to_tcp.shape != (4, 4):
            raise ValueError("hand_to_tcp must have shape (4, 4)")

        dof_names = tuple(robot.dof_names)
        self.arm_indices = tuple(dof_names.index(name) for name in ARM_JOINT_ORDER)
        self.gripper_indices = tuple(
            dof_names.index(name) for name in GRIPPER_JOINT_ORDER
        )
        self.hand_prim_path = _find_link_prim_path(
            root_path + robot_config["prim_path"],
            robot_config["params"]["hand_link"],
        )
        self.gripper_config = robot_config["params"]["gripper"]
        self.max_joint_speed_rad_s = float(
            task_params["control"]["max_joint_speed_rad_s"]
        )
        positions = np.asarray(robot.get_joint_positions(), dtype=np.float64)
        self._commanded_arm = positions[list(self.arm_indices)].copy()
        startup_scalar = float(self.gripper_config.get("startup_scalar", 1.0))
        self._desired_action = np.concatenate(
            [self._commanded_arm, [startup_scalar]]
        )
        self.camera_specs = load_enabled_camera_specs(
            task_params.get("cameras", []), config_dir=config_dir
        )
        self._fixed_gs_images: dict[str, np.ndarray] = {}
        self._label_scene_semantics()
        self.cameras = self._create_cameras()

    def _label_scene_semantics(self) -> None:
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.semantics import add_update_semantics

        def add_class_label(prim: Any, label: str) -> None:
            add_update_semantics(prim, semantic_label=label, type_label="class")

        label_renderable_prims(
            get_prim_at_path(self.robot.prim_path), "robot", add_class_label
        )
        label_renderable_prims(
            get_prim_at_path(self.item.prim_path),
            self.item_semantic_label,
            add_class_label,
        )
        label_renderable_prims(
            get_prim_at_path(self.background.prim_path),
            "BACKGROUND",
            add_class_label,
        )

    def _create_cameras(self) -> dict[str, Any]:
        from real2sim2real.utils.items import create_camera

        cameras: dict[str, Any] = {}
        for spec in self.camera_specs:
            parent_path = (
                self.hand_prim_path if spec.parent_frame == "hand" else self.root_path
            )
            position, orientation = _matrix_pose_wxyz(spec.parent_to_camera)
            camera = create_camera(
                spec.name,
                f"{parent_path}/Sensors/{spec.name}",
                position,
                orientation,
                spec.camera_params,
            )
            camera.add_semantic_segmentation_to_frame()
            cameras[spec.name] = camera
        return cameras

    def set_desired_action(self, action: Iterable[float]) -> None:
        self._desired_action = validate_semantic_vector(action, "action").copy()

    def advance_control(self, physics_dt: float) -> None:
        from omni.isaac.core.utils.types import ArticulationAction

        self._commanded_arm = advance_arm_targets(
            self._commanded_arm,
            self._desired_action[:6],
            max_joint_speed_rad_s=self.max_joint_speed_rad_s,
            physics_dt=physics_dt,
        )
        targets = np.asarray(self.robot.get_joint_positions(), dtype=np.float64).copy()
        targets[list(self.arm_indices)] = self._commanded_arm
        targets[list(self.gripper_indices)] = gripper_scalar_to_joint_targets(
            self._desired_action[6],
            closed_finger_m=float(self.gripper_config["closed_finger_m"]),
            open_finger_m=float(self.gripper_config["open_finger_m"]),
            prop_rad=float(self.gripper_config.get("prop_rad", 0.0)),
        )
        self.robot.apply_action(ArticulationAction(joint_positions=targets))

    def get_observation(self) -> dict[str, Any]:
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.transformations import get_relative_transform

        positions = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        velocities = np.asarray(self.robot.get_joint_velocities(), dtype=np.float64)
        gripper_positions = positions[list(self.gripper_indices)]
        gripper_scalar = gripper_joint_positions_to_scalar(
            gripper_positions,
            closed_finger_m=float(self.gripper_config["closed_finger_m"]),
            open_finger_m=float(self.gripper_config["open_finger_m"]),
        )
        finger_span = float(self.gripper_config["open_finger_m"]) - float(
            self.gripper_config["closed_finger_m"]
        )
        gripper_velocity = float(velocities[self.gripper_indices[1]] / finger_span)
        qpos = np.concatenate([positions[list(self.arm_indices)], [gripper_scalar]])
        qvel = np.concatenate(
            [velocities[list(self.arm_indices)], [gripper_velocity]]
        )
        base_to_hand = get_relative_transform(
            get_prim_at_path(self.hand_prim_path),
            get_prim_at_path(self.robot.prim_path),
        )
        images: dict[str, np.ndarray] = {}
        sim_images: dict[str, np.ndarray] = {}
        render_images: dict[str, np.ndarray] = {}
        fixed_render_images: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        mask_id_to_labels: dict[str, dict[Any, dict[str, Any]]] = {}
        for spec in self.camera_specs:
            name = spec.name
            camera = self.cameras[name]
            rgba = camera.get_rgba()
            if isinstance(rgba, np.ndarray) and rgba.ndim == 3:
                sim_image = np.asarray(rgba[:, :, :3], dtype=np.uint8)
            else:
                width = spec.camera_params[4]
                height = spec.camera_params[5]
                sim_image = np.zeros((height, width, 3), dtype=np.uint8)
            camera_pose = get_relative_transform(
                get_prim_at_path(camera.prim_path),
                get_prim_at_path(self.root_path),
            )
            gaussian_image = render_gaussian_background(
                self.background,
                spec,
                camera_pose,
                self._fixed_gs_images,
            )
            segment_data = camera._custom_annotators[
                "semantic_segmentation"
            ].get_data()
            segment_ids = np.asarray(segment_data["data"])
            id_to_labels = normalize_id_to_labels(
                segment_data["info"].get("idToLabels", {}),
                dynamic_labels=self.dynamic_semantic_labels,
            )
            composite, _ = compose_gaussian_foreground(
                sim_image,
                gaussian_image,
                segment_ids,
                id_to_labels,
                dynamic_labels=self.dynamic_semantic_labels,
            )
            images[name] = composite
            sim_images[name] = sim_image
            masks[name] = segment_ids
            mask_id_to_labels[name] = id_to_labels
            if spec.parent_frame == "base":
                fixed_render_images[name] = gaussian_image
            else:
                render_images[name] = gaussian_image
        item_position, item_orientation = self.item.get_world_pose()
        return {
            "qpos": qpos.astype(np.float32),
            "qvel": qvel.astype(np.float32),
            "tcp_pose": (base_to_hand @ self.hand_to_tcp).astype(np.float32),
            "item_position": np.asarray(item_position, dtype=np.float64),
            "item_orientation_wxyz": np.asarray(item_orientation, dtype=np.float64),
            "images": images,
            "sim_images": sim_images,
            "render_images": render_images,
            "fixed_render_images": fixed_render_images,
            "masks": masks,
            "mask_id_to_labels": mask_id_to_labels,
        }

    def is_success(self) -> bool:
        target = np.asarray(
            self.task_params["expert"]["place_item_position_m"], dtype=np.float64
        )
        tolerance = float(self.task_params["expert"].get("success_tolerance_m", 0.05))
        position, _ = self.item.get_world_pose()
        return bool(
            np.linalg.norm(np.asarray(position)[:2] - target[:2]) <= tolerance
        )
