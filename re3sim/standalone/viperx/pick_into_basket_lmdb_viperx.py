"""ViperX pick-and-place runtime entry.

This file intentionally contains only the accepted runtime boundary.  Stage 9
scene-comparison code and Stage 10 articulation/FK/gripper validation code have
been removed; future task, expert and LMDB collection logic builds on this
entry.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import yaml


DEFAULT_CONFIG = Path(
    "configs/viperx/pick_into_basket/collect_data_viperx.yaml"
)
VIPERX_ARM_JOINT_ORDER = (
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
)
VIPERX_DEEP_BLACK_RGB = (0.015, 0.015, 0.015)


def _base_pose(position: Any, orientation: Any) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(position, dtype=np.float64)
    orientation = np.asarray(orientation, dtype=np.float64)
    norm = np.linalg.norm(orientation)
    if (
        position.shape != (3,)
        or orientation.shape != (4,)
        or not np.all(np.isfinite(position))
        or not np.all(np.isfinite(orientation))
        or norm == 0.0
    ):
        raise ValueError("robot base pose must be finite [xyz] and [wxyz]")
    return position.copy(), orientation / norm


def _sample_robot_base_pose(
    robot_params: dict[str, Any], rng: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    position, orientation = _base_pose(
        robot_params["position_base"], robot_params["orientation_wxyz"]
    )
    randomization = robot_params.get("base_randomization", {})
    if not randomization.get("enabled", False):
        return position, orientation
    xy_range = np.asarray(
        randomization["position_xy_half_range_m"], dtype=np.float64
    )
    yaw_range = float(randomization["yaw_half_range_deg"])
    if xy_range.shape != (2,) or np.any(xy_range < 0) or yaw_range < 0:
        raise ValueError("base randomization ranges must be non-negative")
    rng = np.random.default_rng() if rng is None else rng
    position[:2] += rng.uniform(-xy_range, xy_range)
    yaw = np.deg2rad(rng.uniform(-yaw_range, yaw_range))
    orientation = (
        Rotation.from_euler("z", yaw)
        * Rotation.from_quat(orientation, scalar_first=True)
    ).as_quat(scalar_first=True)
    return position, orientation


def _resolve_robot_base_pose(
    robot_params: dict[str, Any],
    *,
    collect_one: bool,
    replay_path: Path | None,
    rng: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    import pickle

    nominal = _base_pose(
        robot_params["position_base"], robot_params["orientation_wxyz"]
    )
    if replay_path is not None:
        import lmdb

        path = replay_path.expanduser().resolve()
        lmdb_path = path / "lmdb" if (path / "lmdb").is_dir() else path
        environment = lmdb.open(
            str(lmdb_path), readonly=True, lock=False, readahead=False
        )
        try:
            with environment.begin() as transaction:
                payload = transaction.get(b"json_data")
        finally:
            environment.close()
        metadata = pickle.loads(payload) if payload else {}
        pose = metadata.get("robot_base_pose_world")
        if pose:
            return (
                *_base_pose(pose["position_m"], pose["orientation_wxyz"]),
                "replay",
            )
        return (*nominal, "legacy_fallback")
    randomization = robot_params.get("base_randomization", {})
    if collect_one and randomization.get("enabled"):
        return (*_sample_robot_base_pose(robot_params, rng), "sampled")
    return (*nominal, "config")


@dataclass
class EpisodeBuffer:
    """In-memory episode; LMDB is created only after task success."""

    diagnostic: bool = True
    qpos: list[np.ndarray] = field(default_factory=list)
    qvel: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    tcp_poses: list[np.ndarray] = field(default_factory=list)
    images: dict[str, list[np.ndarray]] = field(default_factory=dict)
    sim_images: dict[str, list[np.ndarray]] = field(default_factory=dict)
    render_images: dict[str, list[np.ndarray]] = field(default_factory=dict)
    fixed_render_images: dict[str, np.ndarray] = field(default_factory=dict)
    masks: dict[str, list[np.ndarray]] = field(default_factory=dict)
    mask_id_to_labels: dict[str, dict[Any, dict[str, Any]]] = field(
        default_factory=dict
    )

    def append(self, observation: dict[str, Any], action: np.ndarray) -> None:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (7,):
            raise ValueError(f"action must have shape (7,), got {action_array.shape}")
        qpos = np.asarray(observation["qpos"], dtype=np.float32)
        if qpos.shape != (7,):
            raise ValueError("qpos must have shape (7,)")
        self.qpos.append(qpos.copy())
        self.actions.append(action_array.copy())
        for name, image in observation.get("images", {}).items():
            self.images.setdefault(name, []).append(np.asarray(image).copy())
        if not self.diagnostic:
            return
        qvel = np.asarray(observation["qvel"], dtype=np.float32)
        tcp_pose = np.asarray(observation["tcp_pose"], dtype=np.float32)
        if qvel.shape != (7,):
            raise ValueError("qvel must have shape (7,)")
        if tcp_pose.shape != (4, 4):
            raise ValueError("tcp_pose must have shape (4, 4)")
        self.qvel.append(qvel.copy())
        self.tcp_poses.append(tcp_pose.copy())
        for name, image in observation.get("sim_images", {}).items():
            self.sim_images.setdefault(name, []).append(np.asarray(image).copy())
        for name, image in observation.get("render_images", {}).items():
            self.render_images.setdefault(name, []).append(np.asarray(image).copy())
        for name, image in observation.get("fixed_render_images", {}).items():
            self.fixed_render_images.setdefault(name, np.asarray(image).copy())
        for name, mask in observation.get("masks", {}).items():
            self.masks.setdefault(name, []).append(np.asarray(mask).copy())
        for name, labels in observation.get("mask_id_to_labels", {}).items():
            self.mask_id_to_labels.setdefault(name, {}).update(labels)


def _as_float(value: Any) -> float:
    return float(Fraction(value)) if isinstance(value, str) else float(value)


def _positive_int(value: str) -> int:
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return count


def _resolve_path(value: str, *, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_dir / path).resolve()


def _load_transform(value: Any, *, config_dir: Path) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(
            np.load(_resolve_path(value, config_dir=config_dir)),
            dtype=np.float64,
        )
    return np.asarray(value, dtype=np.float64)


def load_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = config_path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    task = config["tasks"][0]
    return config, task, config_path.parent


def with_wrist_camera_extrinsic_override(
    task: dict[str, Any], matrix: Any
) -> dict[str, Any]:
    """Return a task copy with only the hand-mounted wrist extrinsic replaced."""
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("wrist camera extrinsic override must be a finite 4x4 matrix")
    adjusted = deepcopy(task)
    matches = [
        camera
        for camera in adjusted["params"].get("cameras", [])
        if camera.get("name") == "wrist_camera"
        and camera.get("parent_frame") == "hand"
    ]
    if len(matches) != 1:
        raise ValueError(
            "wrist camera extrinsic override requires exactly one "
            "hand-mounted wrist_camera"
        )
    matches[0]["extrinsic"] = transform.tolist()
    return adjusted


def _load_home_settings(
    control: dict[str, Any],
) -> tuple[np.ndarray, float]:
    arm = np.asarray(control["home_q_rad"], dtype=np.float64)
    timeout_s = float(control["home_timeout_s"])
    if arm.shape != (6,) or not np.all(np.isfinite(arm)):
        raise ValueError("control.home_q_rad must be six finite radians")
    if timeout_s <= 0.0:
        raise ValueError("control.home_timeout_s must be positive")
    return np.concatenate([arm, [1.0]]), timeout_s


def _home_arm_error(current_qpos: Any, home_action: Any) -> float:
    current = np.asarray(current_qpos, dtype=np.float64)
    home = np.asarray(home_action, dtype=np.float64)
    if current.shape != (7,) or home.shape != (7,):
        raise ValueError(
            "home convergence requires two seven-dimensional states"
        )
    return float(np.max(np.abs(current[:6] - home[:6])))


def _ensure_viperx_usd(robot_params: dict[str, Any], config_dir: Path) -> Path:
    usd_path = _resolve_path(robot_params["usd_path"], config_dir=config_dir)
    if usd_path.is_file():
        return usd_path

    import omni.kit.commands
    from omni.importer.urdf import _urdf

    urdf_path = _resolve_path(robot_params["urdf_path"], config_dir=config_dir)
    usd_path.parent.mkdir(parents=True, exist_ok=True)

    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = bool(robot_params.get("fix_base", True))
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = False
    import_config.import_inertia_tensor = True
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = (
        _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    )
    import_config.distance_scale = 1.0
    import_config.density = 0.0

    result, _ = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        dest_path=str(usd_path),
    )
    if not result:
        raise RuntimeError(f"Isaac URDF import failed: {urdf_path}")
    return usd_path


def _configure_arm_pd(
    robot: Any,
    *,
    arm_joint_order: tuple[str, ...],
    stiffness_scale: float,
    damping: float,
) -> tuple[int, ...]:
    dof_names = tuple(robot.dof_names)
    arm_indices = tuple(dof_names.index(name) for name in arm_joint_order)
    controller = robot.get_articulation_controller()
    stiffnesses, dampings = controller.get_gains()
    stiffnesses = np.asarray(stiffnesses, dtype=np.float64).copy()
    dampings = np.asarray(dampings, dtype=np.float64).copy()
    indices = np.asarray(arm_indices, dtype=np.int64)
    stiffnesses[indices] *= float(stiffness_scale)
    dampings[indices] = float(damping)
    controller.set_gains(kps=stiffnesses, kds=dampings)
    return arm_indices


def _bind_viperx_deep_black_material(robot_prim_path: str) -> None:
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    material_path = f"{robot_prim_path}/Looks/ViperXDeepBlack"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*VIPERX_DEEP_BLACK_RGB)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )

    mesh_count = 0
    for prim in Usd.PrimRange(robot_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        if str(imageable.ComputeVisibility()) == "invisible":
            continue
        if str(imageable.ComputePurpose()) not in ("default", "render"):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError(f"No visible ViperX meshes found under {robot_prim_path}")
    print(f"viperx_visual_material=deep_black meshes={mesh_count}")


def _add_robot(
    world: Any,
    root_path: str,
    robot_config: dict[str, Any],
    config_dir: Path,
) -> Any:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.utils.prims import create_prim

    params = robot_config["params"]
    usd_path = _ensure_viperx_usd(params, config_dir)
    prim_path = root_path + robot_config["prim_path"]
    create_prim(prim_path=prim_path, usd_path=str(usd_path))
    _bind_viperx_deep_black_material(prim_path)
    return world.scene.add(
        Articulation(
            prim_path=prim_path,
            name=robot_config["name"],
            position=np.asarray(params["position_base"], dtype=np.float64),
            orientation=np.asarray(
                params["orientation_wxyz"], dtype=np.float64
            ),
        )
    )


def _sample_item_spawn_pose(
    params: dict[str, Any], rng: Any | None = None
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(params["position"], dtype=np.float64).copy()
    xy_half_range = params.get("position_xy_half_range_m")
    yaw_range = params.get("yaw_range_rad")
    if xy_half_range is not None or yaw_range is not None:
        rng = np.random.default_rng() if rng is None else rng
    if xy_half_range is not None:
        xy_half_range = np.asarray(xy_half_range, dtype=np.float64)
        if xy_half_range.shape != (2,) or np.any(xy_half_range < 0.0):
            raise ValueError(
                "position_xy_half_range_m must contain two non-negative values"
            )
        position[0] += float(rng.uniform(-xy_half_range[0], xy_half_range[0]))
        position[1] += float(rng.uniform(-xy_half_range[1], xy_half_range[1]))
    if yaw_range is None:
        orientation = np.asarray(params["orientation"], dtype=np.float64)
    else:
        from scipy.spatial.transform import Rotation

        face_index = (
            int(rng.integers(0, 3))
            if bool(params.get("randomize_support_face", False))
            else 0
        )
        yaw = float(rng.uniform(float(yaw_range[0]), float(yaw_range[1])))
        face_euler_xyz = (
            (0.0, 0.0, 0.0),
            (0.0, -np.pi / 2.0, 0.0),
            (np.pi / 2.0, 0.0, 0.0),
        )
        rotation = Rotation.from_euler("z", yaw) * Rotation.from_euler(
            "xyz", face_euler_xyz[face_index]
        )
        xyzw = rotation.as_quat()
        orientation = np.asarray(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64
        )
    return position, orientation


def _reset_episode_poses(
    robot: Any,
    item: Any,
    robot_params: dict[str, Any],
    item_params: dict[str, Any],
    rng: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_position, base_orientation = _sample_robot_base_pose(
        robot_params, rng
    )
    item_position, item_orientation = _sample_item_spawn_pose(item_params, rng)
    robot.set_world_pose(
        position=base_position, orientation=base_orientation
    )
    item.set_world_pose(position=item_position, orientation=item_orientation)
    item.set_linear_velocity(np.zeros(3, dtype=np.float64))
    item.set_angular_velocity(np.zeros(3, dtype=np.float64))
    print(
        f"viperx_episode_base_position_m={base_position.tolist()} "
        f"item_spawn_position_m={item_position.tolist()}",
        flush=True,
    )
    return base_position, base_orientation, item_position, item_orientation


def _add_item(world: Any, root_path: str, item_config: dict[str, Any]) -> Any:
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.prims import create_prim

    params = item_config["params"]
    prim_path = root_path + item_config["prim_path"]
    position, orientation = _sample_item_spawn_pose(params)
    print(
        f"task_item_spawn_position_m={position.tolist()} "
        f"orientation_wxyz={orientation.tolist()}"
    )
    create_prim(prim_path.rsplit("/", 1)[0], prim_type="Xform")
    return world.scene.add(
        DynamicCuboid(
            prim_path=prim_path,
            name=item_config["name"],
            position=position,
            orientation=orientation,
            scale=np.asarray(params["scale"], dtype=np.float64),
            size=1.0,
            color=np.asarray(params["color"], dtype=np.float64),
            mass=float(params["mass"]),
        )
    )


def _add_robot_support(
    world: Any,
    root_path: str,
    support_config: dict[str, Any],
    config_dir: Path,
) -> Any | None:
    if not bool(support_config.get("enabled", False)):
        return None

    from omni.isaac.core.objects import VisualCuboid
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    prim_path = root_path + str(support_config["prim_path"])
    support = world.scene.add(
        VisualCuboid(
            prim_path=prim_path,
            name=str(support_config["name"]),
            position=np.asarray(support_config["position_m"], dtype=np.float64),
            orientation=np.asarray(
                support_config["orientation_wxyz"], dtype=np.float64
            ),
            scale=np.asarray(support_config["size_m"], dtype=np.float64),
            size=1.0,
            color=np.asarray(support_config["color_rgb"], dtype=np.float64),
        )
    )
    texture_path = _resolve_path(
        support_config["texture_path"], config_dir=config_dir
    )
    if not texture_path.is_file():
        raise FileNotFoundError(f"Robot support texture not found: {texture_path}")

    stage = omni.usd.get_context().get_stage()
    binding = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path))
    UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
        binding.GetDirectBindingRel(), UsdShade.Tokens.weakerThanDescendants
    )
    size = np.asarray(support_config["size_m"], dtype=np.float64)
    u, v = size[:2] / (2.0 * float(support_config["texture_size_m"]))
    z = 0.5 + 1.0e-4 / size[2]
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/TexturedTop")
    mesh.CreatePointsAttr(
        [
            Gf.Vec3f(-0.5, -0.5, z),
            Gf.Vec3f(0.5, -0.5, z),
            Gf.Vec3f(0.5, 0.5, z),
            Gf.Vec3f(-0.5, 0.5, z),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    st.Set(
        [
            Gf.Vec2f(0.5 - u, 0.5 - v),
            Gf.Vec2f(0.5 + u, 0.5 - v),
            Gf.Vec2f(0.5 + u, 0.5 + v),
            Gf.Vec2f(0.5 - u, 0.5 + v),
        ]
    )

    material_path = f"{prim_path}/Looks/KitchenWood"
    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        float(support_config["roughness"])
    )
    reader = UsdShade.Shader.Define(stage, f"{material_path}/STReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(texture_path))
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    scale = float(support_config["texture_color_scale"])
    bias = float(support_config["texture_color_bias"])
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(scale, scale, scale, 1.0)
    )
    texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(bias, bias, bias, 0.0)
    )
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.ConnectableAPI(), "result"
    )
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.ConnectableAPI(), "rgb"
    )
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    print(
        "robot_support_table=READY "
        f"size_m={support_config['size_m']} "
        f"position_m={support_config['position_m']}",
        flush=True,
    )
    return support


def _pose_matrix(position: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    ).as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def _policy_substeps(policy_hz: float, physics_dt: float) -> int:
    substeps = int(round(1.0 / (float(policy_hz) * float(physics_dt))))
    if substeps < 1:
        raise ValueError("policy_hz must not exceed the physics rate")
    return substeps


def _move_to_home(
    *,
    world: Any,
    runtime_task: Any,
    home_action: np.ndarray,
    physics_dt: float,
    tolerance_rad: float,
    timeout_s: float,
    render: bool,
) -> None:
    runtime_task.set_desired_action(home_action)
    max_steps = int(np.ceil(float(timeout_s) / float(physics_dt)))
    for step in range(max_steps + 1):
        positions = np.asarray(
            runtime_task.robot.get_joint_positions(), dtype=np.float64
        )
        measured = np.concatenate(
            [positions[list(runtime_task.arm_indices)], [home_action[6]]]
        )
        error = _home_arm_error(measured, home_action)
        if error <= float(tolerance_rad):
            print(
                f"viperx_home=READY steps={step} "
                f"max_error_rad={error:.6f}"
            )
            return
        if step < max_steps:
            runtime_task.advance_control(physics_dt)
            world.step(render=render)
    raise RuntimeError(
        f"ViperX home timed out after {timeout_s:g} s: "
        f"max error={error:.6f} rad"
    )


def _save_successful_episode(
    buffer: EpisodeBuffer,
    *,
    log_root_path: str,
    image_quality: int,
    metadata: dict[str, Any],
    logger: Any | None = None,
) -> Path:
    from real2sim2real.logger.lmdb_logger import LmdbLogger

    if logger is None:
        logger = LmdbLogger(
            log_root_path=log_root_path,
            image_quality=int(image_quality),
        )
    logger.clear()
    previous = set(Path(logger.log_root_path).glob("log-*/lmdb"))
    if len(buffer.qpos) != len(buffer.actions):
        raise ValueError("qpos and action frame counts must match")
    for qpos, action in zip(buffer.qpos, buffer.actions):
        logger.add_scalar_data("observations/qpos", qpos)
        logger.add_scalar_data("action", action)
    if buffer.diagnostic:
        if not (
            len(buffer.qpos) == len(buffer.qvel) == len(buffer.tcp_poses)
        ):
            raise ValueError("diagnostic frame counts must match qpos")
        for qvel, tcp_pose in zip(buffer.qvel, buffer.tcp_poses):
            logger.add_scalar_data("observations/qvel", qvel)
            logger.add_scalar_data("observations/ee_pose", tcp_pose)
    for camera_name, images in buffer.images.items():
        for image in images:
            logger.add_image_data_jpg(
                f"observations/images/{camera_name}", image
            )
    for camera_name, images in buffer.sim_images.items():
        for image in images:
            logger.add_image_data_jpg(
                f"observations/sim_images/{camera_name}", image
            )
    for camera_name, images in buffer.render_images.items():
        for image in images:
            logger.add_image_data_jpg(
                f"observations/render_images/{camera_name}", image
            )
    for camera_name, image in buffer.fixed_render_images.items():
        logger.add_scalar_data(
            f"observations/fix_render_images/{camera_name}", image
        )
    for camera_name, masks in buffer.masks.items():
        for mask in masks:
            logger.add_image_data_png(
                f"observations/mask/{camera_name}", mask
            )
    for key, value in metadata.items():
        logger.add_json_data(key, value)
    if not logger.save():
        raise RuntimeError("LMDB save failed the disk-space check")
    created = sorted(
        set(Path(logger.log_root_path).glob("log-*/lmdb")) - previous
    )
    if len(created) != 1:
        raise RuntimeError(f"Expected one new saved LMDB, found {created}")
    return created[0].parent


def _load_replay_actions(path: Path) -> list[np.ndarray]:
    import lmdb
    import pickle

    path = path.expanduser().resolve()
    lmdb_path = path / "lmdb" if (path / "lmdb").is_dir() else path
    environment = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False
    )
    try:
        with environment.begin() as transaction:
            payload = transaction.get(b"action")
        if payload is None:
            raise KeyError(f"LMDB has no 'action' key: {lmdb_path}")
        actions = [np.asarray(value, dtype=np.float64) for value in pickle.loads(payload)]
    finally:
        environment.close()
    for index, action in enumerate(actions):
        if action.shape != (7,):
            raise ValueError(f"replay action {index} has shape {action.shape}, expected (7,)")
    return actions


def _run_replay(
    *,
    world: Any,
    runtime_task: Any,
    replay_path: Path,
    physics_dt: float,
    policy_hz: float,
    render: bool,
) -> None:
    actions = _load_replay_actions(replay_path)
    substeps = _policy_substeps(policy_hz, physics_dt)
    for action in actions:
        runtime_task.set_desired_action(action)
        for _ in range(substeps):
            runtime_task.advance_control(physics_dt)
            world.step(render=render)
    print(f"viperx_replay_steps={len(actions)}")
    print("viperx_replay_complete=True")


def _run_collection(
    *,
    world: Any,
    runtime_task: Any,
    controller: Any,
    params: dict[str, Any],
    config: dict[str, Any],
    physics_dt: float,
    render: bool,
    base_position: np.ndarray,
    base_orientation: np.ndarray,
    diagnostic: bool = True,
    logger: Any | None = None,
) -> Path | None:
    from real2sim2real.tasks.pick_and_place_viperx import contract_metadata

    policy_hz = float(params["control"]["policy_hz"])
    substeps = _policy_substeps(policy_hz, physics_dt)
    max_policy_steps = int(params["expert"]["max_policy_steps"])
    buffer = EpisodeBuffer(diagnostic=diagnostic)
    final_info: dict[str, Any] = {"status": "not_started"}
    for _ in range(max_policy_steps):
        # The observation is deliberately sampled before applying action_t.
        observation = runtime_task.get_observation()
        action, done, final_info = controller.forward(observation["qpos"])
        if done:
            break
        buffer.append(observation, action)
        runtime_task.set_desired_action(action)
        for _ in range(substeps):
            runtime_task.advance_control(physics_dt)
            world.step(render=render)
    else:
        final_info = {"status": "failed", "reason": "max_policy_steps"}

    success = final_info.get("status") == "done"
    print(f"viperx_collection_frames={len(buffer.actions)}")
    print(f"viperx_collection_controller={final_info}")
    print(f"viperx_collection_success={success}")
    if not success:
        print("viperx_collection_lmdb=NOT_SAVED")
        return None

    metadata = {
        "data_contract": contract_metadata(),
        "camera_names": sorted(buffer.images),
        "observation_before_action": True,
        "robot_base_pose_world": {
            "position_m": base_position.tolist(),
            "orientation_wxyz": base_orientation.tolist(),
        },
        "success": True,
    }
    if diagnostic:
        metadata.update(
            {
                "policy_home": {
                    "q_rad": list(params["control"]["home_q_rad"]),
                    "gripper_scalar": 1.0,
                    "joint_tolerance_rad": float(
                        params["expert"]["joint_target_tolerance_rad"]
                    ),
                },
                "mask_idToLabels": buffer.mask_id_to_labels,
                "item_names": [runtime_task.item_semantic_label],
                "picking_item_name": runtime_task.item_semantic_label,
            }
        )
    output = _save_successful_episode(
        buffer,
        log_root_path=str(config["data_log_root_path"]),
        image_quality=int(params["collection"]["image_quality"]),
        metadata=metadata,
        logger=logger,
    )
    print(f"viperx_collection_lmdb_root={output}")
    return output


def _collect_batch(target_successes: int, run_attempt: Any) -> list[Path]:
    saved: list[Path] = []
    attempt = 0
    while len(saved) < target_successes:
        attempt += 1
        output = run_attempt(attempt)
        if output is not None:
            saved.append(output)
        print(
            f"viperx_batch_progress={len(saved)}/{target_successes} "
            f"attempts={attempt}",
            flush=True,
        )
    return saved


def run(
    config_path: Path,
    *,
    interactive: bool,
    collect_one: bool = False,
    collect_count: int | None = None,
    replay_path: Path | None = None,
    wrist_camera_extrinsic_override: Path | None = None,
) -> None:
    config, task, config_dir = load_config(config_path)
    if wrist_camera_extrinsic_override is not None:
        override_path = wrist_camera_extrinsic_override.expanduser().resolve()
        task = with_wrist_camera_extrinsic_override(
            task, np.load(override_path)
        )
        config["tasks"][0] = task
        print(
            f"wrist_camera_extrinsic_override={override_path}",
            flush=True,
        )
    params = task["params"]
    robot_config = task["robots"][0]
    robot_params = robot_config["params"]
    nominal_robot_params = deepcopy(robot_params)
    base_position, base_orientation, base_source = _resolve_robot_base_pose(
        robot_params,
        collect_one=collect_one,
        replay_path=replay_path,
    )
    robot_params["position_base"] = base_position.tolist()
    robot_params["orientation_wxyz"] = base_orientation.tolist()
    from scipy.spatial.transform import Rotation

    base_yaw_deg = float(
        Rotation.from_quat(base_orientation, scalar_first=True).as_euler(
            "xyz", degrees=True
        )[2]
    )
    print(f"viperx_base_pose_source={base_source}", flush=True)
    print(f"viperx_base_position_m={base_position.tolist()}", flush=True)
    print(f"viperx_base_yaw_deg={base_yaw_deg:.9f}", flush=True)
    if base_source == "legacy_fallback":
        print(
            "viperx_base_pose_compatibility=LMDB_METADATA_MISSING_USING_CONFIG",
            flush=True,
        )

    # Importing Isaac Sim first exposes the bundled omni modules.
    import isaacsim  # noqa: F401
    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": False if interactive else bool(config.get("headless", True)),
            "anti_aliasing": 0,
        }
    )
    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.prims import create_prim

        from real2sim2real.background import Background
        from real2sim2real.controllers.pick_and_place_mplib_viperx import (
            ViperXPickAndPlaceController,
            build_pick_place_sequence,
        )
        from real2sim2real.tasks.pick_and_place_viperx import (
            ViperXPickAndPlaceTask,
        )

        simulator = config.get("simulator", {})
        physics_dt = _as_float(simulator.get("physics_dt", "1/30"))
        rendering_dt = _as_float(
            simulator.get("rendering_dt", simulator.get("physics_dt", "1/30"))
        )
        world = World(
            physics_dt=physics_dt,
            rendering_dt=rendering_dt,
            stage_units_in_meters=1.0,
        )
        create_prim(
            "/World/Light",
            "DomeLight",
            attributes={"inputs:intensity": 1000.0},
        )

        root_path = "/World/env_0"
        create_prim(root_path, prim_type="Xform")
        asset_root = _resolve_path(params["asset_root"], config_dir=config_dir)
        marker_to_world = _load_transform(
            params["marker_to_isaacsim"], config_dir=config_dir
        )
        background = Background(
            str(asset_root), root_path, marker_to_world, task["background"]
        )
        background.load()

        robot_config = task["robots"][0]
        robot = _add_robot(world, root_path, robot_config, config_dir)
        robot_support = _add_robot_support(
            world,
            root_path,
            params.get("robot_support_table", {}),
            config_dir,
        )
        item = _add_item(world, root_path, task["items"][0])
        world.reset()

        control = params["control"]
        arm_joint_order = tuple(
            robot_config["params"].get(
                "arm_joint_order", VIPERX_ARM_JOINT_ORDER
            )
        )
        _configure_arm_pd(
            robot,
            arm_joint_order=arm_joint_order,
            stiffness_scale=float(control["arm_stiffness_scale"]),
            damping=float(control["arm_damping"]),
        )
        runtime_task = ViperXPickAndPlaceTask(
            robot=robot,
            item=item,
            background=background,
            root_path=root_path,
            robot_config=robot_config,
            task_params=params,
            config_dir=config_dir,
            robot_support=robot_support,
        )

        startup_steps = int(simulator.get("startup_steps", 30))
        render_runtime = interactive or bool(config.get("render", True))
        for _ in range(startup_steps):
            runtime_task.advance_control(physics_dt)
            world.step(render=render_runtime)

        home_action, home_timeout_s = _load_home_settings(control)
        _move_to_home(
            world=world,
            runtime_task=runtime_task,
            home_action=home_action,
            physics_dt=physics_dt,
            tolerance_rad=float(
                params["expert"]["joint_target_tolerance_rad"]
            ),
            timeout_s=home_timeout_s,
            render=render_runtime,
        )

        print(f"runtime_task={task['name']}")
        print(f"runtime_robot_dofs={tuple(robot.dof_names)}")
        print(
            "runtime_control="
            f"policy_hz={float(control['policy_hz']):g}, "
            f"max_joint_speed_rad_s={float(control['max_joint_speed_rad_s']):g}"
        )
        print(f"runtime_hand_to_tcp={robot_config['params']['hand_to_tcp']}")
        print(f"runtime_cameras={tuple(runtime_task.cameras)}")
        print("viperx_runtime=READY")

        if replay_path is not None:
            _run_replay(
                world=world,
                runtime_task=runtime_task,
                replay_path=replay_path,
                physics_dt=physics_dt,
                policy_hz=float(control["policy_hz"]),
                render=render_runtime,
            )
        elif collect_one or collect_count is not None:
            planner_config = params["planner"]
            collision_points = None
            if bool(planner_config.get("use_background_point_cloud", False)):
                from real2sim2real.utils.prim import get_points_at_path

                collision_points = get_points_at_path(
                    background.prim_path,
                    relative_frame_prim_path=root_path,
                )
            srdf_value = planner_config.get("srdf_path")
            srdf_path = (
                _resolve_path(srdf_value, config_dir=config_dir)
                if srdf_value
                else None
            )

            def make_controller(
                current_base_position: np.ndarray,
                current_base_orientation: np.ndarray,
            ) -> Any:
                item_position, item_orientation = item.get_world_pose()
                phases = build_pick_place_sequence(
                    _pose_matrix(item_position, item_orientation),
                    params["expert"],
                    home_q_rad=control["home_q_rad"],
                )
                return ViperXPickAndPlaceController(
                    phases=phases,
                    hand_to_tcp=runtime_task.hand_to_tcp,
                    urdf_path=_resolve_path(
                        robot_config["params"]["urdf_path"],
                        config_dir=config_dir,
                    ),
                    srdf_path=srdf_path,
                    hand_link=robot_config["params"]["hand_link"],
                    policy_dt=1.0 / float(control["policy_hz"]),
                    joint_target_tolerance_rad=float(
                        params["expert"]["joint_target_tolerance_rad"]
                    ),
                    base_pose_world=_pose_matrix(
                        current_base_position, current_base_orientation
                    ),
                    joint_vel_limits=planner_config.get("joint_vel_limits"),
                    collision_points=collision_points,
                    collision_resolution_m=float(
                        planner_config["collision_resolution_m"]
                    ),
                )

            if collect_one:
                _run_collection(
                    world=world,
                    runtime_task=runtime_task,
                    controller=make_controller(
                        base_position, base_orientation
                    ),
                    params=params,
                    config=config,
                    physics_dt=physics_dt,
                    render=render_runtime,
                    base_position=base_position,
                    base_orientation=base_orientation,
                )
            else:
                from real2sim2real.logger.lmdb_logger import LmdbLogger

                batch_logger = LmdbLogger(
                    log_root_path=str(config["data_log_root_path"]),
                    image_quality=int(params["collection"]["image_quality"]),
                )
                rng = np.random.default_rng()

                def run_attempt(_: int) -> Path | None:
                    _move_to_home(
                        world=world,
                        runtime_task=runtime_task,
                        home_action=home_action,
                        physics_dt=physics_dt,
                        tolerance_rad=float(
                            params["expert"]["joint_target_tolerance_rad"]
                        ),
                        timeout_s=home_timeout_s,
                        render=render_runtime,
                    )
                    current_base_position, current_base_orientation, _, _ = (
                        _reset_episode_poses(
                            robot,
                            item,
                            nominal_robot_params,
                            task["items"][0]["params"],
                            rng,
                        )
                    )
                    for _ in range(startup_steps):
                        runtime_task.advance_control(physics_dt)
                        world.step(render=render_runtime)
                    return _run_collection(
                        world=world,
                        runtime_task=runtime_task,
                        controller=make_controller(
                            current_base_position, current_base_orientation
                        ),
                        params=params,
                        config=config,
                        physics_dt=physics_dt,
                        render=render_runtime,
                        base_position=current_base_position,
                        base_orientation=current_base_orientation,
                        diagnostic=False,
                        logger=batch_logger,
                    )

                saved = _collect_batch(int(collect_count), run_attempt)
                print(f"viperx_batch_complete={len(saved)}", flush=True)
                print(
                    f"viperx_batch_root={batch_logger.log_root_path}",
                    flush=True,
                )

        if interactive:
            while simulation_app.is_running():
                world.step(render=True)
    finally:
        simulation_app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ViperX pick-and-place scene."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open Isaac Sim and keep the scene running for inspection.",
    )
    parser.add_argument(
        "--wrist-camera-extrinsic-override",
        type=Path,
        help=(
            "Use this derived 4x4 .npy only for the current process; "
            "the YAML and Stage-7 calibration remain unchanged."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-one",
        action="store_true",
        help="Run one deterministic expert episode and save it only on success.",
    )
    mode.add_argument(
        "--collect",
        type=_positive_int,
        metavar="N",
        help="Collect N successful compact episodes in one Isaac Sim process.",
    )
    mode.add_argument(
        "--replay",
        type=Path,
        help="Replay the seven-dimensional action list from an existing LMDB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.config,
        interactive=args.interactive,
        collect_one=args.collect_one,
        collect_count=args.collect,
        replay_path=args.replay,
        wrist_camera_extrinsic_override=(
            args.wrist_camera_extrinsic_override
        ),
    )


if __name__ == "__main__":
    main()
