"""ViperX pick-and-place runtime entry.

This file intentionally contains only the accepted runtime boundary.  Stage 9
scene-comparison code and Stage 10 articulation/FK/gripper validation code have
been removed; future task, expert and LMDB collection logic builds on this
entry.
"""

from __future__ import annotations

import argparse
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


@dataclass
class EpisodeBuffer:
    """In-memory episode; LMDB is created only after task success."""

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
        qvel = np.asarray(observation["qvel"], dtype=np.float32)
        tcp_pose = np.asarray(observation["tcp_pose"], dtype=np.float32)
        if qpos.shape != (7,) or qvel.shape != (7,):
            raise ValueError("qpos and qvel must both have shape (7,)")
        if tcp_pose.shape != (4, 4):
            raise ValueError("tcp_pose must have shape (4, 4)")
        self.qpos.append(qpos.copy())
        self.qvel.append(qvel.copy())
        self.actions.append(action_array.copy())
        self.tcp_poses.append(tcp_pose.copy())
        for name, image in observation.get("images", {}).items():
            self.images.setdefault(name, []).append(np.asarray(image).copy())
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


def _add_item(world: Any, root_path: str, item_config: dict[str, Any]) -> Any:
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.core.utils.prims import create_prim

    params = item_config["params"]
    prim_path = root_path + item_config["prim_path"]
    create_prim(prim_path.rsplit("/", 1)[0], prim_type="Xform")
    return world.scene.add(
        DynamicCuboid(
            prim_path=prim_path,
            name=item_config["name"],
            position=np.asarray(params["position"], dtype=np.float64),
            orientation=np.asarray(params["orientation"], dtype=np.float64),
            scale=np.asarray(params["scale"], dtype=np.float64),
            size=1.0,
            color=np.asarray(params["color"], dtype=np.float64),
            mass=float(params["mass"]),
        )
    )


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
) -> Path:
    from real2sim2real.logger.lmdb_logger import LmdbLogger

    logger = LmdbLogger(
        log_root_path=log_root_path,
        image_quality=int(image_quality),
    )
    for qpos, qvel, action, tcp_pose in zip(
        buffer.qpos, buffer.qvel, buffer.actions, buffer.tcp_poses
    ):
        logger.add_scalar_data("observations/qpos", qpos)
        logger.add_scalar_data("observations/qvel", qvel)
        logger.add_scalar_data("action", action)
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
    episodes = sorted(Path(logger.log_root_path).glob("log-*/lmdb"))
    if len(episodes) != 1:
        raise RuntimeError(f"Expected one saved LMDB, found {episodes}")
    return episodes[0].parent


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
    print(f"viperx_replay_success={runtime_task.is_success()}")


def _run_collection(
    *,
    world: Any,
    runtime_task: Any,
    controller: Any,
    params: dict[str, Any],
    config: dict[str, Any],
    physics_dt: float,
    render: bool,
) -> None:
    from real2sim2real.tasks.pick_and_place_viperx import contract_metadata

    policy_hz = float(params["control"]["policy_hz"])
    substeps = _policy_substeps(policy_hz, physics_dt)
    max_policy_steps = int(params["expert"]["max_policy_steps"])
    buffer = EpisodeBuffer()
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

    success = final_info.get("status") == "done" and runtime_task.is_success()
    print(f"viperx_collection_frames={len(buffer.actions)}")
    print(f"viperx_collection_controller={final_info}")
    print(f"viperx_collection_success={success}")
    if not success:
        print("viperx_collection_lmdb=NOT_SAVED")
        return

    metadata = {
        "data_contract": contract_metadata(),
        "policy_home": {
            "q_rad": list(params["control"]["home_q_rad"]),
            "gripper_scalar": 1.0,
            "joint_tolerance_rad": float(
                params["expert"]["joint_target_tolerance_rad"]
            ),
        },
        "camera_names": sorted(buffer.images),
        "mask_idToLabels": buffer.mask_id_to_labels,
        "item_names": [runtime_task.item_semantic_label],
        "picking_item_name": runtime_task.item_semantic_label,
        "observation_before_action": True,
        "success": True,
    }
    output = _save_successful_episode(
        buffer,
        log_root_path=str(config["data_log_root_path"]),
        image_quality=int(params["collection"]["image_quality"]),
        metadata=metadata,
    )
    print(f"viperx_collection_lmdb_root={output}")


def run(
    config_path: Path,
    *,
    interactive: bool,
    collect_one: bool = False,
    replay_path: Path | None = None,
) -> None:
    config, task, config_dir = load_config(config_path)
    params = task["params"]

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
        item = _add_item(world, root_path, task["items"][0])
        print("runtime_debug=world_reset_begin", flush=True)
        world.reset()
        print("runtime_debug=world_reset_end", flush=True)

        control = params["control"]
        arm_joint_order = tuple(
            robot_config["params"].get(
                "arm_joint_order", VIPERX_ARM_JOINT_ORDER
            )
        )
        print("runtime_debug=configure_arm_pd_begin", flush=True)
        _configure_arm_pd(
            robot,
            arm_joint_order=arm_joint_order,
            stiffness_scale=float(control["arm_stiffness_scale"]),
            damping=float(control["arm_damping"]),
        )
        print("runtime_debug=configure_arm_pd_end", flush=True)
        print("runtime_debug=runtime_task_init_begin", flush=True)
        try:
            runtime_task = ViperXPickAndPlaceTask(
                robot=robot,
                item=item,
                background=background,
                root_path=root_path,
                robot_config=robot_config,
                task_params=params,
                config_dir=config_dir,
            )
        except BaseException as exc:
            import traceback

            print(
                "runtime_debug=runtime_task_init_exception "
                f"type={type(exc).__name__} value={exc!r}",
                flush=True,
            )
            traceback.print_exc()
            raise
        print("runtime_debug=runtime_task_init_end", flush=True)

        startup_steps = int(simulator.get("startup_steps", 30))
        render_runtime = interactive or bool(config.get("render", True))
        print("runtime_debug=startup_loop_begin", flush=True)
        for _ in range(startup_steps):
            runtime_task.advance_control(physics_dt)
            world.step(render=render_runtime)
        print("runtime_debug=startup_loop_end", flush=True)

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
        elif collect_one:
            item_position, item_orientation = item.get_world_pose()
            phases = build_pick_place_sequence(
                _pose_matrix(item_position, item_orientation), params["expert"]
            )
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
            controller = ViperXPickAndPlaceController(
                phases=phases,
                hand_to_tcp=runtime_task.hand_to_tcp,
                urdf_path=_resolve_path(
                    robot_config["params"]["urdf_path"], config_dir=config_dir
                ),
                srdf_path=srdf_path,
                hand_link=robot_config["params"]["hand_link"],
                policy_dt=1.0 / float(control["policy_hz"]),
                joint_target_tolerance_rad=float(
                    params["expert"]["joint_target_tolerance_rad"]
                ),
                joint_vel_limits=planner_config.get("joint_vel_limits"),
                collision_points=collision_points,
                collision_resolution_m=float(
                    planner_config["collision_resolution_m"]
                ),
            )
            _run_collection(
                world=world,
                runtime_task=runtime_task,
                controller=controller,
                params=params,
                config=config,
                physics_dt=physics_dt,
                render=render_runtime,
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--collect-one",
        action="store_true",
        help="Run one deterministic expert episode and save it only on success.",
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
        replay_path=args.replay,
    )


if __name__ == "__main__":
    main()
