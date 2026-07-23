"""Compute the fixed ChArUco marker pose in the ViperX base frame.

This is the ViperX replacement for ``get_marker2base_aruco.py``.  It keeps the
original per-frame computation:

``T_base_marker = T_base_hand @ T_hand_camera @ T_camera_marker``.

The robot-specific boundary uses the measured six-joint ``q_rad`` samples,
the dataset's Stage-4 mapping snapshot, and the accepted full-URDF
``ViperXModel``.  The saved ``poses/`` records are not solver inputs.  This is
an offline script and never connects to robot hardware.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


SCRIPT_DIR = Path(__file__).resolve().parent
REAL_DEPLOYMENT_DIR = SCRIPT_DIR.parent
REAL_DEPLOYMENT_DIR_TEXT = str(REAL_DEPLOYMENT_DIR)
if REAL_DEPLOYMENT_DIR_TEXT not in sys.path:
    sys.path.insert(0, REAL_DEPLOYMENT_DIR_TEXT)

from viperx_adapter import ViperXUrdfMapping
from viperx_model import load_viperx_model


EXPECTED_ARM_JOINT_COUNT = 6
DEFAULT_MAPPING_FILENAME = "viperx_urdf_mapping.json"
CHARUCO_BOARD_SHAPE = (5, 5)
CHARUCO_SQUARE_LENGTH_M = 0.036
CHARUCO_MARKER_LENGTH_M = 0.027


def resolve_mapping_path(
    data_root: str | Path,
    mapping_path: str | Path | None,
) -> Path:
    """Return the Stage-4 mapping snapshot associated with the dataset."""

    data_root = Path(data_root).expanduser().resolve()
    candidate = (
        data_root / "configs" / DEFAULT_MAPPING_FILENAME
        if mapping_path is None
        else Path(mapping_path).expanduser()
    )
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"ViperX mapping file not found: {candidate}. "
            "Pass --mapping if the dataset snapshot has a different path."
        )
    return candidate


def load_mapping_metadata(mapping_path: str | Path) -> dict[str, Any]:
    """Load the mapping fields needed to reproduce capture-time FK."""

    mapping_path = Path(mapping_path).expanduser().resolve()
    with mapping_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    if not isinstance(metadata, dict):
        raise ValueError(f"Mapping root must be a JSON object: {mapping_path}")

    required = {"urdf_path", "joint_order", "base_link", "end_link"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Mapping is missing solver metadata: {sorted(missing)}")
    metadata["_source_path"] = str(mapping_path)
    return metadata


def resolve_urdf_path(
    mapping_metadata: dict[str, Any],
    urdf_path: str | Path | None,
) -> Path:
    """Resolve the capture-time URDF, requiring an override if it moved."""

    if urdf_path is not None:
        candidate = Path(urdf_path).expanduser()
    else:
        candidate = Path(str(mapping_metadata["urdf_path"])).expanduser()
        if not candidate.is_absolute():
            source_path = Path(str(mapping_metadata["_source_path"]))
            candidate = source_path.parent / candidate

    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"ViperX URDF file not found: {candidate}. "
            "Pass --urdf with the local copy of the accepted full URDF."
        )
    return candidate


def validate_model_contract(mapping: Any, model: Any) -> None:
    """Require the dataset mapping and model to describe the same FK chain."""

    if tuple(mapping.joint_order) != tuple(model.arm_joint_names):
        raise ValueError(
            "Mapping/model joint_order mismatch: "
            f"{tuple(mapping.joint_order)} != {tuple(model.arm_joint_names)}."
        )
    if mapping.base_link != model.base_link:
        raise ValueError(
            f"Mapping/model base_link mismatch: {mapping.base_link!r} != "
            f"{model.base_link!r}."
        )
    if mapping.end_link != model.end_link:
        raise ValueError(
            f"Mapping/model end_link mismatch: {mapping.end_link!r} != "
            f"{model.end_link!r}."
        )
    if int(model.n) != EXPECTED_ARM_JOINT_COUNT:
        raise ValueError(
            f"Expected a six-joint ViperX model, got model.n={model.n}."
        )


def joint_to_hand(joints: Sequence[float], model: Any) -> np.ndarray:
    """Return ``T_base_hand`` from one measured six-joint ViperX sample."""

    q_arm = np.asarray(joints, dtype=np.float64)
    if q_arm.shape != (int(model.n),):
        raise ValueError(
            f"Expected measured joint shape {(int(model.n),)}, got {q_arm.shape}."
        )
    if not np.all(np.isfinite(q_arm)):
        raise ValueError("Measured joints contain NaN or infinity.")

    q_full = np.asarray(model.full_joint_defaults, dtype=np.float64).copy()
    arm_joint_indices = tuple(model.arm_joint_indices)
    if q_full.ndim != 1 or len(arm_joint_indices) != int(model.n):
        raise ValueError("ViperX model has an invalid full-joint expansion contract.")
    q_full[list(arm_joint_indices)] = q_arm

    pose = model.robot_model.fkine(
        q_full,
        start=model.base_link,
        end=model.end_link,
    )
    return model.validate_transform(getattr(pose, "A", pose))


def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, charuco_dict)

    if len(corners) > 0:
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if charuco_ids is not None and len(charuco_corners) > 3:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                board,
                intrinsics_matrix,
                dist_coeffs,
                None,
                None,
            )
            if valid:
                R_target2cam = cv2.Rodrigues(rvec)[0]
                t_target2cam = tvec.reshape(3, 1)
                target2cam = np.eye(4)
                target2cam[:3, :3] = R_target2cam
                target2cam[:3, 3] = t_target2cam.reshape(-1)
                return np.linalg.inv(target2cam)
    return None


def create_camera_model(size=0.1):
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0, 0, 0]
    )


def show_pose(camera_pose, size=0.1):
    camera_pose = np.array(camera_pose)
    camera_pose[:3, :3] = camera_pose[:3, :3] / np.abs(
        np.linalg.det(camera_pose[:3, :3])
    ) ** (1 / 3)
    camera_model = create_camera_model(size)
    camera_model.transform(camera_pose)
    return camera_model


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average translations and rotations without averaging 4x4 entries."""

    transform_array = np.asarray(transforms, dtype=np.float64)
    if transform_array.ndim != 3 or transform_array.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected transform array with shape (N, 4, 4), got "
            f"{transform_array.shape}."
        )
    if transform_array.shape[0] == 0:
        raise ValueError("No valid marker-to-base transforms were produced.")

    mean_transform = np.eye(4, dtype=np.float64)
    mean_transform[:3, :3] = R.from_matrix(
        transform_array[:, :3, :3]
    ).mean().as_matrix()
    mean_transform[:3, 3] = np.mean(transform_array[:, :3, 3], axis=0)
    return mean_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute marker-to-base for a ViperX shooting dataset."
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="ViperX shooting dataset root.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help=(
            "Stage-4 mapping snapshot. Defaults to "
            "<data_root>/configs/viperx_urdf_mapping.json."
        ),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Local accepted vx300s_full.urdf; overrides mapping metadata path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    mapping_path = resolve_mapping_path(data_root, args.mapping)
    mapping_metadata = load_mapping_metadata(mapping_path)
    mapping = ViperXUrdfMapping.load(mapping_path)
    urdf_path = resolve_urdf_path(mapping_metadata, args.urdf)

    model = load_viperx_model(
        urdf_path=urdf_path,
        base_link=mapping.base_link,
        end_link=mapping.end_link,
    )
    validate_model_contract(mapping, model)

    intrinsic_path = data_root / "rgb_intrinsics.npz"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(f"RGB intrinsics not found: {intrinsic_path}")
    with np.load(intrinsic_path) as intrinsic_zip:
        fx = float(intrinsic_zip["fx"])
        fy = float(intrinsic_zip["fy"])
        cx = float(intrinsic_zip["ppx"])
        cy = float(intrinsic_zip["ppy"])
        rgb_coeffs = np.asarray(intrinsic_zip["coeffs"], dtype=np.float64).reshape(-1)
    intrinsic_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    cam_to_hand_path = data_root / "cam_to_hand_pose.npy"
    if not cam_to_hand_path.is_file():
        raise FileNotFoundError(f"Camera-to-hand pose not found: {cam_to_hand_path}")
    cam_2_ee = model.validate_transform(np.load(cam_to_hand_path))

    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        CHARUCO_BOARD_SHAPE,
        CHARUCO_SQUARE_LENGTH_M,
        CHARUCO_MARKER_LENGTH_M,
        charuco_dict,
    )

    joints_paths = sorted(data_root.glob("joints/joints_*.npy"))
    if not joints_paths:
        raise ValueError(f"No measured ViperX joints found under {data_root / 'joints'}.")

    item_to_show = []
    marker_2_base_list = []
    for joints_path in joints_paths:
        match = re.fullmatch(r"joints_(.+)\.npy", joints_path.name)
        if match is None:
            raise ValueError(f"Unexpected joints filename: {joints_path.name}")
        frame_id = match.group(1)

        image_path = data_root / "rgb" / f"{frame_id}.png"
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"RGB image could not be loaded: {image_path}")

        cam_2_marker = estimate_pose(
            image,
            charuco_dict,
            intrinsic_matrix,
            rgb_coeffs,
            board,
        )
        if cam_2_marker is None:
            print(f"image_{frame_id} has no marker")
            continue

        qpos = np.load(joints_path)
        ee_2_base = joint_to_hand(qpos, model)
        cam_2_base = ee_2_base @ cam_2_ee
        marker_2_cam = np.linalg.inv(cam_2_marker)
        marker_2_base = model.validate_transform(cam_2_base @ marker_2_cam)
        marker_2_base_list.append(marker_2_base)
        item_to_show.append(show_pose(marker_2_base))

    marker_2_base = model.validate_transform(
        average_transforms(marker_2_base_list)
    )
    print(f"Marker to base:\n{marker_2_base}")

    frame_base = show_pose(np.eye(4))
    item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(item_to_show)

    output_path = data_root / "marker_2_base.npy"
    np.save(output_path, marker_2_base)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
