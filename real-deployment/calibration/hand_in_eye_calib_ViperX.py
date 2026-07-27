"""Solve ViperX wrist-camera hand-eye calibration from shooting data.

This is the ViperX replacement for ``hand_in_eye_calib.py``.  The OpenCV
ChArUco and hand-eye algorithm remain unchanged.  Only the Panda-specific
robot boundary is replaced:

* input robot state is the six measured URDF-radian joints written by
  ``hand_in_eye_shooting_ViperX.ipynb``;
* FK uses the accepted ViperX full-URDF model and the dataset's Stage-4 mapping
  metadata for joint order and base/end frames;
* the calibration ``hand`` is ``vx300s/ee_gripper_link`` (or the exact end link
  recorded by that mapping), not a task TCP;
* missing joints are an error.  The Panda TCP -> IK -> hand fallback is not
  meaningful for this dataset and is deliberately removed.

The eight raw actuator values saved by shooting are diagnostic records.  They
are not solver inputs and this offline script never connects to hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REAL_DEPLOYMENT_DIR = SCRIPT_DIR.parent
for import_dir in (SCRIPT_DIR, REAL_DEPLOYMENT_DIR):
    import_dir_text = str(import_dir)
    if import_dir_text not in sys.path:
        sys.path.insert(0, import_dir_text)

from calibration.hand_in_eye import HandinEyeCalibrator
from calibration.utils import read_data
from viperx_adapter import ViperXUrdfMapping
from viperx_model import load_viperx_model


EXPECTED_ARM_JOINT_COUNT = 6
DEFAULT_MAPPING_FILENAME = "viperx_urdf_mapping.json"
CHARUCO_BOARD_SHAPE = (5, 5)
CHARUCO_SQUARE_LENGTH_M = 0.036
CHARUCO_MARKER_LENGTH_M = 0.027
MIN_VALID_SAMPLES_PER_CENTER = 1


def load_rgb_frame_ids(data_root: str | Path) -> list[int]:
    """Return frame IDs in the same filename order used by ``read_data``."""

    rgb_dir = Path(data_root).expanduser().resolve() / "rgb"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB calibration directory not found: {rgb_dir}")

    rgb_paths = sorted(
        (
            path
            for path in rgb_dir.iterdir()
            if path.is_file() and path.suffix == ".png"
        ),
        key=lambda path: path.name,
    )
    frame_ids = []
    for path in rgb_paths:
        try:
            frame_ids.append(int(path.stem))
        except ValueError as exc:
            raise ValueError(
                f"RGB calibration filename must be <integer>.png: {path.name}"
            ) from exc
    return frame_ids


def resolve_mapping_path(
    data_root: str | Path,
    mapping_path: str | Path | None,
) -> Path:
    """Return the exact Stage-4 mapping snapshot associated with the dataset."""

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
    """Load the mapping JSON fields needed to reproduce capture-time FK."""

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


def validate_joint_sample(
    joints: Sequence[float],
    expected_joint_count: int = EXPECTED_ARM_JOINT_COUNT,
) -> np.ndarray:
    """Validate one measured q sample without applying command-target limits."""

    q_arm = np.asarray(joints, dtype=np.float64)
    if q_arm.shape != (expected_joint_count,):
        raise ValueError(
            f"Expected measured joint shape {(expected_joint_count,)}, got {q_arm.shape}."
        )
    if not np.all(np.isfinite(q_arm)):
        raise ValueError("Measured joints contain NaN or infinity.")
    return q_arm


def validate_model_contract(mapping: Any, model: Any) -> None:
    """Require dataset mapping and ViperX model to describe the same FK chain."""

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
    """Return measured ``T_base_hand`` for one six-joint shooting sample.

    ``ViperXModel.fk`` correctly enforces limits for IK and command targets.
    These joints are already measured states, so this read-only FK path checks
    only shape/finite values and preserves an out-of-limit measurement for
    diagnosis instead of discarding it.
    """

    q_arm = validate_joint_sample(joints, expected_joint_count=int(model.n))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve ViperX hand-eye calibration from shooting data."
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

    (
        rgb_list,
        _depth_list,
        _saved_pose_list,
        rgb_intrinsics,
        rgb_coeffs,
        _depth_intrinsics,
        _depth_coeffs,
        _depth_scale,
        joints_list,
    ) = read_data(data_root)

    if rgb_list is None or not rgb_list:
        raise ValueError(f"No RGB calibration images found under {data_root / 'rgb'}.")
    if joints_list is None:
        raise ValueError(
            f"No measured ViperX joints found under {data_root / 'joints'}. "
            "The Panda pose/TCP fallback is intentionally unsupported."
        )
    if len(rgb_list) != len(joints_list):
        raise ValueError(
            f"RGB/joint sample count mismatch: {len(rgb_list)} != {len(joints_list)}."
        )
    if rgb_intrinsics is None or rgb_coeffs is None:
        raise ValueError(f"Missing RGB intrinsics under {data_root}.")

    frame_ids = load_rgb_frame_ids(data_root)
    manifest_path = data_root / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Capture manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    samples_per_center = int(manifest["samples_per_center"])
    q_centers = manifest["q_centers_rad"]
    if samples_per_center <= 0 or not isinstance(q_centers, list) or not q_centers:
        raise ValueError(
            "capture_manifest must contain positive samples_per_center and "
            "non-empty q_centers_rad."
        )
    sample_center_ids = [
        frame_id // samples_per_center for frame_id in frame_ids
    ]
    if any(center_id >= len(q_centers) for center_id in sample_center_ids):
        raise ValueError("RGB frame IDs exceed the centers recorded in the manifest.")

    pose_list = [joint_to_hand(joints, model) for joints in joints_list]
    print(f"mapping_path={mapping_path}")
    print(f"urdf_path={urdf_path}")
    print(f"joint_order={mapping.joint_order}")
    print(f"base_link={mapping.base_link}")
    print(f"end_link={mapping.end_link}")
    print(f"{len(pose_list)} poses found")
    print(f"Camera matrix: {rgb_intrinsics}")

    # These parameters match the physical A4 ChArUco board generated by
    # utils/generate_charuco_board.py: 180 mm board, 36 mm squares, 27 mm
    # markers.  Pose translations are therefore estimated in metres.
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        CHARUCO_BOARD_SHAPE,
        CHARUCO_SQUARE_LENGTH_M,
        CHARUCO_MARKER_LENGTH_M,
        charuco_dict,
    )

    calibrator = HandinEyeCalibrator(
        rgb_intrinsics,
        rgb_coeffs,
        charuco_dict,
        board,
    )
    R_cam2hand_avg, t_cam2hand_avg = calibrator.perform(
        rgb_list,
        pose_list,
        sample_ids=frame_ids,
        sample_center_ids=sample_center_ids,
        min_valid_samples_per_center=MIN_VALID_SAMPLES_PER_CENTER,
    )

    print("Average Camera to hand rotation matrix:")
    print(R_cam2hand_avg)
    print("Average Camera to hand translation vector:")
    print(t_cam2hand_avg)

    cam_to_hand_pose = np.eye(4, dtype=np.float64)
    cam_to_hand_pose[:3, :3] = R_cam2hand_avg
    cam_to_hand_pose[:3, 3] = t_cam2hand_avg.squeeze()
    cam_to_hand_pose = model.validate_transform(cam_to_hand_pose)
    print(f"Camera to hand pose:\n{cam_to_hand_pose}")

    output_path = data_root / "cam_to_hand_pose.npy"
    np.save(output_path, cam_to_hand_pose)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
