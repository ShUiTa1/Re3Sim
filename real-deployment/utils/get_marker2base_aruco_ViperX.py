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
ANALYSIS_FILENAME = "marker_2_base_analysis.json"
MIN_VALID_SAMPLES_PER_CENTER = 1


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
    # Stage 8 keeps cv2.imread() output in BGR; Stage 7 reaches the same
    # grayscale image via read_data() RGB output and COLOR_RGB2GRAY.
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


def residuals_to_reference(
    transforms: Sequence[np.ndarray],
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return translation-mm and rotation-degree residuals to one transform."""

    transform_array = np.asarray(transforms, dtype=np.float64)
    translation_mm = (
        np.linalg.norm(
            transform_array[:, :3, 3] - reference[:3, 3],
            axis=1,
        )
        * 1000.0
    )
    reference_rotation = R.from_matrix(reference[:3, :3])
    rotation_deg = np.asarray(
        [
            np.degrees(
                (reference_rotation.inv() * R.from_matrix(transform[:3, :3]))
                .magnitude()
            )
            for transform in transform_array
        ],
        dtype=np.float64,
    )
    return translation_mm, rotation_deg


def summarize_residuals(values: Sequence[float]) -> dict[str, float]:
    """Return the scalar statistics used by the Stage-8 quality report."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty residual array.")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def select_solution_frame_records(
    frame_records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Drop centers with fewer than the required detected ChArUco frames."""

    samples_per_center = int(manifest["samples_per_center"])
    q_centers = manifest["q_centers_rad"]
    if samples_per_center <= 0 or not isinstance(q_centers, list) or not q_centers:
        raise ValueError(
            "capture_manifest must contain positive samples_per_center and "
            "non-empty q_centers_rad."
        )

    center_counts = {center_id: 0 for center_id in range(len(q_centers))}
    for record in frame_records:
        center_id = int(record["frame_id"]) // samples_per_center
        if center_id >= len(q_centers):
            raise ValueError(
                f"Frame {record['frame_id']} maps to center {center_id}, but the "
                f"manifest contains only {len(q_centers)} centers."
            )
        center_counts[center_id] += 1

    dropped_center_ids = [
        center_id
        for center_id, count in center_counts.items()
        if count < MIN_VALID_SAMPLES_PER_CENTER
    ]
    dropped_center_id_set = set(dropped_center_ids)
    included_records = [
        record
        for record in frame_records
        if int(record["frame_id"]) // samples_per_center
        not in dropped_center_id_set
    ]
    return included_records, dropped_center_ids


def build_analysis_report(
    frame_records: Sequence[dict[str, Any]],
    included_frame_ids: Sequence[int],
    dropped_center_ids: Sequence[int],
    failed_frame_ids: Sequence[int],
    all_frame_ids: Sequence[int],
    manifest: dict[str, Any],
    marker_2_base: np.ndarray,
    cam_2_ee: np.ndarray,
) -> dict[str, Any]:
    """Build global, per-center, and per-frame marker-pose diagnostics."""

    samples_per_center = int(manifest["samples_per_center"])
    q_centers = manifest["q_centers_rad"]
    if samples_per_center <= 0:
        raise ValueError("capture_manifest samples_per_center must be positive.")
    if not isinstance(q_centers, list) or not q_centers:
        raise ValueError("capture_manifest q_centers_rad must be a non-empty list.")

    transforms = [record["marker_2_base"] for record in frame_records]
    all_translation_mm, all_rotation_deg = residuals_to_reference(
        transforms,
        marker_2_base,
    )
    included_frame_id_set = set(included_frame_ids)
    included_indices = [
        index
        for index, record in enumerate(frame_records)
        if int(record["frame_id"]) in included_frame_id_set
    ]
    global_translation_mm = all_translation_mm[included_indices]
    global_rotation_deg = all_rotation_deg[included_indices]

    report_frames: dict[int, dict[str, Any]] = {}
    for record, translation_mm, rotation_deg in zip(
        frame_records,
        all_translation_mm,
        all_rotation_deg,
    ):
        frame_id = int(record["frame_id"])
        center_id = frame_id // samples_per_center
        if center_id >= len(q_centers):
            raise ValueError(
                f"Frame {frame_id} maps to center {center_id}, but the manifest "
                f"contains only {len(q_centers)} centers."
            )
        report_frames[frame_id] = {
            "frame_id": frame_id,
            "center_id": center_id,
            "included_in_solution": frame_id in included_frame_id_set,
            "marker_2_base": np.asarray(
                record["marker_2_base"], dtype=np.float64
            ).tolist(),
            "residual_to_global": {
                "translation_mm": float(translation_mm),
                "rotation_deg": float(rotation_deg),
            },
        }

    center_reports = []
    all_frame_id_set = set(all_frame_ids)
    failed_frame_id_set = set(failed_frame_ids)
    for center_id, q_center_rad in enumerate(q_centers):
        first_frame = center_id * samples_per_center
        expected_ids = list(range(first_frame, first_frame + samples_per_center))
        present_ids = [frame_id for frame_id in expected_ids if frame_id in all_frame_id_set]
        valid_ids = [frame_id for frame_id in expected_ids if frame_id in report_frames]
        failed_ids = [
            frame_id for frame_id in expected_ids if frame_id in failed_frame_id_set
        ]

        center_report: dict[str, Any] = {
            "center_id": center_id,
            "q_center_rad": q_center_rad,
            "expected_frame_ids": expected_ids,
            "present_frame_ids": present_ids,
            "valid_frame_ids": valid_ids,
            "failed_frame_ids": failed_ids,
            "valid_count": len(valid_ids),
            "included_in_solution": center_id not in set(dropped_center_ids),
        }
        if valid_ids:
            center_transforms = [
                np.asarray(report_frames[frame_id]["marker_2_base"], dtype=np.float64)
                for frame_id in valid_ids
            ]
            center_mean = average_transforms(center_transforms)
            within_translation_mm, within_rotation_deg = residuals_to_reference(
                center_transforms,
                center_mean,
            )
            center_translation_mm, center_rotation_deg = residuals_to_reference(
                [center_mean],
                marker_2_base,
            )
            center_report.update(
                {
                    "mean_marker_2_base": center_mean.tolist(),
                    "within_center": {
                        "translation_mm": summarize_residuals(
                            within_translation_mm
                        ),
                        "rotation_deg": summarize_residuals(within_rotation_deg),
                    },
                    "center_mean_to_global": {
                        "translation_mm": float(center_translation_mm[0]),
                        "rotation_deg": float(center_rotation_deg[0]),
                    },
                }
            )
            for frame_id, translation_mm, rotation_deg in zip(
                valid_ids,
                within_translation_mm,
                within_rotation_deg,
            ):
                report_frames[frame_id]["residual_to_center"] = {
                    "translation_mm": float(translation_mm),
                    "rotation_deg": float(rotation_deg),
                }
        center_reports.append(center_report)

    return {
        "analysis_version": 1,
        "samples_per_center": samples_per_center,
        "min_valid_samples_per_center": MIN_VALID_SAMPLES_PER_CENTER,
        "total_frame_count": len(all_frame_ids),
        "detected_frame_count": len(frame_records),
        "included_frame_count": len(included_frame_ids),
        "failed_frame_ids": sorted(int(frame_id) for frame_id in failed_frame_ids),
        "dropped_center_ids": sorted(int(center_id) for center_id in dropped_center_ids),
        "dropped_frame_ids": sorted(
            int(record["frame_id"])
            for record in frame_records
            if int(record["frame_id"]) not in included_frame_id_set
        ),
        "cam_to_hand_pose": np.asarray(cam_2_ee, dtype=np.float64).tolist(),
        "global": {
            "mean_marker_2_base": marker_2_base.tolist(),
            "translation_mm": summarize_residuals(global_translation_mm),
            "rotation_deg": summarize_residuals(global_rotation_deg),
        },
        "centers": center_reports,
        "frames": [report_frames[frame_id] for frame_id in sorted(report_frames)],
    }


def print_analysis_report(report: dict[str, Any]) -> None:
    """Print a compact terminal summary while keeping full details in JSON."""

    print(
        "\nStage-8 marker-to-base analysis: "
        f"detected={report['detected_frame_count']}/{report['total_frame_count']}, "
        f"included={report['included_frame_count']}/{report['total_frame_count']}"
    )
    print(f"failed_frames={report['failed_frame_ids']}")
    print(
        "min_valid_samples_per_center="
        f"{report['min_valid_samples_per_center']}"
    )
    print(f"dropped_centers={report['dropped_center_ids']}")
    print(f"dropped_frames={report['dropped_frame_ids']}")
    for metric_name in ("translation_mm", "rotation_deg"):
        stats = report["global"][metric_name]
        print(
            f"global_{metric_name}: "
            f"mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
            f"rms={stats['rms']:.3f}, p95={stats['p95']:.3f}, "
            f"max={stats['max']:.3f}"
        )

    print("per_center:")
    for center in report["centers"]:
        if center["valid_count"] == 0:
            print(
                f"  center={center['center_id']} valid=0/"
                f"{len(center['expected_frame_ids'])} included=False"
            )
            continue
        translation = center["within_center"]["translation_mm"]
        rotation = center["within_center"]["rotation_deg"]
        offset = center["center_mean_to_global"]
        print(
            f"  center={center['center_id']} "
            f"valid={center['valid_count']}/{len(center['expected_frame_ids'])} "
            f"included={center['included_in_solution']} "
            f"within_translation_mm(median/p95/max)="
            f"{translation['median']:.3f}/{translation['p95']:.3f}/"
            f"{translation['max']:.3f} "
            f"within_rotation_deg(median/p95/max)="
            f"{rotation['median']:.3f}/{rotation['p95']:.3f}/"
            f"{rotation['max']:.3f} "
            f"center_mean_to_global="
            f"{offset['translation_mm']:.3f}mm/"
            f"{offset['rotation_deg']:.3f}deg"
        )

    largest_translation = sorted(
        [
            frame
            for frame in report["frames"]
            if frame["included_in_solution"]
        ],
        key=lambda frame: frame["residual_to_global"]["translation_mm"],
        reverse=True,
    )[:5]
    print("largest_included_global_translation_residual_frames:")
    for frame in largest_translation:
        residual = frame["residual_to_global"]
        print(
            f"  frame={frame['frame_id']} center={frame['center_id']} "
            f"translation_mm={residual['translation_mm']:.3f} "
            f"rotation_deg={residual['rotation_deg']:.3f}"
        )

    largest_rotation = sorted(
        [
            frame
            for frame in report["frames"]
            if frame["included_in_solution"]
        ],
        key=lambda frame: frame["residual_to_global"]["rotation_deg"],
        reverse=True,
    )[:5]
    print("largest_included_global_rotation_residual_frames:")
    for frame in largest_rotation:
        residual = frame["residual_to_global"]
        print(
            f"  frame={frame['frame_id']} center={frame['center_id']} "
            f"translation_mm={residual['translation_mm']:.3f} "
            f"rotation_deg={residual['rotation_deg']:.3f}"
        )


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

    frame_records = []
    failed_frame_ids = []
    all_frame_ids = []
    for joints_path in joints_paths:
        match = re.fullmatch(r"joints_(.+)\.npy", joints_path.name)
        if match is None:
            raise ValueError(f"Unexpected joints filename: {joints_path.name}")
        frame_id = match.group(1)
        try:
            numeric_frame_id = int(frame_id)
        except ValueError as exc:
            raise ValueError(
                f"Expected a numeric frame id, got {frame_id!r} in "
                f"{joints_path.name}."
            ) from exc
        all_frame_ids.append(numeric_frame_id)

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
            failed_frame_ids.append(numeric_frame_id)
            continue

        qpos = np.load(joints_path)
        ee_2_base = joint_to_hand(qpos, model)
        cam_2_base = ee_2_base @ cam_2_ee
        marker_2_cam = np.linalg.inv(cam_2_marker)
        marker_2_base = model.validate_transform(cam_2_base @ marker_2_cam)
        frame_records.append(
            {
                "frame_id": numeric_frame_id,
                "marker_2_base": marker_2_base,
            }
        )

    manifest_path = data_root / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Capture manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    included_frame_records, dropped_center_ids = (
        select_solution_frame_records(frame_records, manifest)
    )
    included_frame_ids = [
        int(record["frame_id"]) for record in included_frame_records
    ]
    marker_2_base = model.validate_transform(
        average_transforms(
            [record["marker_2_base"] for record in included_frame_records]
        )
    )
    print(f"Marker to base:\n{marker_2_base}")

    analysis_report = build_analysis_report(
        frame_records=frame_records,
        included_frame_ids=included_frame_ids,
        dropped_center_ids=dropped_center_ids,
        failed_frame_ids=failed_frame_ids,
        all_frame_ids=all_frame_ids,
        manifest=manifest,
        marker_2_base=marker_2_base,
        cam_2_ee=cam_2_ee,
    )
    print_analysis_report(analysis_report)

    item_to_show = [
        show_pose(record["marker_2_base"]) for record in included_frame_records
    ]
    frame_base = show_pose(np.eye(4))
    item_to_show.append(frame_base)
    o3d.visualization.draw_geometries(item_to_show)

    output_path = data_root / "marker_2_base.npy"
    np.save(output_path, marker_2_base)
    print(f"saved={output_path}")
    analysis_path = data_root / ANALYSIS_FILENAME
    with analysis_path.open("w", encoding="utf-8") as file:
        json.dump(analysis_report, file, indent=2)
        file.write("\n")
    print(f"analysis_saved={analysis_path}")


if __name__ == "__main__":
    main()
