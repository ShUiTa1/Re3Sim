"""Evaluate fitted ViperX hand-eye transforms without solving them again.

For each Stage-8 included frame, this script evaluates the absolute equation

``T_base_hand @ T_hand_camera = T_base_marker @ T_marker_camera``.

The EPFL-compatible values are in-sample coordinate-chain fit residuals, not
errors to a physical ground-truth transform.  The script also recomputes the
existing Stage-8 mean translation distance and SO(3) rotation angle to the
saved ``marker_2_base.npy`` result.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REAL_DEPLOYMENT_DIR = SCRIPT_DIR.parent
for import_dir in (SCRIPT_DIR, REAL_DEPLOYMENT_DIR):
    import_dir_text = str(import_dir)
    if import_dir_text not in sys.path:
        sys.path.insert(0, import_dir_text)

from calibration.hand_in_eye import HandinEyeCalibrator
from hand_in_eye_calib_ViperX import (
    CHARUCO_BOARD_SHAPE,
    CHARUCO_MARKER_LENGTH_M,
    CHARUCO_SQUARE_LENGTH_M,
    joint_to_hand,
    load_mapping_metadata,
    resolve_mapping_path,
    resolve_urdf_path,
    validate_model_contract,
)
from viperx_adapter import ViperXUrdfMapping
from viperx_model import load_viperx_model


OUTPUT_FILENAME = "hand_eye_fit_analysis.json"
STAGE8_ANALYSIS_FILENAME = "marker_2_base_analysis.json"
MEAN_MATCH_ATOL = 1e-6
MEAN_MATCH_RTOL = 1e-9


def rotation_matrix_to_zyx_deg(rotation: np.ndarray) -> np.ndarray:
    """Return ZYX Euler components as ``[yaw_z, pitch_y, roll_x]`` degrees."""

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation shape (3, 3), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Rotation matrix contains NaN or infinity.")

    pitch_y = math.atan2(
        -matrix[2, 0],
        math.sqrt(matrix[0, 0] ** 2 + matrix[1, 0] ** 2),
    )
    if math.isclose(pitch_y, math.pi / 2.0, abs_tol=1e-3):
        yaw_z = math.atan2(matrix[1, 2], matrix[1, 1])
        roll_x = 0.0
    elif math.isclose(pitch_y, -math.pi / 2.0, abs_tol=1e-3):
        yaw_z = math.atan2(-matrix[1, 2], matrix[1, 1])
        roll_x = 0.0
    else:
        yaw_z = math.atan2(matrix[1, 0], matrix[0, 0])
        roll_x = math.atan2(matrix[2, 1], matrix[2, 2])

    return np.degrees(np.array([yaw_z, pitch_y, roll_x], dtype=np.float64))


def compute_frame_metrics(
    A: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one fitted absolute equation ``A @ X = Y @ B``."""

    transforms = {
        name: np.asarray(transform, dtype=np.float64)
        for name, transform in (("A", A), ("X", X), ("Y", Y), ("B", B))
    }
    for name, transform in transforms.items():
        if transform.shape != (4, 4):
            raise ValueError(
                f"Expected {name} transform shape (4, 4), got {transform.shape}."
            )
        if not np.all(np.isfinite(transform)):
            raise ValueError(f"{name} transform contains NaN or infinity.")

    left = transforms["A"] @ transforms["X"]
    right = transforms["Y"] @ transforms["B"]
    translation_abs_xyz_mm = (
        np.abs(left[:3, 3] - right[:3, 3]) * 1000.0
    )
    rotation_error = left[:3, :3] @ right[:3, :3].T
    rotation_abs_zyx_deg = np.abs(
        rotation_matrix_to_zyx_deg(rotation_error)
    )

    T_camera_marker = np.linalg.inv(transforms["B"])
    marker_to_base_i = (
        transforms["A"] @ transforms["X"] @ T_camera_marker
    )
    translation_residual_mm = (
        np.linalg.norm(marker_to_base_i[:3, 3] - transforms["Y"][:3, 3])
        * 1000.0
    )
    rotation_residual_deg = np.degrees(
        (
            Rotation.from_matrix(transforms["Y"][:3, :3]).inv()
            * Rotation.from_matrix(marker_to_base_i[:3, :3])
        ).magnitude()
    )

    return {
        "epfl": {
            "frobenius_error": float(np.linalg.norm(left - right)),
            "translation_abs_xyz_mm": translation_abs_xyz_mm.tolist(),
            "rotation_abs_zyx_deg": rotation_abs_zyx_deg.tolist(),
        },
        "stage8": {
            "marker_to_base": marker_to_base_i.tolist(),
            "translation_residual_mm": float(translation_residual_mm),
            "rotation_residual_deg": float(rotation_residual_deg),
        },
    }


def aggregate_metrics(
    frame_metrics: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-frame values with the EPFL and Stage-8 definitions."""

    frames = list(frame_metrics)
    if not frames:
        raise ValueError("Cannot aggregate an empty frame list.")

    frobenius_errors = np.asarray(
        [frame["epfl"]["frobenius_error"] for frame in frames],
        dtype=np.float64,
    )
    translation_xyz_mm = np.asarray(
        [frame["epfl"]["translation_abs_xyz_mm"] for frame in frames],
        dtype=np.float64,
    )
    rotation_zyx_deg = np.asarray(
        [frame["epfl"]["rotation_abs_zyx_deg"] for frame in frames],
        dtype=np.float64,
    )
    stage8_translation_mm = np.asarray(
        [frame["stage8"]["translation_residual_mm"] for frame in frames],
        dtype=np.float64,
    )
    stage8_rotation_deg = np.asarray(
        [frame["stage8"]["rotation_residual_deg"] for frame in frames],
        dtype=np.float64,
    )

    return {
        "epfl": {
            "total_error": float(
                np.sqrt(np.sum(np.square(frobenius_errors))) / len(frames)
            ),
            "translation_mean_abs_xyz_mm": np.mean(
                translation_xyz_mm,
                axis=0,
            ).tolist(),
            "rotation_mean_abs_zyx_deg": np.mean(
                rotation_zyx_deg,
                axis=0,
            ).tolist(),
        },
        "stage8": {
            "translation_residual_mean_mm": float(
                np.mean(stage8_translation_mm)
            ),
            "rotation_residual_mean_deg": float(
                np.mean(stage8_rotation_deg)
            ),
        },
    }


def extract_included_frame_ids(stage8_analysis: dict[str, Any]) -> list[int]:
    """Return the exact frame IDs marked as included by Stage 8."""

    records = stage8_analysis.get("frames")
    if not isinstance(records, list):
        raise ValueError("Stage-8 analysis must contain a frames list.")

    included_frame_ids = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each Stage-8 frame record must be an object.")
        if "frame_id" not in record or "included_in_solution" not in record:
            raise ValueError(
                "Each Stage-8 frame record requires frame_id and "
                "included_in_solution."
            )
        frame_id = record["frame_id"]
        if isinstance(frame_id, bool) or not isinstance(frame_id, int):
            raise ValueError(f"Invalid Stage-8 frame_id: {frame_id!r}.")
        if bool(record["included_in_solution"]):
            included_frame_ids.append(frame_id)

    if not included_frame_ids:
        raise ValueError("Stage 8 did not include any frames.")
    if len(set(included_frame_ids)) != len(included_frame_ids):
        raise ValueError("Stage-8 included frame IDs contain duplicates.")
    return included_frame_ids


def validate_stage8_mean_consistency(
    aggregate: dict[str, Any],
    stage8_analysis: dict[str, Any],
) -> dict[str, float]:
    """Require recomputed Stage-8 means to match its saved analysis."""

    try:
        reference = {
            "translation_residual_mean_mm": float(
                stage8_analysis["global"]["translation_mm"]["mean"]
            ),
            "rotation_residual_mean_deg": float(
                stage8_analysis["global"]["rotation_deg"]["mean"]
            ),
        }
        recomputed = {
            "translation_residual_mean_mm": float(
                aggregate["stage8"]["translation_residual_mean_mm"]
            ),
            "rotation_residual_mean_deg": float(
                aggregate["stage8"]["rotation_residual_mean_deg"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage-8 analysis does not contain valid global mean residuals."
        ) from exc

    for metric_name, reference_value in reference.items():
        recomputed_value = recomputed[metric_name]
        if not np.isfinite(reference_value) or not np.isfinite(recomputed_value):
            raise ValueError(f"{metric_name} is not finite.")
        if not np.isclose(
            recomputed_value,
            reference_value,
            atol=MEAN_MATCH_ATOL,
            rtol=MEAN_MATCH_RTOL,
        ):
            raise ValueError(
                f"Recomputed {metric_name}={recomputed_value:.12g} does not "
                f"match Stage-8 value {reference_value:.12g}."
            )
    return reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate existing ViperX Stage-7/8 transforms with "
            "EPFL-compatible in-sample fit metrics."
        )
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="ViperX shooting dataset root containing Stage-7/8 outputs.",
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
        help="Local accepted vx300s_full.urdf; overrides mapping metadata.",
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

    stage8_analysis_path = data_root / STAGE8_ANALYSIS_FILENAME
    if not stage8_analysis_path.is_file():
        raise FileNotFoundError(
            f"Stage-8 analysis not found: {stage8_analysis_path}"
        )
    with stage8_analysis_path.open("r", encoding="utf-8") as file:
        stage8_analysis = json.load(file)
    if not isinstance(stage8_analysis, dict):
        raise ValueError("Stage-8 analysis root must be a JSON object.")
    included_frame_ids = extract_included_frame_ids(stage8_analysis)

    cam_to_hand_path = data_root / "cam_to_hand_pose.npy"
    marker_to_base_path = data_root / "marker_2_base.npy"
    if not cam_to_hand_path.is_file():
        raise FileNotFoundError(
            f"Stage-7 camera-to-hand pose not found: {cam_to_hand_path}"
        )
    if not marker_to_base_path.is_file():
        raise FileNotFoundError(
            f"Stage-8 marker-to-base pose not found: {marker_to_base_path}"
        )
    cam_to_hand = model.validate_transform(np.load(cam_to_hand_path))
    marker_to_base = model.validate_transform(np.load(marker_to_base_path))

    try:
        recorded_cam_to_hand = model.validate_transform(
            np.asarray(stage8_analysis["cam_to_hand_pose"], dtype=np.float64)
        )
        recorded_marker_to_base = model.validate_transform(
            np.asarray(
                stage8_analysis["global"]["mean_marker_2_base"],
                dtype=np.float64,
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Stage-8 analysis does not contain valid saved X/Y transforms."
        ) from exc
    if not np.allclose(
        cam_to_hand,
        recorded_cam_to_hand,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(
            "cam_to_hand_pose.npy differs from the transform used by Stage 8."
        )
    if not np.allclose(
        marker_to_base,
        recorded_marker_to_base,
        atol=1e-10,
        rtol=1e-10,
    ):
        raise ValueError(
            "marker_2_base.npy differs from the Stage-8 global mean transform."
        )

    intrinsic_path = data_root / "rgb_intrinsics.npz"
    if not intrinsic_path.is_file():
        raise FileNotFoundError(f"RGB intrinsics not found: {intrinsic_path}")
    with np.load(intrinsic_path) as intrinsic_zip:
        intrinsics_matrix = np.array(
            [
                [
                    float(intrinsic_zip["fx"]),
                    0.0,
                    float(intrinsic_zip["ppx"]),
                ],
                [
                    0.0,
                    float(intrinsic_zip["fy"]),
                    float(intrinsic_zip["ppy"]),
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion_coeffs = np.asarray(
            intrinsic_zip["coeffs"],
            dtype=np.float64,
        ).reshape(-1)

    manifest_path = data_root / "capture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Capture manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("Capture manifest root must be a JSON object.")
    try:
        samples_per_center = int(manifest["samples_per_center"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Capture manifest requires a valid samples_per_center."
        ) from exc
    if samples_per_center <= 0:
        raise ValueError("samples_per_center must be positive.")

    charuco_dict = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_6X6_250
    )
    board = cv2.aruco.CharucoBoard(
        CHARUCO_BOARD_SHAPE,
        CHARUCO_SQUARE_LENGTH_M,
        CHARUCO_MARKER_LENGTH_M,
        charuco_dict,
    )
    calibrator = HandinEyeCalibrator(
        intrinsics_matrix,
        distortion_coeffs,
        charuco_dict,
        board,
    )

    frame_reports = []
    frame_metrics = []
    for frame_id in included_frame_ids:
        image_path = data_root / "rgb" / f"{frame_id}.png"
        joints_path = data_root / "joints" / f"joints_{frame_id}.npy"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Included Stage-8 RGB frame not found: {image_path}"
            )
        if not joints_path.is_file():
            raise FileNotFoundError(
                f"Included Stage-8 joint frame not found: {joints_path}"
            )

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise ValueError(f"RGB frame could not be decoded: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rvec, tvec = calibrator.estimate_pose(image_rgb)
        if rvec is None or tvec is None:
            raise ValueError(
                f"Stage 8 included frame {frame_id}, but ChArUco pose "
                "estimation now fails."
            )

        T_camera_marker = np.eye(4, dtype=np.float64)
        T_camera_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_camera_marker[:3, 3] = np.asarray(
            tvec,
            dtype=np.float64,
        ).reshape(3)
        T_marker_camera = np.linalg.inv(T_camera_marker)
        T_base_hand = joint_to_hand(np.load(joints_path), model)

        metrics = compute_frame_metrics(
            T_base_hand,
            cam_to_hand,
            marker_to_base,
            T_marker_camera,
        )
        frame_metrics.append(metrics)
        translation_xyz = metrics["epfl"]["translation_abs_xyz_mm"]
        rotation_zyx = metrics["epfl"]["rotation_abs_zyx_deg"]
        frame_reports.append(
            {
                "frame_id": frame_id,
                "center_id": frame_id // samples_per_center,
                "epfl": {
                    "frobenius_error": metrics["epfl"][
                        "frobenius_error"
                    ],
                    "translation_abs_xyz_mm": {
                        "x": translation_xyz[0],
                        "y": translation_xyz[1],
                        "z": translation_xyz[2],
                    },
                    "rotation_abs_zyx_deg": {
                        "yaw_z": rotation_zyx[0],
                        "pitch_y": rotation_zyx[1],
                        "roll_x": rotation_zyx[2],
                    },
                },
                "stage8": metrics["stage8"],
            }
        )

    aggregate = aggregate_metrics(frame_metrics)
    stage8_reference = validate_stage8_mean_consistency(
        aggregate,
        stage8_analysis,
    )
    translation_mean = aggregate["epfl"][
        "translation_mean_abs_xyz_mm"
    ]
    rotation_mean = aggregate["epfl"]["rotation_mean_abs_zyx_deg"]

    report = {
        "analysis_version": 1,
        "metric_scope": "in_sample_coordinate_chain_fit_not_ground_truth",
        "equation": "A_i @ X = Y @ B_i",
        "data_root": str(data_root),
        "mapping_path": str(mapping_path),
        "urdf_path": str(urdf_path),
        "joint_order": list(mapping.joint_order),
        "base_link": mapping.base_link,
        "end_link": mapping.end_link,
        "cam_to_hand_pose": cam_to_hand.tolist(),
        "marker_to_base_pose": marker_to_base.tolist(),
        "included_frame_ids": included_frame_ids,
        "failed_frame_ids": [
            int(frame_id)
            for frame_id in stage8_analysis.get("failed_frame_ids", [])
        ],
        "dropped_frame_ids": [
            int(frame_id)
            for frame_id in stage8_analysis.get("dropped_frame_ids", [])
        ],
        "metrics": {
            "epfl": {
                "total_error": aggregate["epfl"]["total_error"],
                "translation_mean_abs_xyz_mm": {
                    "x": translation_mean[0],
                    "y": translation_mean[1],
                    "z": translation_mean[2],
                },
                "rotation_mean_abs_zyx_deg": {
                    "yaw_z": rotation_mean[0],
                    "pitch_y": rotation_mean[1],
                    "roll_x": rotation_mean[2],
                },
            },
            "stage8_geometric": {
                **aggregate["stage8"],
                "reference": stage8_reference,
                "consistency_gate": "PASS",
            },
        },
        "frames": frame_reports,
    }

    print(
        f"included_frames={len(included_frame_ids)} "
        f"failed_frames={report['failed_frame_ids']} "
        f"dropped_frames={report['dropped_frame_ids']}"
    )
    print("EPFL-compatible in-sample fit:")
    print(f"  total_error={aggregate['epfl']['total_error']:.12g}")
    print(
        "  translation_mean_abs_xyz_mm: "
        f"x={translation_mean[0]:.6f} "
        f"y={translation_mean[1]:.6f} "
        f"z={translation_mean[2]:.6f}"
    )
    print(
        "  rotation_mean_abs_zyx_deg: "
        f"yaw_z={rotation_mean[0]:.6f} "
        f"pitch_y={rotation_mean[1]:.6f} "
        f"roll_x={rotation_mean[2]:.6f}"
    )
    print("Stage-8 geometric residual mean:")
    print(
        "  translation_mm="
        f"{aggregate['stage8']['translation_residual_mean_mm']:.6f}"
    )
    print(
        "  rotation_deg="
        f"{aggregate['stage8']['rotation_residual_mean_deg']:.6f}"
    )
    print("stage8_mean_consistency=PASS")

    output_path = data_root / OUTPUT_FILENAME
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"analysis_saved={output_path}")


if __name__ == "__main__":
    main()
