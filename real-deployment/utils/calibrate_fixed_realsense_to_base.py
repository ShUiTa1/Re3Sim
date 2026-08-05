"""Calibrate a fixed RealSense camera in the ViperX base frame."""

import argparse
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


SCRIPT_DIR = Path(__file__).resolve().parent
REAL_DEPLOYMENT_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = (
    REAL_DEPLOYMENT_DIR / "calibration" / "data" / "viperx_hand_eye"
)
BOARD_SHAPE = (5, 5)
SQUARE_LENGTH_M = 0.036
MARKER_LENGTH_M = 0.027
WARMUP_FRAMES = 30


def select_camera_serial(
    available: Sequence[str], requested: str | None
) -> str:
    """Select an explicit RealSense serial or the only connected device."""

    serials = [str(serial) for serial in available]
    if requested is not None:
        requested = str(requested)
        if requested not in serials:
            raise ValueError(
                f"RealSense serial {requested!r} is not connected; "
                f"available serials: {serials}"
            )
        return requested
    if not serials:
        raise ValueError("No RealSense camera is connected")
    if len(serials) > 1:
        raise ValueError(
            "Found multiple RealSense cameras; select one with --serial: "
            f"{serials}"
        )
    return serials[0]


def validate_transform(matrix: Any, label: str) -> np.ndarray:
    """Return one finite homogeneous 4 x 4 transform."""

    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4 x 4 transform")
    return transform


def base_from_camera(
    base_from_marker: Any, camera_from_marker: Any
) -> np.ndarray:
    """Compose ``T_base_camera`` from marker calibration and OpenCV PnP."""

    base_from_marker = validate_transform(
        base_from_marker, "marker_2_base"
    )
    camera_from_marker = validate_transform(
        camera_from_marker, "camera_from_marker"
    )
    return base_from_marker @ np.linalg.inv(camera_from_marker)


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average translations arithmetically and rotations on SO(3)."""

    if len(transforms) == 0:
        raise ValueError("Transform averaging requires at least one sample")
    values = np.asarray(
        [validate_transform(value, "camera pose") for value in transforms],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_matrix(values[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = np.mean(values[:, :3, 3], axis=0)
    return result


def camera_params(intrinsics: Any) -> list[float | int]:
    """Return the fixed camera parameter order consumed by the ViperX YAML."""

    return [
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.ppx),
        float(intrinsics.ppy),
        int(intrinsics.width),
        int(intrinsics.height),
    ]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the hardware command without opening a camera."""

    parser = argparse.ArgumentParser(
        description=(
            "Calibrate one fixed RealSense camera against marker_2_base.npy."
        )
    )
    parser.add_argument(
        "--serial",
        help="RealSense serial; required only when multiple cameras are connected",
    )
    parser.add_argument(
        "--marker-to-base",
        type=Path,
        default=DEFAULT_DATA_DIR / "marker_2_base.npy",
        help="accepted T_base_marker .npy file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory for extrinsic, intrinsics, and preview",
    )
    parser.add_argument(
        "--name", default="top_camera", help="output camera name prefix"
    )
    parser.add_argument("--width", type=_positive_int, default=640)
    parser.add_argument("--height", type=_positive_int, default=480)
    parser.add_argument("--fps", type=_positive_int, default=30)
    parser.add_argument(
        "--samples",
        type=_positive_int,
        default=1,
        help="number of valid fixed-pose detections to average",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_int,
        default=300,
        help="maximum captured frames used to find valid detections",
    )
    return parser


def output_paths(
    output_dir: str | Path, name: str
) -> tuple[Path, Path, Path]:
    """Return the three files produced by one fixed-camera calibration."""

    output_dir = Path(output_dir)
    return (
        output_dir / f"{name}_to_base.npy",
        output_dir / f"{name}_intrinsics.npz",
        output_dir / f"{name}_preview.png",
    )


def create_board() -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    return cv2.aruco.CharucoBoard(
        BOARD_SHAPE,
        SQUARE_LENGTH_M,
        MARKER_LENGTH_M,
        dictionary,
    )


def estimate_charuco_pose(
    image_bgr: np.ndarray,
    intrinsic_matrix: np.ndarray,
    distortion: np.ndarray,
    board: Any,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``T_camera_marker`` and an axis preview for one BGR frame."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = detector.detectBoard(gray)

    if corners is None or ids is None or len(corners) < 4:
        return None
    object_points, image_points = board.matchImagePoints(corners, ids)
    valid, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        np.asarray(intrinsic_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not valid or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None

    camera_from_marker = np.eye(4, dtype=np.float64)
    camera_from_marker[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_from_marker[:3, 3] = np.asarray(tvec).reshape(3)
    preview = image_bgr.copy()
    cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
    cv2.drawFrameAxes(
        preview,
        np.asarray(intrinsic_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
        rvec,
        tvec,
        0.09,
        3,
    )
    return camera_from_marker, preview


def _intrinsic_matrix(intrinsics: Any) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _connected_serials(rs: Any) -> list[str]:
    context = rs.context()
    return [
        device.get_info(rs.camera_info.serial_number)
        for device in context.query_devices()
    ]


def run_calibration(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Capture one fixed RealSense pose and write the calibration files."""

    if args.samples > args.max_frames:
        raise ValueError("--samples must not exceed --max-frames")
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is required; run this in the existing lab "
            "RealSense calibration environment"
        ) from exc

    serial = select_camera_serial(_connected_serials(rs), args.serial)
    marker_path = Path(args.marker_to_base).expanduser().resolve()
    if not marker_path.is_file():
        raise FileNotFoundError(f"marker-to-base transform not found: {marker_path}")
    marker_to_base = validate_transform(
        np.load(marker_path), "marker_2_base"
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    started = False
    try:
        profile = pipeline.start(config)
        started = True
        for _ in range(WARMUP_FRAMES):
            pipeline.wait_for_frames()

        color_profile = profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        intrinsic_matrix = _intrinsic_matrix(intrinsics)
        distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
        board = create_board()

        poses: list[np.ndarray] = []
        preview: np.ndarray | None = None
        for frame_index in range(args.max_frames):
            color_frame = pipeline.wait_for_frames().get_color_frame()
            if not color_frame:
                continue
            image_bgr = np.asanyarray(color_frame.get_data())
            estimate = estimate_charuco_pose(
                image_bgr,
                intrinsic_matrix,
                distortion,
                board,
            )
            if estimate is None:
                continue
            camera_from_marker, preview = estimate
            poses.append(base_from_camera(marker_to_base, camera_from_marker))
            print(
                f"valid_charuco_sample={len(poses)}/{args.samples} "
                f"frame={frame_index + 1}"
            )
            if len(poses) == args.samples:
                break

        if len(poses) != args.samples or preview is None:
            raise RuntimeError(
                f"Detected {len(poses)}/{args.samples} valid ChArUco samples "
                f"within {args.max_frames} frames"
            )

        camera_to_base = average_transforms(poses)
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        extrinsic_path, intrinsics_path, preview_path = output_paths(
            output_dir, args.name
        )
        np.save(extrinsic_path, camera_to_base)
        np.savez(
            intrinsics_path,
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
            coeffs=distortion,
            width=int(intrinsics.width),
            height=int(intrinsics.height),
        )
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"Failed to write preview image: {preview_path}")

        print(f"camera_serial={serial}")
        print("T_base_camera=")
        print(camera_to_base)
        print(f"camera_params: {camera_params(intrinsics)}")
        print(f"extrinsic={extrinsic_path}")
        print(f"intrinsics={intrinsics_path}")
        print(f"preview={preview_path}")
        return extrinsic_path, intrinsics_path, preview_path
    finally:
        if started:
            pipeline.stop()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.samples > args.max_frames:
        parser.error("--samples must not exceed --max-frames")
    run_calibration(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
