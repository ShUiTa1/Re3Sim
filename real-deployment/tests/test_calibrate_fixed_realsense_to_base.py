import importlib.util
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2


REAL_DEPLOYMENT_DIR = Path(__file__).resolve().parents[1]
NEW_SCRIPT = (
    REAL_DEPLOYMENT_DIR / "utils" / "calibrate_fixed_realsense_to_base.py"
)
OLD_SCRIPT = REAL_DEPLOYMENT_DIR / "utils" / "get_side_camera_pose.py"
SCRIPT_PATH = NEW_SCRIPT if NEW_SCRIPT.is_file() else OLD_SCRIPT

spec = importlib.util.spec_from_file_location(
    "calibrate_fixed_realsense_to_base", SCRIPT_PATH
)
calibration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = calibration
spec.loader.exec_module(calibration)


class FixedCameraCalibrationContractTest(unittest.TestCase):
    def require(self, name):
        self.assertTrue(
            hasattr(calibration, name),
            f"calibration script is missing public function {name}",
        )
        return getattr(calibration, name)

    def test_only_connected_camera_is_selected_automatically(self):
        select = self.require("select_camera_serial")
        self.assertEqual(select(["123"], None), "123")

    def test_multiple_connected_cameras_require_an_explicit_serial(self):
        select = self.require("select_camera_serial")
        with self.assertRaisesRegex(ValueError, "multiple"):
            select(["123", "456"], None)

    def test_requested_camera_must_be_connected(self):
        select = self.require("select_camera_serial")
        with self.assertRaisesRegex(ValueError, "not connected"):
            select(["123"], "456")

    def test_camera_pose_uses_marker_to_base_and_inverts_pnp_pose(self):
        compose = self.require("base_from_camera")
        base_from_marker = np.array(
            [
                [1.0, 0.0, 0.0, 0.50],
                [0.0, 1.0, 0.0, -0.20],
                [0.0, 0.0, 1.0, 0.30],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        camera_from_marker = np.array(
            [
                [0.0, -1.0, 0.0, 0.10],
                [1.0, 0.0, 0.0, 0.20],
                [0.0, 0.0, 1.0, 0.80],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected = np.array(
            [
                [0.0, 1.0, 0.0, 0.30],
                [-1.0, 0.0, 0.0, -0.10],
                [0.0, 0.0, 1.0, -0.50],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(
            compose(base_from_marker, camera_from_marker), expected, atol=1e-12
        )

    def test_invalid_transform_is_rejected_before_composition(self):
        validate = self.require("validate_transform")
        with self.assertRaisesRegex(ValueError, "finite 4 x 4"):
            validate(np.eye(3), "bad")
        invalid = np.eye(4)
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite 4 x 4"):
            validate(invalid, "bad")

    def test_transform_average_uses_translation_and_so3_means(self):
        average = self.require("average_transforms")
        identity = np.eye(4)
        quarter_turn = np.array(
            [
                [0.0, -1.0, 0.0, 2.0],
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 0.0, 1.0, 6.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        root_half = np.sqrt(0.5)
        expected = np.array(
            [
                [root_half, -root_half, 0.0, 1.0],
                [root_half, root_half, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(
            average([identity, quarter_turn]), expected, atol=1e-12
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            average([])

    def test_camera_params_match_runtime_yaml_order(self):
        to_params = self.require("camera_params")
        intrinsics = SimpleNamespace(
            fx=600.5,
            fy=601.5,
            ppx=320.25,
            ppy=240.75,
            width=640,
            height=480,
        )
        self.assertEqual(
            to_params(intrinsics),
            [600.5, 601.5, 320.25, 240.75, 640, 480],
        )

    def test_cli_defaults_match_the_top_camera_capture_contract(self):
        build_parser = self.require("build_arg_parser")
        args = build_parser().parse_args([])
        expected_data = REAL_DEPLOYMENT_DIR / "calibration" / "data" / "viperx_hand_eye"
        self.assertIsNone(args.serial)
        self.assertEqual(args.marker_to_base, expected_data / "marker_2_base.npy")
        self.assertEqual(args.output_dir, expected_data)
        self.assertEqual(args.name, "top_camera")
        self.assertEqual((args.width, args.height, args.fps), (640, 480, 30))
        self.assertEqual(args.samples, 1)
        self.assertEqual(args.max_frames, 300)

    def test_cli_rejects_nonpositive_capture_limits(self):
        parser = self.require("build_arg_parser")()
        for option in ("--samples", "--max-frames"):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([option, "0"])

    def test_output_paths_use_camera_name_without_extra_files(self):
        paths = self.require("output_paths")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertEqual(
                paths(output, "top_camera"),
                (
                    output / "top_camera_to_base.npy",
                    output / "top_camera_intrinsics.npz",
                    output / "top_camera_preview.png",
                ),
            )

    def test_charuco_pose_is_detected_from_a_real_board_image(self):
        estimate = self.require("estimate_charuco_pose")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        board = cv2.aruco.CharucoBoard((5, 5), 0.036, 0.027, dictionary)
        gray = board.generateImage((600, 600), marginSize=40)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        intrinsics = np.array(
            [[800.0, 0.0, 300.0], [0.0, 800.0, 300.0], [0.0, 0.0, 1.0]]
        )

        result = estimate(image, intrinsics, np.zeros(5), board)

        self.assertIsNotNone(result)
        camera_from_marker, preview = result
        self.assertEqual(camera_from_marker.shape, (4, 4))
        self.assertGreater(camera_from_marker[2, 3], 0.0)
        self.assertEqual(preview.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
