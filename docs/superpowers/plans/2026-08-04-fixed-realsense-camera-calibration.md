# Fixed RealSense Camera Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale fixed-side-camera utility with one live RealSense command that saves the top camera's intrinsics, `T_base_camera`, and a ChArUco axis preview.

**Architecture:** Keep hardware access and calibration in one standalone script under `real-deployment/utils`. Isolate serial selection, transform composition, transform averaging, validation, and YAML parameter formatting as pure functions so they can be tested without a camera; keep `pyrealsense2` inside the runtime boundary.

**Tech Stack:** Python 3, NumPy, OpenCV contrib (`cv2.aruco`), SciPy rotations, Intel RealSense SDK (`pyrealsense2`), `unittest`.

## Global Constraints

- Reuse the accepted 5 x 5 `DICT_6X6_250` ChArUco board with 36 mm squares and 27 mm markers.
- Treat `marker_2_base.npy` as `T_base_marker`.
- Save `top_camera_to_base.npy` as `T_base_camera = T_base_marker @ inverse(T_camera_marker)`.
- Default to 640 x 480 RGB at 30 FPS, one valid pose sample, and at most 300 capture frames.
- Do not connect to the robot, run Isaac Sim, or add a second production script.

---

### Task 1: Pure calibration contract

**Files:**
- Rename: `real-deployment/utils/get_side_camera_pose.py` to `real-deployment/utils/calibrate_fixed_realsense_to_base.py`
- Create: `real-deployment/tests/test_calibrate_fixed_realsense_to_base.py`

**Interfaces:**
- Produces: `select_camera_serial(available: Sequence[str], requested: str | None) -> str`
- Produces: `validate_transform(matrix: Any, label: str) -> np.ndarray`
- Produces: `base_from_camera(base_from_marker: Any, camera_from_marker: Any) -> np.ndarray`
- Produces: `average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray`
- Produces: `camera_params(intrinsics: Any) -> list[float | int]`

- [ ] **Step 1: Write failing pure-function tests**

Add tests that assert:

```python
self.assertEqual(select_camera_serial(["123"], None), "123")
with self.assertRaisesRegex(ValueError, "multiple"):
    select_camera_serial(["123", "456"], None)
self.assertEqual(select_camera_serial(["123", "456"], "456"), "456")

expected = base_from_marker @ np.linalg.inv(camera_from_marker)
np.testing.assert_allclose(
    base_from_camera(base_from_marker, camera_from_marker), expected
)

self.assertEqual(
    camera_params(fake_intrinsics),
    [fake_intrinsics.fx, fake_intrinsics.fy,
     fake_intrinsics.ppx, fake_intrinsics.ppy, 640, 480],
)
```

Also reject missing serials, non-finite/non-4 x 4 transforms, and empty transform averaging.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m unittest real-deployment/tests/test_calibrate_fixed_realsense_to_base.py -v
```

Expected: FAIL because `calibrate_fixed_realsense_to_base.py` and its pure functions do not exist.

- [ ] **Step 3: Rename the stale script and replace it with the minimal pure functions**

Use `git mv`, remove the obsolete Franka paths and 40/30 mm board values, and implement only the five interfaces above. `average_transforms` must use a translation arithmetic mean and `Rotation.from_matrix(...).mean()` for SO(3).

- [ ] **Step 4: Run tests and verify GREEN**

Run the same `unittest` command. Expected: all pure contract tests pass without importing `pyrealsense2`.

### Task 2: Live RealSense capture and files

**Files:**
- Modify: `real-deployment/utils/calibrate_fixed_realsense_to_base.py`
- Modify: `real-deployment/tests/test_calibrate_fixed_realsense_to_base.py`

**Interfaces:**
- Consumes: the pure functions from Task 1.
- Produces: `estimate_charuco_pose(image_bgr, intrinsic_matrix, distortion, board) -> tuple[np.ndarray, np.ndarray] | None`, where the first matrix is `T_camera_marker` and the second image is the axis preview.
- Produces: CLI outputs `<name>_to_base.npy`, `<name>_intrinsics.npz`, and `<name>_preview.png`.

- [ ] **Step 1: Write the failing CLI/output-contract tests**

Test default argument values, default marker/output paths relative to `real-deployment`, output filename construction for `top_camera`, and invalid `--samples`/`--max-frames` values. Keep hardware outside these tests.

- [ ] **Step 2: Run tests and verify RED**

Run the test file. Expected: FAIL because the argument parser and output-path contract are missing.

- [ ] **Step 3: Implement the live command**

Implement this exact flow:

```text
enumerate RealSense serials
select requested/only serial
start BGR8 color stream
discard 30 warm-up frames
read active color intrinsics
capture until --samples valid ChArUco poses or --max-frames exhausted
compute one T_base_camera per valid pose
average transforms (one sample is unchanged)
save .npy, .npz, and last valid axis preview
print T_base_camera and YAML camera_params
stop pipeline in finally
```

Reject unavailable cameras, invalid marker transforms, insufficient valid frames, and failed image writes with clear exceptions.

- [ ] **Step 4: Run focused and existing tests**

Run:

```bash
python -m unittest real-deployment/tests/test_calibrate_fixed_realsense_to_base.py -v
python -m py_compile real-deployment/utils/calibrate_fixed_realsense_to_base.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 5: Check the hardware-facing help text**

Run:

```bash
python real-deployment/utils/calibrate_fixed_realsense_to_base.py --help
```

Expected: exits zero without requiring a connected RealSense and documents `--serial`, `--marker-to-base`, `--output-dir`, `--name`, stream dimensions, `--samples`, and `--max-frames`.

