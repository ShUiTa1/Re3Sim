# Fixed RealSense Camera Calibration Design

## Goal

Calibrate one fixed Intel RealSense camera, initially `top_camera`, directly
against the accepted ViperX base frame. The operator only needs to keep the
fixed ChArUco board visible and run one command; no manual position or angle
measurement is required.

## Existing conventions to reuse

- The accepted board is a 5 x 5 ChArUco board using `DICT_6X6_250`, 36 mm
  squares, and 27 mm markers.
- `marker_2_base.npy` is `T_base_marker`: it maps marker-frame points into the
  ViperX base frame.
- OpenCV pose estimation produces `T_camera_marker`; its inverse is
  `T_marker_camera`.
- The runtime fixed-camera extrinsic is `T_base_camera`, because the ViperX
  base is the Isaac world frame.

Therefore the required composition is:

```text
T_base_camera = T_base_marker @ inverse(T_camera_marker)
```

## Script

Replace the stale fixed-side-camera utility with:

```text
real-deployment/utils/calibrate_fixed_realsense_to_base.py
```

The script will:

1. Select a RealSense by `--serial`, or select it automatically when exactly
   one RealSense is connected.
2. Start the configured RGB stream, defaulting to 640 x 480 at 30 FPS.
3. Discard warm-up frames so exposure can settle.
4. Read the active RGB stream's factory intrinsics and distortion values.
5. Capture the first frame with a valid ChArUco pose. `--samples` remains an
   optional averaging control and defaults to one. If increased, translations
   use their arithmetic mean and rotations use an SO(3) mean.
6. Load and validate `marker_2_base.npy`.
7. Compute and save `T_base_camera`.
8. Draw the detected board axes into a preview image for visual verification.
9. Print the exact `camera_params` list expected by the ViperX YAML.

The script must always stop the RealSense pipeline, including after errors.

## Inputs

Required or defaulted command-line inputs:

- `--serial`: required only when more than one RealSense is connected.
- `--marker-to-base`: path to `marker_2_base.npy`, defaulting to the existing
  `calibration/data/viperx_hand_eye` result.
- `--output-dir`: defaults to the same `viperx_hand_eye` directory.
- `--name`: defaults to `top_camera` and controls output filenames.
- `--width`, `--height`, `--fps`: default to 640, 480, and 30.
- `--samples`: defaults to one valid pose.
- `--max-frames`: defaults to 300 captured frames before reporting that the
  board could not be detected.

## Outputs

For the default camera name, the script writes:

```text
top_camera_to_base.npy
top_camera_intrinsics.npz
top_camera_preview.png
```

`top_camera_to_base.npy` is exactly `T_base_camera` and can be assigned directly
to `top_camera.extrinsic`. The intrinsics archive contains `fx`, `fy`, `ppx`,
`ppy`, distortion coefficients, width, and height.

## Failure behavior

The command exits with a clear error when:

- no RealSense is connected;
- multiple cameras are present without `--serial`;
- the requested serial is unavailable;
- the marker-to-base file is missing or is not a finite 4 x 4 transform;
- no valid ChArUco pose is found within the capture limit; or
- an output cannot be written.

No weak pose is silently saved.

## Verification

Pure tests will cover transform direction, transform validation, RealSense
selection, and the exact YAML camera-parameter ordering. Hardware acceptance is
one laboratory run that produces all three outputs and shows the drawn axes
attached to the physical ChArUco board.
