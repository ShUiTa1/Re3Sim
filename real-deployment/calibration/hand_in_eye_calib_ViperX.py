"""
Hand-in-eye calibration entry script.

This script consumes the calibration dataset collected by
``hand_in_eye_shooting.ipynb`` and computes ``cam_to_hand_pose.npy``.

High-level purpose:
1. Load RGB/depth images, camera intrinsics, robot poses, and optionally joints.
2. Convert robot-specific pose records into a common "hand/gripper -> base"
   homogeneous transform expected by OpenCV's hand-eye calibration API.
3. Detect the ChArUco calibration board in each RGB image.
4. Solve the fixed transform from the wrist camera frame to the robot hand frame.

Important migration note for ViperX:
- Everything below that constructs ``rtb.models.Panda()`` or uses the end-effector
  name ``"panda_hand"`` is Franka/Panda-specific.
- For ViperX, replace those functions with an adapter that returns the same
  semantic quantity: a 4x4 transform ``T_hand_to_base`` for every captured frame.
- If you use LeRobot for hardware IO, LeRobot should provide/record joint states
  or end-effector state. If you use roboticstoolbox for kinematics, build/load a
  ViperX model and make ``joint_to_hand`` call that model's FK with the correct
  ViperX end-effector link name.
"""

import numpy as np
import os
import sys
import open3d as o3d
import cv2
from tqdm import trange
import roboticstoolbox as rtb

# The calibration package lives two directories above the current working
# directory when this script is launched from ``real-deployment/calibration``.
# Adding that path keeps the original repo layout working without installing
# the package. If you later package this code properly, replace this with an
# editable install or an absolute project import.
parent_dir = os.path.dirname(os.getcwd())
parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)

from calibration.hand_in_eye import HandinEyeCalibrator
from calibration.utils import read_data


def tcp_to_hand(pose):
    """Convert a Panda TCP pose into the Panda hand-link pose.

    Why this exists:
    - Some robot APIs log the pose of a "TCP" frame, while OpenCV calibration is
      called below with a "gripper/hand" frame. Those frames are not always the
      same. On Franka/Panda, the code wants ``panda_hand``.
    - The original script recovers a Panda joint vector with IK and then runs FK
      to the ``panda_hand`` link. That is a Franka-specific workaround.

    Input:
    - ``pose``: expected to be a 4x4 homogeneous transform for the Panda TCP in
      the robot base frame. A homogeneous transform stores rotation in the upper
      left 3x3 block and translation in the last column.

    Output:
    - ``hand_pose``: 4x4 transform ``T_hand_to_base`` used by
      ``cv2.calibrateHandEye`` as gripper-to-base data.

    ViperX replacement point:
    - If your ViperX API logs TCP pose directly, define the exact frame name:
      is it the flange, wrist, tool tip, or camera mount?
    - Then either return it directly if it already represents your calibration
      "hand" frame, or multiply by a fixed TCP-to-hand transform.
    - Avoid using the Panda IK/FK path here; load a ViperX model or use your own
      frame transform convention.
    """
    panda = rtb.models.Panda()
    joints = panda.ik_LM(pose)[0]
    hand_pose = panda.fkine(joints, end="panda_hand").A
    return hand_pose


def joint_to_hand(joints):
    """Convert logged Panda joint angles into a Panda hand-link pose.

    Why this path is preferred when available:
    - Joint logs are usually the most reproducible source of robot state.
    - Forward kinematics (FK) maps joint angles to an end-effector transform.
    - The output frame is explicitly selected with ``end="panda_hand"``.

    Input:
    - ``joints``: Panda joint vector saved by the shooting notebook from
      ``panda.get_log()["q"][-1]``.

    Output:
    - 4x4 transform ``T_hand_to_base``. This is the pose of the hand/gripper
      frame expressed in the robot base frame.

    ViperX replacement point:
    - Replace ``rtb.models.Panda()`` with your ViperX kinematic model.
    - Replace ``"panda_hand"`` with the ViperX link that you want to define as
      the "hand" frame for calibration.
    - Make sure the joint order, units, zero offsets, and link naming match the
      values recorded by LeRobot or your ViperX driver.
    """
    panda = rtb.models.Panda()
    hand_pose = panda.fkine(joints, end="panda_hand").A
    return hand_pose


# Read data from the calibration capture folder.
#
# Expected folder structure:
#   base_dir/
#     rgb/              RGB PNG images containing the ChArUco board
#     depth/            depth .npy files, mainly for visualization/TSDF checks
#     poses/            4x4 robot pose .npy files saved during capture
#     joints/           optional joint .npy files saved during capture
#     rgb_intrinsics.npz
#     depth_intrinsics.npz
#
# The calibration solve only needs RGB images, RGB intrinsics, distortion
# coefficients, and robot hand poses. Depth is loaded by ``read_data`` because
# the notebook also uses it for point-cloud/TSDF visualization.
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, help="path to data root directory")
args = parser.parse_args()
base_dir = args.data_root
(
    rgb_list,
    depth_list,
    pose_list,
    rgb_intrinsics,
    rgb_coeffs,
    depth_intrinsics,
    depth_coeffs,
    depth_scale,
    joints_list,
) = read_data(base_dir)

# Convert the raw robot records into the exact pose convention required by
# OpenCV:
#   R_gripper2base, t_gripper2base
#
# In this repository "gripper" is effectively the Panda hand frame. For your
# ViperX port, keep the downstream name if convenient, but make the semantic
# quantity identical: ``T_hand_to_base`` for each captured image.
if joints_list is not None:
    # Preferred original path: use logged Panda joints and FK to the Panda hand.
    # ViperX port: use ViperX FK from the logged ViperX joint vector.
    pose_list = [joint_to_hand(joints) for joints in joints_list]
else:
    # Fallback original path: convert logged TCP pose to Panda hand pose via
    # Panda IK + FK. This is more robot-specific and should be replaced for
    # ViperX unless your recorded pose is already the desired hand frame.
    pose_list = [tcp_to_hand(pose) for pose in pose_list]

print(f"{len(rgb_list)} poses found")
print(f"Camera matrix: {rgb_intrinsics}")

# Build the ChArUco board model used in the images.
#
# ChArUco = ArUco markers + chessboard corners. ArUco IDs make detection robust,
# while chessboard corners provide accurate sub-pixel geometry.
#
# Parameters here must match the physical printed board:
# - DICT_6X6_250: ArUco dictionary family.
# - (5, 5): number of chessboard squares in X and Y.
# - 0.04: square length in meters.
# - 0.03: marker length in meters.
#
# If your printed board differs, update these values before trusting the result.
charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)

# ``HandinEyeCalibrator.perform`` detects the board in every RGB image and calls
# ``cv2.calibrateHandEye`` with:
# - robot motion: hand/gripper -> base
# - vision motion: calibration target -> camera
#
# The returned transform is camera -> hand. It is fixed as long as the camera
# mount does not move relative to the wrist/hand.
calibrator = HandinEyeCalibrator(rgb_intrinsics, rgb_coeffs, charuco_dict, board)
R_cam2hand_avg, t_cam2hand_avg = calibrator.perform(rgb_list, pose_list)

print("Average Camera to hand rotation matrix:")
print(R_cam2hand_avg)
print("Average Camera to hand translation vector:")
print(t_cam2hand_avg)

# Pack rotation and translation into a standard 4x4 homogeneous transform.
# This matrix maps a point represented in the camera frame into the hand frame:
#
#   p_hand = cam_to_hand_pose @ p_camera
#
# Downstream reconstruction/alignment code can then chain it with
# ``T_hand_to_base`` to get camera poses in the robot base frame.
cam_to_hand_pose = np.eye(4)
cam_to_hand_pose[:3, :3] = R_cam2hand_avg
cam_to_hand_pose[:3, 3] = t_cam2hand_avg.squeeze()
print(f"Camera to hand pose:\n{cam_to_hand_pose}")

# Saved artifact consumed by later calibration/alignment steps.
# For ViperX, the filename can stay the same if your downstream code only cares
# that it represents camera -> hand for your chosen ViperX hand frame.
np.save(f"{base_dir}/cam_to_hand_pose.npy", cam_to_hand_pose)
