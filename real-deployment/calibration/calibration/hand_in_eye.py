import cv2
import numpy as np


MIN_VALID_HAND_EYE_SAMPLES = 3


class HandinEyeCalibrator:
    def __init__(self, intrinsics_matrix, dist_coeffs, charuco_dict, board):
        self.intrinsics_matrix = intrinsics_matrix
        self.dist_coeffs = dist_coeffs
        self.charuco_dict = charuco_dict
        self.board = board

    def estimate_pose(self, image):
        # calibration.utils.read_data() returns RGB images.
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.charuco_dict)

        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )  # can not pass
            if charuco_ids is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    self.board,
                    self.intrinsics_matrix,
                    self.dist_coeffs,
                    None,
                    None,
                )
                if valid:
                    return rvec, tvec
        return None, None

    def perform(
        self,
        rgb_list,
        pose_list,
        sample_ids=None,
        sample_center_ids=None,
        min_valid_samples_per_center=None,
    ):
        if len(rgb_list) != len(pose_list):
            raise ValueError(
                "RGB/robot-pose sample count mismatch: "
                f"{len(rgb_list)} != {len(pose_list)}."
            )

        if sample_ids is None:
            sample_ids = list(range(len(rgb_list)))
        else:
            sample_ids = list(sample_ids)
            if len(sample_ids) != len(rgb_list):
                raise ValueError(
                    "sample_ids/RGB sample count mismatch: "
                    f"{len(sample_ids)} != {len(rgb_list)}."
                )

        if sample_center_ids is None:
            sample_center_ids = [None] * len(rgb_list)
        else:
            sample_center_ids = list(sample_center_ids)
            if len(sample_center_ids) != len(rgb_list):
                raise ValueError(
                    "sample_center_ids/RGB sample count mismatch: "
                    f"{len(sample_center_ids)} != {len(rgb_list)}."
                )
        if min_valid_samples_per_center is not None:
            min_valid_samples_per_center = int(min_valid_samples_per_center)
            if min_valid_samples_per_center <= 0:
                raise ValueError("min_valid_samples_per_center must be positive.")
            if any(center_id is None for center_id in sample_center_ids):
                raise ValueError(
                    "sample_center_ids are required when center filtering is enabled."
                )

        # Initialize lists to store data
        valid_records = []
        failed_sample_ids = []

        # Loop over each data pair
        for sample_id, center_id, rgb, pose in zip(
            sample_ids,
            sample_center_ids,
            rgb_list,
            pose_list,
        ):
            # Load pose data
            R_gripper2base = pose[0:3, 0:3]
            t_gripper2base = pose[0:3, 3]

            # Estimate pose from RGB image
            rvec, tvec = self.estimate_pose(rgb)
            if rvec is None or tvec is None:
                failed_sample_ids.append(sample_id)
                continue

            R_target2cam = cv2.Rodrigues(rvec)[0]
            t_target2cam = tvec.reshape(3, 1)

            valid_records.append(
                (
                    sample_id,
                    center_id,
                    R_gripper2base,
                    t_gripper2base,
                    R_target2cam,
                    t_target2cam,
                )
            )

        total_sample_count = len(rgb_list)
        print(f"hand_eye_detected_samples={len(valid_records)}/{total_sample_count}")
        print(f"hand_eye_failed_frames={failed_sample_ids}")

        dropped_center_ids = []
        dropped_sample_ids = []
        if min_valid_samples_per_center is not None:
            center_counts = {
                center_id: 0 for center_id in set(sample_center_ids)
            }
            for record in valid_records:
                center_counts[record[1]] += 1
            dropped_center_ids = sorted(
                center_id
                for center_id, count in center_counts.items()
                if count < min_valid_samples_per_center
            )
            dropped_center_id_set = set(dropped_center_ids)
            dropped_sample_ids = [
                record[0]
                for record in valid_records
                if record[1] in dropped_center_id_set
            ]
            valid_records = [
                record
                for record in valid_records
                if record[1] not in dropped_center_id_set
            ]

        print(f"hand_eye_dropped_centers={dropped_center_ids}")
        print(f"hand_eye_dropped_frames={dropped_sample_ids}")
        valid_sample_count = len(valid_records)
        print(f"hand_eye_valid_samples={valid_sample_count}/{total_sample_count}")
        if valid_sample_count < MIN_VALID_HAND_EYE_SAMPLES:
            raise ValueError(
                "Hand-eye calibration requires at least "
                f"{MIN_VALID_HAND_EYE_SAMPLES} valid paired samples; "
                f"got {valid_sample_count}/{total_sample_count}."
            )

        # Convert lists to arrays
        R_gripper2base_array = np.array([record[2] for record in valid_records])
        t_gripper2base_array = np.array([record[3] for record in valid_records])
        R_target2cam_array = np.array([record[4] for record in valid_records])
        t_target2cam_array = np.array([record[5] for record in valid_records])

        # Optional initial guess for camera-to-gripper transformation
        R_cam2gripper_guess = np.eye(3)
        t_cam2gripper_guess = np.zeros((3, 1))

        # Perform hand-eye calibration
        R_cam2gripper_avg, t_cam2gripper_avg = cv2.calibrateHandEye(
            R_gripper2base_array,
            t_gripper2base_array,
            R_target2cam_array,
            t_target2cam_array,
            R_cam2gripper_guess,
            t_cam2gripper_guess,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

        # Return the average results
        return R_cam2gripper_avg, t_cam2gripper_avg
