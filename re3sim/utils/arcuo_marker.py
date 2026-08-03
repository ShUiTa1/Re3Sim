import cv2
import numpy as np


def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if (
        charuco_corners is not None
        and charuco_ids is not None
        and len(charuco_corners) > 3
    ):
        object_points, image_points = board.matchImagePoints(
            charuco_corners, charuco_ids
        )
        valid, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if valid and np.isfinite(rvec).all() and np.isfinite(tvec).all():
            R_target2cam = cv2.Rodrigues(rvec)[0]
            t_target2cam = tvec.reshape(3, 1)
            target2cam = np.eye(4)
            target2cam[:3, :3] = R_target2cam
            target2cam[:3, 3] = t_target2cam.reshape(-1)
            return np.linalg.inv(target2cam)
    return None


if __name__ == "__main__":
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((5, 5), 0.04, 0.03, charuco_dict)
