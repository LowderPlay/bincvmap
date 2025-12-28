import cv2
import numpy as np
import pandas as pd

w, h = (640, 480)

def triangulate(points1, points2):
    valid_indices = ~np.isnan(points1).any(axis=1) & ~np.isnan(points2).any(axis=1)
    points1 = points1[valid_indices]
    points2 = points2[valid_indices]

    f = w  # guessing
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0, 1]], dtype=np.float32)

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # Convert Fundamental Matrix to Essential Matrix
    E = K.T @ F @ K

    _, R, t, _ = cv2.recoverPose(E, points1, points2, K)
    extrinsic1 = np.eye(3, 4)
    proj_matrix1 = K @ extrinsic1
    extrinsic2 = np.hstack((R, t))
    proj_matrix2 = K @ extrinsic2
    pts1 = points1.reshape(-1, 2).T
    pts2 = points2.reshape(-1, 2).T
    points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, pts1, pts2)

    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T  # Transpose to get (N, 3)

    final_3d_cloud = np.full((valid_indices.shape[0], 3), np.nan)
    print(final_3d_cloud.shape)
    print(points_3d.min(), points_3d.max())

    final_3d_cloud[valid_indices] = points_3d
    return final_3d_cloud


if __name__ == '__main__':
    points1 = pd.read_csv('cords1.csv', header=None).to_numpy().astype(np.float32)
    points2 = pd.read_csv('cords2.csv', header=None).to_numpy().astype(np.float32)

    pd.DataFrame(triangulate(points1, points2)).to_csv("3d.csv", header=False, index=False)