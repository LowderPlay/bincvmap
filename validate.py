import pandas as pd
import numpy as np

def validate(points):
    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    mean = np.median(distances[~np.isnan(distances)])
    maximum = mean / 0.75
    valid = np.pad(distances <= maximum, (1, 1), mode='constant', constant_values=False)
    valid = valid[:-1] | valid[1:]

    all_indices = np.arange(len(points))
    valid_indices = all_indices[valid]
    new_points = np.copy(points)
    for i in range(3):
        new_points[:, i] = np.interp(
            all_indices,
            valid_indices,
            points[valid_indices, i]
        )

    p_min = new_points.min()
    p_max = new_points.max()
    normalized = (new_points - p_min) / (p_max - p_min)
    return normalized

if __name__ == '__main__':
    points = pd.read_csv('3d.csv', header=None).to_numpy().astype(np.float32)
    pd.DataFrame(validate(points)).to_csv("corrected.csv", header=False, index=False)
