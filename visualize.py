import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

points = pd.read_csv('result.csv', header=None).dropna().to_numpy().astype(np.float32)

# for i, (x, y, z) in enumerate(points):
#     ax.text(x, y, z, str(i), fontsize=8, color="red")

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]
ax.scatter(x, y, z, c=z, marker='o', s=20, alpha=0.8)
ax.plot(x, y, z, c='b')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis (Depth)')
ax.set_title('3D Point Cloud Reconstruction')

ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])

plt.show()