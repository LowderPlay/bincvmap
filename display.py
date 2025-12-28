import random
import time

import pandas as pd
import numpy as np
from led import NUM_LEDS, send_wled_rgb
from scipy.spatial.transform import Rotation

points = pd.read_csv('corrected.csv', header=None).to_numpy().astype(np.float32)

RED = np.array([255, 0, 0])
GREEN = np.array([0, 255, 0])

def col_lerp(c1, c2, amount):
    return (c1 + (c2 - c1) * amount).astype(np.uint8)

while True:
    r = Rotation.from_euler('xyz', [random.randint(0, 180) for _ in range(3)], degrees=True)
    rotated = r.apply(points)[:, 1]
    p_min = rotated.min()
    p_max = rotated.max()
    normalized = (rotated - p_min) / (p_max - p_min)
    lin = list(np.linspace(0, 1, 100))
    for i in lin + list(reversed(lin)):
        cols = [col_lerp(RED, GREEN, (i - x)) if x < i else col_lerp(RED, GREEN, (x - i)) for x in normalized]
        send_wled_rgb(cols)
        time.sleep(0.01)