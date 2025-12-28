import math

import cv2
import time

import numpy as np
import pandas as pd

from led import NUM_LEDS, send_wled_states
from triangulate import triangulate
from validate import validate

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2800)

CAMERA_DELAY = 0.5 # sec
BASE = 2

if not cap.isOpened():
    print("Cannot open camera")
    quit()


def n_digit(a, n, b):
    for i in range(1, n + 1):
        a = a // b

    return a % b


def get_frame(black):
    ret, frame = cap.read()
    # frame = cv2.medianBlur(frame,5)
    # black = cv2.medianBlur(black,5)
    frame = cv2.subtract(frame, black)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # b, g, r = cv2.split(frame)
    # block_size = 9
    # c = -2
    # b = cv2.adaptiveThreshold(b,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,c)
    # g = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,c)
    # r = cv2.adaptiveThreshold(r,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,c)
    # frame = cv2.merge([b, g, r])
    frame = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    frame = cv2.medianBlur(frame, 5)
    frame = frame.astype(np.float32) / 255
    cv2.imshow("filter", frame)
    cv2.waitKey(1)
    return frame


def pred_pixel(digits, black, count_digits, index):
    final_image = np.full(black.shape, 1, dtype=np.float32)
    for i in range(count_digits):
        color = n_digit(index, i, BASE)
        target = digits[i] if color else cv2.bitwise_not(digits[i])
        final_image = cv2.bitwise_and(final_image, target)
    return cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)


def run_webcam():
    send_wled_states(False for _ in range(NUM_LEDS))
    time.sleep(CAMERA_DELAY)
    ret, black = cap.read()
    print(black.shape)

    count_digits = math.ceil(math.log(NUM_LEDS, BASE))

    digits = []
    for i in range(count_digits):
        send_wled_states(n_digit(x, i, BASE) for x in range(NUM_LEDS))
        time.sleep(CAMERA_DELAY)
        frame = get_frame(black)
        digits.append(frame)

    cv2.destroyAllWindows()

    coords = []
    frames = []
    for i in range(NUM_LEDS):
        frame = pred_pixel(digits, black, count_digits, i)
        coord = bright_spot(frame)
        coords.append(coord)
        frames.append(frame)

    # print("done")
    #
    # for i in range(NUM_LEDS):
    #     coord = coords[i]
    #     send_wled_states(x == i for x in range(NUM_LEDS))
    #     time.sleep(CAMERA_DELAY)
    #     ret, real = cap.read()
    #
    #     if coord is not None:
    #         real = cv2.circle(real, coord, 8, (0,0,255), 1)
    #
    #     cv2.imshow("final_real", real)
    #     cv2.imshow("final_pred", frames[i])
    #     cv2.waitKey(1)

    return np.array(coords)

def bright_spot(gray_image):
    gray_image = np.array(gray_image * 255, dtype=np.uint8)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.nan, np.nan

    biggest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(biggest_contour)
    if M["m00"] == 0:
        return np.nan, np.nan

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


if __name__ == '__main__':
    coords1 = run_webcam()
    input("Press Enter to take next frame")
    coords2 = run_webcam()
    cap.release()
    cv2.destroyAllWindows()

    triangulated = triangulate(coords1, coords2)
    validated = validate(triangulated)
    pd.DataFrame(validated).to_csv("result.csv", header=False, index=False)

