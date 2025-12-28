
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE, 300)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 2800)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break