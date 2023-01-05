import cv2
import os
import numpy as np
import pandas as pd

# cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
# haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# face_detector = cv2.CascadeClassifier(haar_model)
vid = cv2.VideoCapture(0)

while True:
    ret, frame = vid.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()

cv2.destroyAllWindows()