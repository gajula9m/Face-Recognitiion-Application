import cv2
import os
import numpy as np
import pandas as pd

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

face_detector = cv2.CascadeClassifier(haar_model)
vid = cv2.VideoCapture(0)


# Create directory to store faces
faces_dir = 'data/faces'

if not os.path.isdir(faces_dir):
    os.makedirs(faces_dir)
faces_dir = os.path.dirname(os.path.abspath(faces_dir))



def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points



while True:
    ret, frame = vid.read()
    face_points = get_face(frame)
    
    if type(face_points) != tuple :
        x, y, w, h = face_points[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
        # face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()



