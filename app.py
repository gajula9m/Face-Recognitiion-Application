import cv2
import os
import numpy as np
from datetime import date
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask
from google.cloud import storage
import joblib
import shutil

# Intialize the face detector 
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# print(haar_model)
face_detector = cv2.CascadeClassifier(haar_model)
# vid = cv2.VideoCapture(0)

faces_dir = 'faces/'
attendance_dir = 'attendance/'


# Get Date for Attendance
def get_date():
    return date.today.strftime("%m_%d_%y")

# Create directory to store faces
def init_dir():
    if not os.path.isdir(faces_dir):
        os.makedirs(faces_dir)
    if not os.path.isdir(attendance_dir):
        os.makedirs(attendance_dir)

    current_day_csv = f'Attendance_({get_date()}.csv'
    if current_day_csv not in os.listdir('Attendance'):
        with open(f'Attendance/{current_day_csv}', 'w') as file:
            file.write('Name, ID, Time')


# Identify and get face from the given frame
def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

# Add the face to be saved for future model training
def add_face():

    # Get name from user  and create a directory for them
    name = input('Enter a name: ')
    name_folder = faces_dir + name
    if not os.path.isdir(name_folder):
        os.makedirs(name_folder)

    # Start capturing their image
    vid = cv2.VideoCapture(0)
    
    num_img, num_frame = 0, 0
    while 1:
        ret, frame = vid.read()
        face = get_face(frame)

        print(len(face))

        # make sure there is only one face in the image screen 
        if len(face) > 1:
            # handle error properly
            print('Too many faces. Please add one face for the user.')
            break
        else:
            # if no face found, continue looking for face
            if type(face) == tuple:
                continue
            
            # Capture and save 10 images of the user's face
            x, y, w, h = face[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {num_img}/10', (30, 30), 
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 20),
                            2, cv2.LINE_AA)

            if num_frame%10 == 0:
                img_name = name + '_' + str(num_img) + '.jpg'
                cv2.imwrite(name_folder + '/' + img_name, frame[y:y+h, x:x+w])
                upload_blob("faces-bucket-3132022", name_folder + '/' + img_name, name + '/' + img_name )
                num_img+=1
            num_frame+=1

            cv2.imshow('Frame', frame)

            if num_img == 7:
                break
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    shutil.rmtree(name_folder)
    vid.release()
    cv2.destroyAllWindows()

    # train_model()
    # print("Trained model")

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('faces')
    for user in userlist:
        for img_file in os.listdir(f'faces/{user}/'):
            img = cv2.imread(f'faces/{user}/{img_file}')
            resized_img = cv2.resize(img, (50, 50))
            faces.append(resized_img.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)

    model_dir = 'models/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    joblib.dump(knn, 'models/model.pkl')

def identify_face():
    model = joblib.load('models/model.pkl')
    cap = cv2.VideoCapture(0, )
    ret = True
    while ret:
        ret, frame = cap.read()
        faces = get_face(frame)

        if type(faces) == tuple:
            continue
        
        for face in faces:
            x, y, w, h = face

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            resized_face = cv2.resize(frame[y:y+h, x:x+w], (50,50))
            identified_person = model.predict(resized_face.reshape(1,-1))[0]

            print(identified_person)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.releaseAllWindows()


def upload_blob(bucket_name, source_file_name, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match = generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}"
    )


# 1. Add faces
# 2. Train model
# 3. Identify faces
# 4. If new faces are added, re train model


app = Flask(__name__)

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'

if __name__ == '__main__':
    add_face()
    # identify_face()
    # app.run()
