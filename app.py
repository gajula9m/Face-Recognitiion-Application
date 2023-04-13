import cv2
import os
import numpy as np
from datetime import date
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
import io
from google.cloud import storage
import joblib
import shutil
import glob

# Intialize the face detector 
cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'data/haarcascade_frontalface_default.xml')

# print(haar_model)
face_detector = cv2.CascadeClassifier(haar_model)
# vid = cv2.VideoCapture(0)

faces_dir = 'faces/'

# Create directory to store faces
def init_dir():
    if not os.path.isdir(faces_dir):
        os.makedirs(faces_dir)


# Identify and get face from the given frame
def get_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.3, 5)
    return face_points

def add_student(name, img):
    name_folder = faces_dir + name
    if not os.path.isdir(name_folder):
        os.makedirs(name_folder)
    
    
    face = get_face(np.asarray(img))

    if len(face) == 1:
        num_img =  count_blobs_with_prefix("faces-bucket-3132022", name)
        img_path = name_folder + '/' + name + '_' + str(num_img) + '.png'
       
        x, y, w, h = face[0]
        cv2.imwrite(img_path, cv2.cvtColor(np.asarray(img)[y:y+h, x:x+w], cv2.COLOR_RGB2BGR))
        img_name = name + '_' + str(num_img)
        upload_blob("faces-bucket-3132022", img_path, name + '/' + img_name + '.png')

def train_model():
    print("Training model")
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
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(faces, labels)

    model_dir = 'models/'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    joblib.dump(knn, 'models/model.pkl')

def identify_face(img):
    model = joblib.load('models/model.pkl')
    faces = get_face(np.asarray(img))
    
    if len(faces) == 1:
        x,y,w,h = faces[0]
        resized_face = cv2.resize(cv2.cvtColor(np.asarray(img)[y:y+h, x:x+w], cv2.COLOR_RGB2BGR), (50,50))
        identified_person = model.predict(resized_face.reshape(1,-1))[0]

        name = identified_person
        return name

def upload_blob(bucket_name, source_file_name, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match = generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}"
    )

def count_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    return sum(1 for _ in blobs)

# 1. Add faces
# 2. Train model
# 3. Identify faces
# 4. If new faces are added, re train model


app = Flask(__name__)

@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def index():
    return render_template('index.html')

@app.route('/add_student', methods=['POST'])
def add_student_route():
    # get the data URL from the JSON payload
    name = request.json['name']
    data_url = request.json['dataUrl']
    
    # extract the base64-encoded image data from the data URL
    encoded_image_data = data_url.split(',')[1]
    
    # decode the base64-encoded image data into bytes
    decoded_image_data = base64.b64decode(encoded_image_data)
    
    # create a PIL Image object from the decoded image data
    image = Image.open(io.BytesIO(decoded_image_data))
    
    # save the image to disk
    add_student(name, image)
    
    # return a success response
    return 'Frames processed successfully'

@app.route('/identify_face', methods=['POST'])
def identify_face_route():
    
    data_url = request.json['dataUrl']
    encoded_image_data = data_url.split(',')[1]
    decoded_image_data = base64.b64decode(encoded_image_data)
    image = Image.open(io.BytesIO(decoded_image_data))

    name = identify_face(image)

    return jsonify({'name': name})

@app.route('/train_model', methods=['POST'])
def train_model_route():
    print('receive train model')
    train_model()
    return jsonify({'status': 'Model trained successfully'})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
