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

haar_model = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(haar_model)

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
    
    face = get_face(np.asarray(img))

    if len(face) == 1:
        num_img =  count_blobs_with_prefix("faces-bucket-3132022", name)
       
        x, y, w, h = face[0]
        image_np = cv2.cvtColor(np.asarray(img)[y:y+h, x:x+w], cv2.COLOR_RGB2BGR)
        retval, image_bytes = cv2.imencode('.png', image_np)
        img_name = name + '_' + str(num_img)
        upload_blob("faces-bucket-3132022", image_bytes.tobytes(), name + '/' + img_name + '.png')

def train_model():
    # print("Training model")
    faces = []
    labels = []
    # userlist = os.listdir('faces')
    # for user in userlist:
    #     for img_file in os.listdir(f'faces/{user}/'):
    #         img = cv2.imread(f'faces/{user}/{img_file}')
    #         resized_img = cv2.resize(img, (50, 50))
    #         faces.append(resized_img.ravel())
    #         labels.append(user)

    blobs = getBlobs("faces-bucket-3132022")

    for blob in blobs:

        name = blob.name
        if name != "model.pkl":
            image_bytes = blob.download_as_bytes()
            print(name)
            image_np = cv2.resize(cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR), (50,50))
            labels.append(name.split("/")[0])
            faces.append(image_np.ravel())


    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(faces, labels)

    # model_dir = 'models/'
    # if not os.path.isdir(model_dir):
    #     os.makedirs(model_dir)

    # joblib.dump(knn, 'models/model.pkl')

    model_stream = io.BytesIO()
    joblib.dump(knn, model_stream)
    model_stream.seek(0)

    upload_model("faces-bucket-3132022", model_stream.getvalue(), 'model.pkl')

def identify_face(img):

    model_stream = download_model("faces-bucket-3132022", 'model.pkl')

    model = joblib.load(model_stream)
    faces = get_face(np.asarray(img))
    
    if len(faces) == 1:
        x,y,w,h = faces[0]
        resized_face = cv2.resize(cv2.cvtColor(np.asarray(img)[y:y+h, x:x+w], cv2.COLOR_RGB2BGR), (50,50))
        identified_person = model.predict(resized_face.reshape(1,-1))[0]

        name = identified_person
        return name

def upload_model(bucket_name, string, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    if blob.exists():
        blob.reload()
        blob.upload_from_string(string, if_generation_match = blob.generation)
    else:
        blob.upload_from_string(string)

def upload_blob(bucket_name, string, destination_blob_name):

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    generation_match_precondition = 0

    blob.upload_from_string(string, content_type='image/png', if_generation_match = generation_match_precondition)

    # print(
    #     f"File {source_file_name} uploaded to {destination_blob_name}"
    # )

def download_model(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    model_stream = io.BytesIO()
    blob.download_to_file(model_stream)
    model_stream.seek(0)
    return model_stream

def download_blob(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()

def count_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    storage_client = storage.Client()

    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    return sum(1 for _ in blobs)

def getBlobs(bucket_name):
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    return blobs


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
    
    # # extract the base64-encoded image data from the data URL
    encoded_image_data = data_url.split(',')[1]
    
    # # decode the base64-encoded image data into bytes
    decoded_image_data = base64.b64decode(encoded_image_data)
    
    # # create a PIL Image object from the decoded image data
    image = Image.open(io.BytesIO(decoded_image_data))
    
    # # save the image to disk
    add_student(name, image)
    
    # return a success response
    return jsonify({'status':'Frames processed successfully'})

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
    # print('receive train model')
    train_model()
    return jsonify({'status': 'Model trained successfully'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
