import logging
import os
import json
import time
from datetime import datetime 
import uuid 
from flask import Flask, request, jsonify
import cv2
from deepface import DeepFace
import numpy as np
from flask_cors import CORS
import base64
#from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


# Use 0 for the default webcam
video_capture = cv2.VideoCapture(0)

# Use the open-source haar_cascade classifier.
# TODO: use an alternative classifier to compare effectiveness
haar_cascade = cv2.CascadeClassifier('../classifier/haar_face.xml')

@app.route('/emotion', methods=['POST'])
def analyze_emotion():
    frames = request.files.getlist('frames')
    emotions_data = []

    for frame in frames:
        try:
            nparr = np.fromstring(frame.read(), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            height, width = frame.shape[:2]
            new_width = int(width * 0.5)
            new_height = int(height * 0.5)
            resized_frame = cv2.resize(frame, (new_width, new_height))

            result = DeepFace.analyze(img_path=resized_frame, actions=['emotion'], enforce_detection=False)

            gray_video = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray_video, 1.1, 7)

            for (x, y, w, h) in faces:
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                emotions_data.append({'face': [x.item(), y.item(), w.item(), h.item()]})

            emotion = result[0]['dominant_emotion']
            probability = result[0]['emotion'][emotion]
            emotions_data.append({'frame': image_to_base64(resized_frame), 'emotion': emotion, 'probability': probability})

        except Exception as e:
            print(f'Error processing frame: {e}')

    return jsonify(emotions_data)

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_frame = base64.b64encode(buffer).decode('utf-8')
    return base64_frame

json_filename = "data.json"

@app.route('/submit_form', methods=['POST'])
def submit_form():
    data = request.json
    print('data nah: ', data)

    # Add timestamp and unique ID fields
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())
    data['timestamp'] = format_readable_timestamp(timestamp)
    data['user_id'] = unique_id

    # Load existing data from the JSON file (if any)
    existing_data = []
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            existing_data = json.load(json_file)

    # Append the new form data to the existing data
    existing_data.append(data)

    # Save the updated data back to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    try:
        response = {"message": "Form data received and processed successfully."}
        return jsonify(response), 200
    except Exception as e:
        error_message = str(e)
        return jsonify({"error": error_message}), 500
    
def format_readable_timestamp(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_timestamp = dt_object.strftime('%Y-%b-%d-%H:%M:%S')
    return formatted_timestamp
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)