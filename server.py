import logging
import os
import json
import time
from datetime import datetime  
import threading
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


haar_cascade = cv2.CascadeClassifier('../classifier/haar_face.xml')

streaming_active = False

# Emotion timeline data structure
emotion_timeline = []
emotion_streams = []
json_filename = "./data.json"

def video_processing():
    global streaming_active
    video_capture = cv2.VideoCapture(0)
    haar_cascade = cv2.CascadeClassifier('../classifier/haar_face.xml')

    while streaming_active:
        ret, frame = video_capture.read()

        # Resize the frame to 1/4 of its original size
        height, width = frame.shape[:2]
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        resized_frame = cv2.resize(frame, (new_width, new_height))

        result = DeepFace.analyze(img_path=resized_frame, actions=['emotion'], enforce_detection=False)

        # Convert video stream to gray
        gray_video = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray_video, 1.1, 4)

        # Loop through points numpy array in faces and construct a facial rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Extract dominant emotion and probability
        emotion = result[0]['dominant_emotion']
        probability = result[0]['emotion'][emotion]
        txt = f"{emotion} (Probability: {probability:.2f})"

        cv2.putText(resized_frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Display the frame with emotion
        cv2.imshow('Resized Frame', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            streaming_active = False

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global streaming_active
    if not streaming_active:
        streaming_active = True
        video_processing_thread = threading.Thread(target=video_processing)
        video_processing_thread.start()
        return "Streaming started", 200
    else:
        return "Streaming is already active", 400


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
            # Append emotion and timestamp to the timeline data structure
            timestamp = time.time()

            result = DeepFace.analyze(img_path=resized_frame, actions=['emotion'], enforce_detection=False)

            gray_video = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            faces = haar_cascade.detectMultiScale(gray_video, 1.1, 7)
            print(type(resized_frame), resized_frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                emotions_data.append({'face': [x.item(), y.item(), w.item(), h.item()]})

            emotion = result[0]['dominant_emotion']
            probability = result[0]['emotion'][emotion]
            emotions_data.append({'frame': image_to_base64(resized_frame), 'emotion': emotion, 'probability': probability})
            print(f"Timestamp: {timestamp}, Emotion: {emotion} ")

        except Exception as e:
            print(f'Error processing frame: {e}')

    # return jsonify(emotions_data)
    return jsonify(emotions_data)

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_frame = base64.b64encode(buffer).decode('utf-8')
    return base64_frame

@app.route('/get_data', methods=['GET'])
def get_data():
    if emotion_streams:
        return jsonify(emotion_streams)
    else:
        return "No Data to be returned", 400

@app.route('/submit_form', methods=['POST'])
def submit_form():
    data = request.json

    # Add timestamp and unique ID fields
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())
    data['timestamp'] = format_readable_timestamp(timestamp)
    data['user_id'] = unique_id

    # Load existing data from the JSON file (if any)
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []
    try:
        existing_data.append(data)
    except Exception as e:
        str_ms = str(e)
        print('Error Message: ', str_ms)

    # Save the updated data back to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    try:
        response = {"message": "Form data received and processed successfully."}
        return jsonify(response), 200
    except Exception as e:
        error_message = str(e)
        print("Error:", error_message)
        return jsonify({"error": error_message}), 500
    
def format_readable_timestamp(timestamp):
    dt_object = datetime.fromtimestamp(timestamp)
    formatted_timestamp = dt_object.strftime('%Y-%b-%d-%H:%M:%S')
    return formatted_timestamp

# Print the emotion timeline
for entry in emotion_timeline:
    print(f"Timestamp: {format_readable_timestamp(entry['timestamp'])}, Emotion: {entry['emotion']}")
        

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)