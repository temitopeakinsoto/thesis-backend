import cv2
from deepface import DeepFace

# Use 0 for the default webcam
video_capture = cv2.VideoCapture(0)

haar_cascade = cv2.CascadeClassifier('../classifier/haar_face.xml')

# Emotion timeline data structure
emotion_timeline = [] 

while True:
    # Read a frame from the webcam
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

    # Append emotion to the timeline data structure
    emotion_timeline.append(emotion)

    cv2.putText(resized_frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display the frame with emotion
    cv2.imshow('Resized Frame', resized_frame)

    if cv2.waitKey(1) & 0xff == ord('q'):  # Exit if 'q' is pressed
        break

# Close the webcam window
cv2.destroyAllWindows()



