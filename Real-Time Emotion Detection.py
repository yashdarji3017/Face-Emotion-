# emotion_detection_live.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('D:\Python\emotion_recognition_model.h5')

# Emotion labels corresponding to the output classes
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture from webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert frame to grayscale (as our model expects grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame using Haar Cascade Classifier
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for emotion detection
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255.0   # Normalize
        
        # Reshape for prediction (1 sample of shape (48x48x1))
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        
        # Predict emotion
        predictions = model.predict(roi_gray)
        emotion_index = np.argmax(predictions[0])
        emotion_label = emotion_labels[emotion_index]
        
        # Draw rectangle around face and put text for emotion label
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, emotion_label,(x,y-10), cv2.FONT_HERSHEY_SIMPLEX , 0.9,(255,0,0),2)

    # Display the resulting frame with detected emotions
    cv2.imshow('Live Emotion Detection', frame)
    
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
