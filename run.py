from flask import Flask, render_template, Response
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

app = Flask(__name__)

# Load models
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model_100epochs.h5')
age_model = load_model('age_model_50epochs.h5')
gender_model = load_model('gender_model_50epochs.h5')

# Labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Preprocess for emotion detection
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Emotion Prediction
            preds = emotion_model.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Preprocess for gender and age detection
            roi_color = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(roi_color, (200, 200), interpolation=cv2.INTER_AREA)
            roi_color = np.array(roi_color).reshape(-1, 200, 200, 3)

            # Gender Prediction
            gender_predict = gender_model.predict(roi_color)
            gender_predict = (gender_predict >= 0.5).astype(int)[:, 0]
            gender_label = gender_labels[gender_predict[0]]
            gender_label_position = (x, y + h + 25)
            cv2.putText(frame, gender_label, gender_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Age Prediction
            age_predict = age_model.predict(roi_color)
            age = round(age_predict[0, 0])
            age_label_position = (x, y + h + 50)
            cv2.putText(frame, f"Age={age}", age_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
