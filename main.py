import tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from datetime import datetime

face_classifier = cv2.CascadeClassifier(r'C:\Users\barto\PycharmProjects\facialemotionrecognizerinrealtime\haarcascade_frontalface_default.xml')
classifier = load_model(r'C:\Users\barto\PycharmProjects\facialemotionrecognizerinrealtime\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

# Get frame dimensions from the webcam
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25  # Fall back to 25 if webcam reports 0

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
video_path = rf'C:\Users\barto\PycharmProjects\facialemotionrecognizerinrealtime\recording_{timestamp_str}.avi'
log_path = r'C:\Users\barto\PycharmProjects\facialemotionrecognizerinrealtime\emotion_log.txt'

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

log_file = open(log_path, 'w')
log_file.write(f"Emotion Detection Log - Started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
log_file.write("-" * 50 + "\n")

CONFIDENCE_THRESHOLD = 0.15
last_label = None
last_confidence = None

try:
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                confidence = prediction.max()

                emotion_changed = label != last_label
                confidence_shifted = last_confidence is None or abs(confidence - last_confidence) > CONFIDENCE_THRESHOLD

                if emotion_changed or confidence_shifted:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    log_file.write(f"[{timestamp}] Emotion: {label} (confidence: {confidence:.2f})\n")
                    log_file.flush()
                    last_label = label
                    last_confidence = confidence

                cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text_size = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        text_x = frame_width - text_size[0] - 10
        text_y = frame_height - 10
        cv2.putText(frame, ts, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        video_writer.write(frame)  # Save the annotated frame to video

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    log_file.close()
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()
