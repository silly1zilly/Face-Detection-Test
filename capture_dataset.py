import cv2
import os
import time

dataset_dir = 'dataset'
person_id = "0"
max_images = 100

save_path = os.path.join("dataset", "0")
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")  # type: ignore

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to connect to camera.")
        time.sleep(2)
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        cv2.imshow("Face", face)
    cv2.imshow("Camera", frame)
