import cv2
import os
import time

dataset_dir = 'dataset'
person_id = "0"
save_path = os.path.join(dataset_dir, person_id)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

face1 = None
face2 = None
frame_count = 0
boo = True

while True:
    ret, frame = cap.read()

    if boo == True:
        time.sleep(2)

    if not ret:
        print("Failed to capture frame.")
        time.sleep(2)
        break

    frame_count += 1

    if frame_count % 2 == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for i, (x, y, w, h) in enumerate(faces[:2]):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            if i == 0:
                face1 = face_crop
            elif i == 1:
                face2 = face_crop
        boo = False

    if face1 is not None:
        cv2.imshow("Face 1", face1)
    if face2 is not None:
        cv2.imshow("Face 2", face2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
