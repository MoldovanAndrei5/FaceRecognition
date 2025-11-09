import cv2
import face_recognition
import os
import numpy as np
import tensorflow.keras.applications
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_and_preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def get_embedding(model, img_path):
    processed = load_and_preprocess_image(img_path)
    embedding = model.predict(processed, verbose=0)
    return embedding.flatten()


def load_pet_embeddings():
    """Compute average embedding for all pet images."""
    model = tensorflow.keras.applications.MobileNetV3Large(weights="imagenet", include_top=False, pooling="avg")
    embeddings = []
    if not os.path.exists('dataset/pet'):
        return None, model
    for filename in os.listdir('dataset/pet'):
        if filename.endswith('.jpg'):
            img_path = os.path.join('dataset/pet', filename)
            embeddings.append(get_embedding(model, img_path))
    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        return avg_embedding, model
    return None, model


def load_owner_face():
    face_encodings = []
    if not os.path.exists('dataset/owner'):
        os.makedirs('dataset/owner', exist_ok=True)
    for filename in os.listdir('dataset/owner'):
        if filename.endswith('.jpg'):
            image_path = os.path.join('dataset/owner', filename)
            image = face_recognition.load_image_file(image_path)
            enc = face_recognition.face_encodings(image)
            if enc:
                face_encodings.append(enc[0])
    return face_encodings


def run_app():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Initializing", "Loading known faces and pet embeddings...")
    known_face_encodings = load_owner_face()
    pet_embedding, pet_model = load_pet_embeddings()
    if not known_face_encodings:
        messagebox.showwarning("Missing Data", "No owner images found. Please add them first.")
        root.destroy()
        return
    if pet_embedding is None:
        messagebox.showwarning("Missing Pet Data", "No pet images found. Pet recognition will be limited.")
    messagebox.showinfo("Loading Models", "Loading YOLOv8 for real-time detection...")
    yolo = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Could not open webcam.")
        root.destroy()
        return
    messagebox.showinfo("Ready", "Press 'q' to quit. Starting detection...")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        label_text = []

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            label = "Owner" if True in matches else "Unknown"
            color = (0, 255, 0) if label == "Owner" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            label_text.append(label)

        results = yolo(frame, imgsz=640, conf=0.5, verbose=False)
        for res in results:
            for box in res.boxes:
                cls = int(box.cls.cpu().numpy())
                name = yolo.model.names[cls]

                if name in ("dog", "cat"):
                    x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
                    pet_crop = frame[y1:y2, x1:x2]

                    # Pet identification
                    if pet_embedding is not None and pet_crop.size > 0:
                        resized = cv2.resize(pet_crop, (224, 224))
                        arr = np.expand_dims(resized, axis=0)
                        arr = preprocess_input(arr)
                        new_emb = pet_model.predict(arr, verbose=0).flatten()

                        # Cosine similarity
                        sim = np.dot(pet_embedding, new_emb) / (np.linalg.norm(pet_embedding) * np.linalg.norm(new_emb))
                        if sim > 0.7:
                            pet_label = "Owner's Pet"
                            color = (255, 165, 0)
                        else:
                            pet_label = "Other Pet"
                            color = (0, 165, 255)
                    else:
                        pet_label = "Pet"
                        color = (255, 165, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, pet_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    label_text.append(pet_label)

        combined_label = ', '.join(label_text) if label_text else "Nobody"
        cv2.putText(frame, combined_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Face & Pet Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Exit", "Detection stopped.")
    root.destroy()
