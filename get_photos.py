import cv2
import os
import face_recognition
import time
import tkinter as tk
from tkinter import messagebox


def get_owner_face_photos():
    root = tk.Tk()
    root.withdraw()
    owner_dir = 'dataset/owner'
    os.makedirs(owner_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Could not open webcam. Please check your camera connection.")
        return
    time.sleep(2)
    count = 0
    messagebox.showinfo("Instructions", "Press 'c' to capture an image, 'q' to quit.\nTry capturing from different angles.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Capture Owner Images', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            face_locations = face_recognition.face_locations(frame)
            if len(face_locations) == 1:
                filename = f'dataset/owner/face_{count}.jpg'
                cv2.imwrite(filename, frame)
                count += 1
                messagebox.showinfo("Saved", f"Saved {filename}")
            elif len(face_locations) == 0:
                messagebox.showwarning("No Face Detected", "No face detected. Please ensure your face is visible.")
            else:
                messagebox.showwarning("Multiple Faces", "Multiple faces detected. Please ensure only one face is visible.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"Collected {count} owner images.")
    root.destroy()


def get_pet_photos():
    root = tk.Tk()
    root.withdraw()
    owner_dir = 'dataset/pet'
    os.makedirs(owner_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Could not open webcam. Please check your camera connection.")
        return
    time.sleep(2)
    count = 0
    messagebox.showinfo("Instructions", "Press 'c' to capture an image, 'q' to quit.\nTry capturing from different angles.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Capture Pet Images', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            filename = f'dataset/pet/pet_{count}.jpg'
            cv2.imwrite(filename, frame)
            count += 1
            messagebox.showinfo("Saved", f"Saved {filename}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Done", f"Collected {count} pet images.")
    root.destroy()
