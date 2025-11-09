import os
import tkinter as tk
from get_photos import get_owner_face_photos, get_pet_photos
from webcam import run_app


def create_gui():
    root = tk.Tk()
    root.title("Face & Pet Recognition App")
    root.geometry("400x350")
    root.configure(bg="#f2f2f2")

    title = tk.Label(root, text="Face & Pet Recognition", font=("Arial", 16, "bold"), bg="#f2f2f2")
    title.pack(pady=20)

    btn_owner = tk.Button(root, text="Add Owner Face", command=get_owner_face_photos, font=("Arial", 12), bg="#4CAF50", fg="white", width=20, height=2)
    btn_owner.pack(pady=10)
    btn_pet = tk.Button(root, text="Add Pet Images", command=get_pet_photos, font=("Arial", 12), bg="#2196F3", fg="white", width=20, height=2)
    btn_pet.pack(pady=10)
    btn_detect = tk.Button(root, text="Start Detection", command=run_app, font=("Arial", 12), bg="#FF9800", fg="white", width=20, height=2)
    btn_detect.pack(pady=10)
    btn_exit = tk.Button(root, text="Exit", command=root.destroy, font=("Arial", 12), bg="#F44336", fg="white", width=20, height=2)
    btn_exit.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    # Ensure PyTorch doesn't attempt to use GPU DLLs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["QT_OPENGL"] = "software"

    create_gui()
