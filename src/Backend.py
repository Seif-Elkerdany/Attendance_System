import os
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN
from datetime import datetime
import csv
from modeling.model.SNN_B1 import CNNBackbone, SiameseNetwork
from modeling.model.test import predict
import torch
import uuid
import torchvision.transforms.functional as TF

# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 1024
backbone = CNNBackbone(embedding_dim=embedding_dim)
MODEL = SiameseNetwork(base_net=backbone)
MODEL = MODEL.to(device)

checkpoint_path = "/home/seif_elkerdany/projects/modeling/model/checkpoints/B1/best_model_epoch_19.pt"  
MODEL.load_state_dict(torch.load(checkpoint_path, map_location=device))
MODEL.eval()

# Path to the Desktop directory
if os.name == 'nt':  # 'nt' stands for Windows
    home_dir = os.environ['USERPROFILE']
else:                # This for linux (Seif Elkerdany)
    home_dir = os.environ['HOME']

desktop_path = os.path.join(home_dir, 'Desktop')

# Set up the main folder for storing images and course data
MAIN_FOLDER = os.path.join(desktop_path, "AttendanceSystemData")
os.makedirs(MAIN_FOLDER, exist_ok=True)
print(f"Main folder path: {MAIN_FOLDER}")

# Initialize the MTCNN face detector
face_detector = MTCNN()

# List to store course names
courses_list = []

# Adding existed courses
for folder in os.listdir(MAIN_FOLDER):
    courses_list.append(folder)

# Function to update the course list in the Listbox and Combobox
def update_courses_list():
    course_listbox.delete(0, tk.END)
    course_combobox_register['values'] = courses_list
    course_combobox_attendance['values'] = courses_list
    if courses_list:
        course_combobox_register.current(0)
        course_combobox_attendance.current(0)
    for course in courses_list:
        course_listbox.insert(tk.END, course)

# Function to detect a face in an image
def detect_face(img):
    result = face_detector.detect_faces(img)
    if result:
        x, y, w, h = result[0]['box']
        face = img[y:y+h, x:x+w]
        return cv2.resize(face, (112, 112))
    return None

# Function to save the student's face image for a specific course
def save_face(img, student_id, course_name):
    course_folder = os.path.join(MAIN_FOLDER, course_name)
    os.makedirs(course_folder, exist_ok=True)
    path = os.path.join(course_folder, student_id + f"_{uuid.uuid1()}.jpg")
    cv2.imwrite(path, img)
    return path


# Placeholder function for comparing faces using a trained model
def compare_faces(img, course_name):
    student_imgs_path = os.path.join(MAIN_FOLDER, course_name)

    # Convert the detected face to grayscale if it's not already (to match the dataset preprocessing)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_tensor = TF.to_tensor(img).repeat(3, 1, 1).unsqueeze(0).to(device)

    for student_file in os.listdir(student_imgs_path):
        student_img_path = os.path.join(student_imgs_path, student_file)

        # Load the student image in grayscale mode
        current_student = cv2.imread(student_img_path, cv2.IMREAD_GRAYSCALE)
        if current_student is None:
            continue

        current_student = cv2.resize(current_student, (112, 112), interpolation=cv2.INTER_AREA)
        current_student_tensor = TF.to_tensor(current_student).repeat(3, 1, 1).unsqueeze(0).to(device)

        prediction = predict(MODEL, img_tensor, current_student_tensor)

        if prediction == 1:
            name_without_ext = os.path.splitext(student_file)[0]
            student_id = name_without_ext.split("_")[0]
            return student_id

    return None

# GUI
root = tk.Tk()
root.title("Attendance System")
notebook = ttk.Notebook(root)

# Course registration tab
course_tab = ttk.Frame(notebook)
notebook.add(course_tab, text="Courses")

tk.Label(course_tab, text="Course Name:").pack()
course_entry_add = tk.Entry(course_tab)
course_entry_add.pack()

# Listbox to display courses
course_listbox = tk.Listbox(course_tab, height=6)
course_listbox.pack(pady=10)

# Function to add a new course
def add_course():
    course_name = course_entry_add.get().strip()
    if not course_name:
        messagebox.showerror("Error", "Please enter course name")
        return
    if course_name in courses_list:
        messagebox.showerror("Error", "Course already exists")
        return
    course_folder = os.path.join(MAIN_FOLDER, course_name)
    os.makedirs(course_folder, exist_ok=True)
    courses_list.append(course_name)
    update_courses_list()
    messagebox.showinfo("Success", f"Course '{course_name}' added successfully!")
    course_entry_add.delete(0, tk.END)

tk.Button(course_tab, text="Add Course", command=add_course).pack(pady=10)

# Student registration tab
register_tab = ttk.Frame(notebook)
notebook.add(register_tab, text="Register")

tk.Label(register_tab, text="Student ID:").pack()
id_entry_register = tk.Entry(register_tab)
id_entry_register.pack()

tk.Label(register_tab, text="Select Course:").pack()
course_combobox_register = ttk.Combobox(register_tab, values=courses_list)
course_combobox_register.pack()
if courses_list:
    course_combobox_register.current(0)

# Function to choose registration method (upload or capture face)
def choose_registration_method(course_name):
    method_window = tk.Toplevel(root)
    method_window.title("Choose Registration Method")

    def upload_image():
        student_id = id_entry_register.get().strip()
        if not student_id:
            messagebox.showerror("Error", "Please enter student ID")
            return
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png")])
        if filepath:
            img = cv2.imread(filepath)
            face = detect_face(img)
            if face is not None:
                save_face(face, student_id, course_name)
                messagebox.showinfo("Success", "Face registered successfully!")
                id_entry_register.delete(0, tk.END)  # Clear ID after registration
            else:
                messagebox.showerror("Error", "No face detected")

    def capture_faces():
        student_id = id_entry_register.get().strip()
        if not student_id:
            messagebox.showerror("Error", "Please enter student ID")
            return
        cap = cv2.VideoCapture(0)
        images = []
        count = 0
        while count < 5:
            ret, frame = cap.read()

            cv2.imshow("Capturing Face", frame)
            cv2.waitKey(1)

            key = cv2.waitKey(500) & 0xFF
            if ret:
                if key == ord('c'):
                    face = detect_face(frame)
                    if face is not None:
                        images.append(face)
                        count += 1

        cap.release()
        cv2.destroyAllWindows()

        for img in images:
            save_face(img, student_id, course_name)
        messagebox.showinfo("Success", "Faces registered successfully!")
        id_entry_register.delete(0, tk.END)  # Clear ID after registration

    tk.Button(method_window, text="Upload Image", command=upload_image).pack(pady=10)
    tk.Button(method_window, text="Capture Faces", command=capture_faces).pack(pady=10)

def register_student():
    course_name = course_combobox_register.get()
    student_id = id_entry_register.get().strip()
    if not student_id or not course_name:
        messagebox.showerror("Error", "Please enter both student ID and select a course")
        return
    choose_registration_method(course_name)

tk.Button(register_tab, text="Register Student", command=register_student).pack(pady=10)

# Attendance tab
attendance_tab = ttk.Frame(notebook)
notebook.add(attendance_tab, text="Attendance")

tk.Label(attendance_tab, text="Select Course for Attendance:").pack(pady=5)

# Combobox to select course for attendance
course_combobox_attendance = ttk.Combobox(attendance_tab, values=courses_list)
course_combobox_attendance.pack(pady=5)
if courses_list:
    course_combobox_attendance.current(0)

video_label = tk.Label(attendance_tab)
video_label.pack()
cap = None 
# Start the video stream
def start_video_stream():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

# Update frame function to show the live video stream
def update_frame():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        video_label.after(10, update_frame)

# Stop the video stream
def stop_video_stream():
    global cap
    if cap is not None:
        cap.release()
        cap = None
    video_label.configure(image='')

# Function to capture and check attendance
def capture_and_check_attendance():
    course_name = course_combobox_attendance.get()
    if not course_name:
        messagebox.showerror("Error", "Please select a course")
        return

    global cap
    if cap is None:
        messagebox.showerror("Error", "Video stream not started")
        return

    ret, frame = cap.read()
    if ret:
        face = detect_face(frame)
        if face is not None:
            recognized_id = compare_faces(face, course_name) # **THIS IS WHERE YOU NEED TO COMPARE FACES USING THE MODEL**
            if recognized_id:  # If the face is recognized, proceed to mark attendance
                # Record attendance in CSV file
                csv_file = os.path.join(desktop_path, f'{course_name}_attendance.csv')
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([recognized_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                messagebox.showinfo("Attendance", f"Attendance recorded for student: {recognized_id}")
            else:
                messagebox.showerror("Error", "Face not recognized in the course.")
        else:
            messagebox.showerror("Error", "No face detected.")

tk.Button(attendance_tab, text="Start Video Stream", command=start_video_stream).pack(pady=5)
tk.Button(attendance_tab, text="Stop Video Stream", command=stop_video_stream).pack(pady=5)
tk.Button(attendance_tab, text="Capture & Check Attendance", command=capture_and_check_attendance).pack(pady=5)

# Running the app
notebook.pack(expand=True, fill='both', padx=10, pady=10)
root.mainloop()
