import os
import cv2
import numpy as np
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime
from functools import wraps
from mtcnn.mtcnn import MTCNN
import uuid
import csv
import io
import torch 
import torchvision.transforms.functional as TF
from modeling.model.NN_B3 import SiameseClassifier

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['STORAGE_PATH'] = '/home/seif_elkerdany/Desktop/AttendanceSystem'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.config['STORAGE_PATH'], 'attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

def _load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor

# For captured face arrays (not from disk)

def _preprocess_array(img_array):
    # img_array: HxWx3 BGR
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    return tensor

# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = SiameseClassifier(embedding_dim=256).to(device)
ckpt_path = "/home/seif_elkerdany/projects/modeling/model/checkpoints/B3.1/checkpoint_epoch1.pt"
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt.get("model_state_dict", ckpt)

cleaned = {
    (k.replace("module.", "") if k.startswith("module.") else k): v
    for k, v in state_dict.items()
}
MODEL.load_state_dict(cleaned)
MODEL.eval()

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
face_detector = MTCNN()

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_professor = db.Column(db.Boolean, default=False)
    courses_taught = db.relationship('Course', backref='professor', lazy=True)
    attendances = db.relationship('Attendance', backref='student', lazy=True)

    def __repr__(self):
        return f"User('{self.name}', '{self.email}')"

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    professor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    students = db.relationship('Student', backref='course', lazy=True)
    sessions = db.relationship('Session', backref='course', lazy=True)
    semester = db.Column(db.String(100)) 

    def __repr__(self):
        return f"Course('{self.code}', '{self.name}')"

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    face_images = db.relationship('FaceImage', backref='student', lazy=True)

    def __repr__(self):
        return f"Student('{self.student_id}', '{self.name}')"

class FaceImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(200), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

    def __repr__(self):
        return f"FaceImage('{self.path}')"

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    attendances = db.relationship('Attendance', backref='session', lazy=True)

    def __repr__(self):
        return f"Session('{self.name}')"

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    method = db.Column(db.String(50), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=True)

    def __repr__(self):
        return f"Attendance('{self.student_id}', '{self.timestamp}')"

    @property
    def total(self):
        return Session.query.filter_by(course_id=self.course_id).count()

# Create base directory and database
with app.app_context():
    if not os.path.exists(app.config['STORAGE_PATH']):
        os.makedirs(app.config['STORAGE_PATH'])
    db.create_all()

    # Create default professor if not exists
    if not User.query.filter_by(email="professor@aiu.edu").first():
        professor = User(
            email="professor@aiu.edu",
            name="Dr. Ahmed Mohamed",
            password=generate_password_hash("professor123"),
            is_professor=True
        )
        db.session.add(professor)
        db.session.commit()

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'shrouqwaleed7@gmail.com'
app.config['MAIL_PASSWORD'] = '---'
app.config['MAIL_DEFAULT_SENDER'] = 'shrouqwaleed7@gmail.com'

class EnhancedLivenessDetector:
    def __init__(self):
        self.eye_ar_thresh = 0.22
        self.eye_ar_consec_frames = 2
        self.blink_counter = 0
        self.total_blinks = 0
        self.prev_gray = None
        self.motion_frames = 0
        self.required_motion_frames = 5
        self.min_face_confidence = 0.97
        self.min_face_size_ratio = 0.15
        self.photo_warning = False
        self.no_face_warning = False
        self.multi_face_warning = False
        self.small_face_warning = False

    def eye_aspect_ratio(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def check_face_size(self, face_box, frame_shape):
        x, y, w, h = face_box
        frame_height, frame_width = frame_shape[:2]
        face_area = w * h
        frame_area = frame_width * frame_height
        return (face_area / frame_area) >= self.min_face_size_ratio

    def detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
            
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        self.prev_gray = gray
        
        if np.sum(thresh) > 10000:
            self.motion_frames += 1
            if self.motion_frames >= self.required_motion_frames:
                return True
        return False

    def detect_liveness(self, frame):
        self.reset_warnings()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.detect_faces(rgb_frame)
        
        if not results:
            self.no_face_warning = True
            return False, frame
            
        if len(results) > 1:
            self.multi_face_warning = True
            return False, frame
            
        face = results[0]
        
        if face['confidence'] < self.min_face_confidence:
            self.photo_warning = True
            return False, frame
            
        if not self.check_face_size(face['box'], frame.shape):
            self.small_face_warning = True
            return False, frame
            
        landmarks = face['keypoints']
        left_eye = np.array([landmarks['left_eye'], [landmarks['left_eye'][0], landmarks['left_eye'][1] - 5],
                           [landmarks['left_eye'][0] - 5, landmarks['left_eye'][1]],
                           [landmarks['left_eye'][0] + 5, landmarks['left_eye'][1]],
                           [landmarks['left_eye'][0], landmarks['left_eye'][1] + 5],
                           landmarks['left_eye']])
                        
        right_eye = np.array([landmarks['right_eye'], [landmarks['right_eye'][0], landmarks['right_eye'][1] - 5],
                            [landmarks['right_eye'][0] - 5, landmarks['right_eye'][1]],
                            [landmarks['right_eye'][0] + 5, landmarks['right_eye'][1]],
                            [landmarks['right_eye'][0], landmarks['right_eye'][1] + 5],
                            landmarks['right_eye']])
                        
        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        if ear < self.eye_ar_thresh:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.eye_ar_consec_frames:
                self.total_blinks += 1
            self.blink_counter = 0
        
        motion_detected = self.detect_motion(frame)
        
        if self.total_blinks >= 1 and motion_detected:
            return True, frame
            
        return False, frame

    def reset_warnings(self):
        self.photo_warning = False
        self.no_face_warning = False
        self.multi_face_warning = False
        self.small_face_warning = False



def save_face_images(images, student_id, course_code):
    saved_paths = []
    try:
        student_folder = os.path.join(
            app.config['STORAGE_PATH'],
            "AttendanceData",
            course_code,
            "Students",
            student_id
        )

        faces_folder = os.path.join(
            app.config['STORAGE_PATH'],
            "AttendanceData",
            course_code,
            "Faces"
        )

        try:
            os.makedirs(student_folder, exist_ok=True)
            os.makedirs(faces_folder, exist_ok=True)
        except Exception as e:
            print(f"Error creating folders: {e}")
            return []

        detector = MTCNN()

        for i, image in enumerate(images, start=1):
            try:
                
                original_path = os.path.join(student_folder, f"{student_id}_{i}.jpg")
                cv2.imwrite(original_path, image)
                saved_paths.append(original_path)

               
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_image)

                for j, face in enumerate(faces, 1):
                    x, y, w, h = face['box']
                    x, y = max(0, x), max(0, y)
                    cropped_face = image[y:y+h, x:x+w]

                    if cropped_face.size > 0:
                        face_path = os.path.join(
                            faces_folder,
                            f"{student_id}_{i}_face_{j}.jpg"
                        )
                        cv2.imwrite(face_path, cropped_face)
                        saved_paths.append(face_path)

            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue

        return saved_paths
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def professor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        user = User.query.get(session['user_id'])
        if not user or not user.is_professor:
            flash('Professor access required', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function
def ensure_course_folders(course_code):
    """Ensure all required folders for a course exist"""
    try:
        base_path = os.path.join(app.config['STORAGE_PATH'], "AttendanceData")
        course_path = os.path.join(base_path, course_code)
        students_path = os.path.join(course_path, "Students")
        faces_path = os.path.join(course_path, "Faces")
        
        os.makedirs(students_path, exist_ok=True)
        os.makedirs(faces_path, exist_ok=True)
        
        print(f"Created folders: {students_path} and {faces_path}")
        
        return {
            'course_path': course_path,
            'students_path': students_path,
            'faces_path': faces_path
        }
    except Exception as e:
        print(f"Error creating folders: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password, password):
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
        
        session['user_id'] = user.id
        session['user_email'] = user.email
        session['user_name'] = user.name
        session['is_professor'] = user.is_professor
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        is_professor = request.form.get('is_professor') == 'on'
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        user = User(name=name, email=email, password=hashed_password, is_professor=is_professor)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('auth/forgot_password.html')
        
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            try:
                token = serializer.dumps(email, salt='password-reset-salt')
                reset_url = url_for('reset_password', token=token, _external=True)
                
                msg = Message(
                    'AIU Attendance - Password Reset',
                    recipients=[email],
                    html=render_template('email/reset_password.html',
                                      user_name=user.name,
                                      reset_url=reset_url)
                )
                mail.send(msg)
                flash('Password reset link has been sent to your email', 'success')
            except Exception as e:
                flash('Failed to send email. Please try again later.', 'danger')
            
            return redirect(url_for('login'))
        
        flash('If this email exists, a reset link has been sent', 'info')
        return redirect(url_for('forgot_password'))

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except:
        flash('The reset link is invalid or has expired', 'danger')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(request.url)
        
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(password)
            db.session.commit()
            flash('Your password has been updated successfully', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found', 'danger')
            return redirect(url_for('forgot_password'))
    
    return render_template('auth/reset_password.html', token=token)

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    
    if user.is_professor:
        courses = Course.query.filter_by(professor_id=user.id).all()
        if not courses:
            flash('You have no courses yet. Please add courses from the "My Courses" page.', 'info')
        
        recent_attendances = Attendance.query.join(Course).filter(
            Course.professor_id == user.id
        ).order_by(Attendance.timestamp.desc()).limit(5).all()
        
        return render_template('dashboard/home.html', 
                             user_name=user.name,
                             courses=courses,
                             history=recent_attendances)
    else:
        # Student dashboard
        attendances = Attendance.query.filter_by(student_id=user.id).order_by(
            Attendance.timestamp.desc()).limit(5).all()
        return render_template('dashboard/home.html',
                             user_name=user.name,
                             history=attendances)

@app.route('/dashboard/my-courses', methods=['GET', 'POST'])
@login_required
@professor_required
def my_courses():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add_course':
            course_code = request.form.get('course_code')
            course_name = request.form.get('course_name')
            semester = request.form.get('semester')

            semester = request.form.get('semester', 'Spring 2024-2025')
            
            if not course_code or not course_name:
                flash('Both course code and name are required', 'danger')
            elif Course.query.filter_by(code=course_code).first():
                flash('Course code already exists', 'danger')
            else:
                course = Course(
                    code=course_code,
                    name=course_name,
                    semester=semester,
                    professor_id=session['user_id']
                )
                db.session.add(course)
                db.session.commit()
                ensure_course_folders(course_code)
                flash(f'Course {course_code} added successfully', 'success')
                
        elif action == 'delete_course':
            course_code = request.form.get('course_code')
            course = Course.query.filter_by(
                code=course_code,
                professor_id=session['user_id']
            ).first()
            
            if course:
                db.session.delete(course)
                db.session.commit()
                flash(f'Course {course_code} deleted successfully', 'success')
            else:
                flash('Invalid course selected', 'danger')
        
        return redirect(url_for('my_courses'))
    
    courses = Course.query.filter_by(professor_id=session['user_id']).all()
    return render_template('dashboard/my_courses.html',
                         user_name=session['user_name'],
                         courses=courses,
                         university_email="algalia@alamein.com")

@app.route('/dashboard/register-students', methods=['GET', 'POST'])
@login_required
@professor_required
def register_students():
    if request.method == 'POST':
        course_id = request.form.get('course')
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')
        
        course = Course.query.filter_by(
            id=course_id,
            professor_id=session['user_id']
        ).first()
        
        if not course:
            flash('Invalid course selected', 'danger')
            return redirect(url_for('register_students'))
            
        # Add student to course
        if not Student.query.filter_by(student_id=student_id, course_id=course.id).first():
            student = Student(
                student_id=student_id,
                name=student_name,
                course_id=course.id
            )
            db.session.add(student)
            db.session.commit()
            flash(f'Student {student_name} added to {course.code}', 'success')
        else:
            flash('Student ID already exists in this course', 'warning')
        
        return redirect(url_for('register_students'))
    
    courses = Course.query.filter_by(professor_id=session['user_id']).all()
    return render_template('dashboard/register.html', 
                         courses=courses)

@app.route('/dashboard/take-attendance', methods=['GET'])
@login_required
@professor_required
def take_attendance_page():
    courses = Course.query.filter_by(professor_id=session['user_id']).all()
    return render_template('dashboard/capture.html', 
                         courses=courses)

@app.route('/face-registration', methods=['POST'])
@login_required
@professor_required
def face_registration():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    student_id = request.form.get('student_id')
    course_code = request.form.get('course_code')
    student_name = request.form.get('student_name')
    image_number = request.form.get('image_number', 1)
    
    if not all([student_id, course_code, student_name]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    course = Course.query.filter_by(
        code=course_code,
        professor_id=session['user_id']
    ).first()
    
    if not course:
        return jsonify({
            'error': 'Course does not exist or you are not the professor',
            'course_not_found': True
        }), 400
    
    try:
        nparr = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            return jsonify({'error': 'Invalid image file'}), 400
            
        if image.shape[0] < 100 or image.shape[1] < 100:
            return jsonify({'error': 'Image too small (min 100x100 pixels)'}), 400
        
        # Save the image
        student = Student.query.filter_by(
            student_id=student_id,
            course_id=course.id
        ).first()
        
        if not student:
            student = Student(
                student_id=student_id,
                name=student_name,
                course_id=course.id
            )
            db.session.add(student)
            db.session.commit()
        
        # Create necessary folders
        student_folder = os.path.join(
            app.config['STORAGE_PATH'],
            "AttendanceData",
            course_code,
            "Students",
            student_id
        )
        faces_folder = os.path.join(
            app.config['STORAGE_PATH'],
            "AttendanceData",
            course_code,
            "Faces"
        )
        os.makedirs(student_folder, exist_ok=True)
        os.makedirs(faces_folder, exist_ok=True)
        
        # Save original image
        filename = f"{student_id}_{image_number}.jpg"
        original_path = os.path.join(student_folder, filename)
        cv2.imwrite(original_path, image)
        
        # Detect and save cropped faces
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(rgb_image)
        face_paths = []
        
        for i, face in enumerate(faces, 1):
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped_face = image[y:y+h, x:x+w]
            
            if cropped_face.size > 0:
                face_filename = f"{student_id}_{image_number}_face_{i}.jpg"
                face_path = os.path.join(faces_folder, face_filename)
                cv2.imwrite(face_path, cropped_face)
                face_paths.append(face_path)
        
        # Save paths to database
        face_image = FaceImage(path=original_path, student_id=student.id)
        db.session.add(face_image)
        
        for path in face_paths:
            face_image = FaceImage(path=path, student_id=student.id)
            db.session.add(face_image)
            
        db.session.commit()
        
        return jsonify({
            'success': True,
            'original_image_path': original_path,
            'face_paths': face_paths,
            'student_id': student_id,
            'student_name': student_name,
            'course_code': course_code,
            'image_number': image_number,
            'faces_detected': len(faces)
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/check-course-folders/<course_code>')
@login_required
@professor_required
def check_course_folders(course_code):
    messages = []
    base_path = os.path.join(app.config['STORAGE_PATH'], "AttendanceData")
    course_path = os.path.join(base_path, course_code)
    students_path = os.path.join(course_path, "Students")
    faces_path = os.path.join(course_path, "Faces")
    
    paths = {
        'base_path': base_path,
        'course_path': course_path,
        'students_path': students_path,
        'faces_path': faces_path
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            os.makedirs(path)
            messages.append(f"Created folder: {path}")
        else:
            messages.append(f"Folder exists: {path}")
    
    return jsonify({
        'course': course_code,
        'messages': messages,
        'storage_path': app.config['STORAGE_PATH']
    })
# ------------------ MODEL INFERENCE METHOD ------------------
# This method uses the trained Siamese Neural Network (SNN) model
# to compare the uploaded face image with stored student images
# to identify the student by face matching.

def compare_faces(img, course_code):
    """
    Compare a face image with student images for a specific course
    using a trained Siamese Neural Network model.
    
    Returns:
        student_id (str) if a match is found, otherwise None.
    """
    face_tensor = _preprocess_array(img).unsqueeze(0).to(device)
    student_folder = os.path.join(app.config['STORAGE_PATH'], "AttendanceData", course_code, "Faces")
    
    for fname in os.listdir(student_folder):
        path = os.path.join(student_folder, fname)
        try:
            stud_tensor = _load_image(path).unsqueeze(0).to(device)
        except Exception:
            continue
        
       
        probs, preds = MODEL.predict(face_tensor, stud_tensor)
        if preds.item():
            sid = os.path.splitext(fname)[0].split('_')[0]
            print(f"[MATCH FOUND] Student ID: {sid}")
            return sid
    return None

@app.route('/mark-attendance', methods=['POST'])

def mark_attendance_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    image_file = request.files['image']
    course_code = request.form.get('course_code')
    session_name = request.form.get('session_name')

    course = Course.query.filter_by(code=course_code).first()
    if not course:
        return jsonify({'error': 'Invalid course code'}), 400

    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  
    student_sid = compare_faces(image, course_code)
    
    print(f"[QUERY] Database Query: {student_sid} {course.id}")

    if not student_sid:
        return jsonify({'error': 'Student not recognized'}), 400

    student_obj = Student.query.filter_by(student_id=student_sid, course_id=course.id).first()
    if not student_obj:
        return jsonify({'error': 'Student not found in course'}), 400

    session_obj = None
    if session_name:
        session_obj = Session.query.filter_by(
            name=session_name,
            course_id=course.id
        ).first()

        if not session_obj:
            session_obj = Session(name=session_name, course_id=course.id)
            db.session.add(session_obj)
            db.session.commit()

   
    attendance = Attendance(
        method='face',
        student_id=student_sid,
        course_id=course.id,
        session_id=session_obj.id if session_obj else None
    )
    db.session.add(attendance)
    db.session.commit()

    return jsonify({
        'success': True,
        'attendance_record': {
            'student_id': student_sid,
            'student_name': student_obj.name,
            'course_code': course.code,
            'timestamp': attendance.timestamp.isoformat(),
            'session_name': session_name
        }
    })


@app.route('/check-liveness', methods=['POST'])
def check_liveness():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    nparr = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    liveness_detector = EnhancedLivenessDetector()
    is_live, _ = liveness_detector.detect_liveness(frame)
    
    response = {
        'is_live': is_live,
        'warnings': {
            'no_face': liveness_detector.no_face_warning,
            'multi_face': liveness_detector.multi_face_warning,
            'photo': liveness_detector.photo_warning,
            'small_face': liveness_detector.small_face_warning
        }
    }
    
    return jsonify(response)

"""@app.route('/recognize-student', methods=['POST'])
@login_required
@professor_required
def recognize_student():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    course_code = request.form.get('course_code')
    course = Course.query.filter_by(
        code=course_code,
        professor_id=session['user_id']
    ).first()
    
    if not course:
        return jsonify({'error': 'Invalid course code'}), 400
    
    image_file = request.files['image']
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    liveness_detector = EnhancedLivenessDetector()
    is_live, processed_img = liveness_detector.detect_liveness(image)
    
    if not is_live:
        return jsonify({
            'success': False,
            'warning': 'photo' if liveness_detector.photo_warning else 
                      'no_face' if liveness_detector.no_face_warning else 
                      'multi_face' if liveness_detector.multi_face_warning else
                      'small_face' if liveness_detector.small_face_warning else 
                      'no_blink_or_motion'
        })
    
    # In a real app, you would implement face recognition here
    # For demo, return the first student in the course
    student = course.students.first()
    
    if student:
        images = [img.path for img in student.face_images]
        return jsonify({
            'success': True,
            'student': {
                'id': student.student_id,
                'name': student.name,
                'images': images
            }
        })
    
    return jsonify({
        'success': False,
        'warning': 'no_match'
    })"""

@app.route('/dashboard/history')
@login_required
def attendance_history_page():
    user = User.query.get(session['user_id'])
    
    if user.is_professor:
        attendances = Attendance.query.join(Course).filter(
            Course.professor_id == user.id
        ).order_by(Attendance.timestamp.desc()).all()
    else:
        attendances = Attendance.query.filter_by(
            student_id=user.id
        ).order_by(Attendance.timestamp.desc()).all()
    
    return render_template('dashboard/history.html', 
                         history=attendances)

@app.route('/export-attendance/<course_code>')
@login_required
@professor_required
def export_attendance(course_code):
    session_name = request.args.get('session_name')
    
    try:
        course = Course.query.filter_by(
            code=course_code,
            professor_id=session['user_id']
        ).first()
        
        if not course:
            return jsonify({'error': 'Course not found or unauthorized'}), 404
            
        query = Attendance.query.filter_by(course_id=course.id)
        if session_name:
            session_obj = Session.query.filter_by(
                name=session_name,
                course_id=course.id
            ).first()
            if session_obj:
                query = query.filter_by(session_id=session_obj.id)
        
        records = query.join(User).add_columns(
            User.name.label('student_name'),
            Attendance.timestamp,
            Attendance.method
        ).order_by(Attendance.timestamp.desc()).all()
        
        if not records:
            return jsonify({'error': 'No attendance records found'}), 404
            
        # Create CSV output
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Student Name', 'Timestamp', 'Method', 'Session'])
        
        for record in records:
            writer.writerow([
                record.student_name,
                record.timestamp,
                record.method,
                session_name if session_name else 'General'
            ])
        
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            as_attachment=True,
            download_name=f'attendance_{course_code}_{session_name if session_name else "all"}_{datetime.now().strftime("%Y%m%d")}.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test-email')
def test_email():
    try:
        msg = Message(
            'Test Email from AIU Attendance',
            recipients=['shrouqwaleed7@gmail.com'],
            body='This is a test email from your Flask app'
        )
        mail.send(msg)
        return "Test email sent successfully to shrouqwaleed7@gmail.com!"
    except Exception as e:
        return f"Failed to send test email: {str(e)}"

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)