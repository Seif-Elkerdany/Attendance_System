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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['STORAGE_PATH'] = r'C:\Users\shrou\OneDrive\Desktop\AttendanceSystemData'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(app.config['STORAGE_PATH'], 'attendance.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
face_detector = MTCNN()

def _load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))
    return img

def _preprocess_array(img_array):
    rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (160, 160))
    return resized

def detect_and_crop_face(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(rgb_image)
    
    if not results:
        return None
        
    best_face = max(results, key=lambda x: x['confidence'])
    x, y, w, h = best_face['box']
    
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    face = image[y:y+h, x:x+w]
    return face

def save_cropped_faces(images, student_id, course_code):
    base_path = os.path.join(app.config['STORAGE_PATH'], "AttendanceData", course_code, "Faces")
    os.makedirs(base_path, exist_ok=True)
    
    saved_paths = []
    for i, image in enumerate(images, start=1):
        cropped_face = detect_and_crop_face(image)
        if cropped_face is None:
            continue
            
        filename = f"{student_id}_{i}.jpg"
        filepath = os.path.join(base_path, filename)
        cv2.imwrite(filepath, cropped_face)
        saved_paths.append(filepath)
    
    return saved_paths

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_professor = db.Column(db.Boolean, default=False)
    courses_taught = db.relationship('Course', backref='professor', lazy=True)
    attendances = db.relationship('Attendance', backref='student', lazy=True)

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    professor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    students = db.relationship('Student', backref='course', lazy=True)
    sessions = db.relationship('Session', backref='course', lazy=True)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    face_images = db.relationship('FaceImage', backref='student', lazy=True)

class FaceImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(200), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)

class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    attendances = db.relationship('Attendance', backref='session', lazy=True)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    method = db.Column(db.String(50), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('course.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('session.id'), nullable=True)

with app.app_context():
    if not os.path.exists(app.config['STORAGE_PATH']):
        os.makedirs(app.config['STORAGE_PATH'])
    db.create_all()

    if not User.query.filter_by(email="professor@aiu.edu").first():
        professor = User(
            email="professor@aiu.edu",
            name="Dr. Ahmed Mohamed",
            password=generate_password_hash("professor123"),
            is_professor=True
        )
        db.session.add(professor)
        db.session.commit()

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'shrouqwaleed7@gmail.com'
app.config['MAIL_PASSWORD'] = '---'
app.config['MAIL_DEFAULT_SENDER'] = 'shrouqwaleed7@gmail.com'

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
    base_path = os.path.join(app.config['STORAGE_PATH'], "AttendanceData")
    course_path = os.path.join(base_path, course_code)
    students_path = os.path.join(course_path, "Students")
    faces_path = os.path.join(course_path, "Faces")
    
    os.makedirs(students_path, exist_ok=True)
    os.makedirs(faces_path, exist_ok=True)
    
    return {
        'course_path': course_path,
        'students_path': students_path,
        'faces_path': faces_path
    }

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
        attendances = Attendance.query.filter_by(student_id=user.id).order_by(
            Attendance.timestamp.desc()).limit(5).all()
        return render_template('dashboard/student_home.html',
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
            semester = request.form.get('semester', 'Spring 2024-2025')
            
            if not course_code or not course_name:
                flash('Both course code and name are required', 'danger')
            elif Course.query.filter_by(code=course_code).first():
                flash('Course code already exists', 'danger')
            else:
                course = Course(
                    code=course_code,
                    name=course_name,
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
        
        student_folder = os.path.join(
            app.config['STORAGE_PATH'],
            "AttendanceData",
            course_code,
            "Students",
            student_id
        )
        os.makedirs(student_folder, exist_ok=True)
        
        cropped_face = detect_and_crop_face(image)
        if cropped_face is None:
            return jsonify({'error': 'No face detected in the image'}), 400
            
        filename = f"{student_id}_{str(uuid.uuid4())[:8]}.jpg"
        faces_folder = os.path.join(
            app.config['STORAGE_PATH'],
            "AttendanceData",
            course_code,
            "Faces"
        )
        os.makedirs(faces_folder, exist_ok=True)
        face_path = os.path.join(faces_folder, filename)
        cv2.imwrite(face_path, cropped_face)
        
        face_image = FaceImage(path=face_path, student_id=student.id)
        db.session.add(face_image)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'student_id': student_id,
            'student_name': student_name,
            'course_code': course_code,
            'face_image': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def simple_face_recognition(image, course_code):
    faces_folder = os.path.join(app.config['STORAGE_PATH'], "AttendanceData", course_code, "Faces")
    if not os.path.exists(faces_folder):
        return None
    
    input_face = detect_and_crop_face(image)
    if input_face is None:
        return None
    
    input_face = _preprocess_array(input_face)
    
    best_match = None
    best_score = 0
    
    for face_file in os.listdir(faces_folder):
        if not face_file.endswith('.jpg'):
            continue
            
        student_id = face_file.split('_')[0]
        
        stored_face_path = os.path.join(faces_folder, face_file)
        stored_face = _load_image(stored_face_path)
        
        hist_input = cv2.calcHist([input_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_stored = cv2.calcHist([stored_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        score = cv2.compareHist(hist_input, hist_stored, cv2.HISTCMP_CORREL)
        
        if score > best_score:
            best_score = score
            best_match = student_id
    
    return best_match if best_score > 0.7 else None

@app.route('/mark-attendance', methods=['POST'])
@login_required
def mark_attendance_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    course_code = request.form.get('course_code')
    session_name = request.form.get('session_name')

    course = Course.query.filter_by(code=course_code).first()
    if not course:
        return jsonify({'error': 'Invalid course code'}), 400

    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.detect_faces(rgb_image)
    if not results:
        return jsonify({'error': 'No face detected'}), 400

    student_sid = simple_face_recognition(image, course_code)
    if not student_sid:
        return jsonify({'error': 'Student not recognized'}), 400

    student_obj = Student.query.filter_by(student_id=student_sid, course_id=course.id).first()
    if not student_obj:
        return jsonify({'error': 'Student not found in course'}), 400

    user = User.query.filter_by(name=student_obj.name).first()
    if not user:
        return jsonify({'error': 'User record not found'}), 400

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
        student_id=user.id,
        course_id=course.id,
        session_id=session_obj.id if session_obj else None
    )
    db.session.add(attendance)
    db.session.commit()

    recognized_face = FaceImage.query.filter_by(student_id=student_obj.id).first()
    face_path = recognized_face.path if recognized_face else None

    return jsonify({
        'success': True,
        'attendance_record': {
            'student_id': student_obj.student_id,
            'student_name': student_obj.name,
            'course_code': course.code,
            'timestamp': attendance.timestamp.isoformat(),
            'session_name': session_name if session_name else 'General'
        },
        'face_image': face_path
    })

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