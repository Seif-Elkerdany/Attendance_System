import os
import cv2
import numpy as np
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime
from functools import wraps
from mtcnn.mtcnn import MTCNN
import uuid
import csv
import io

# Initialize Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')
app.config['STORAGE_PATH'] = 'AttendanceSystemData'  # Default storage path

# Email configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'shrouqwaleed7@gmail.com'
app.config['MAIL_PASSWORD'] = '---' ## your token goes here
app.config['MAIL_DEFAULT_SENDER'] = 'shrouqwaleed7@gmail.com'
mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Mock database
users_db = {
    "professor@aiu.edu": {
        "name": "Dr. Ahmed Mohamed",
        "password": generate_password_hash("professor123"),
        "courses": ["CS101", "MATH202", "ENG301"]
    }
}

courses_db = {
    "CS101": {
        "name": "Introduction to Computer Science",
        "students": [],
        "professor": "professor@aiu.edu"
    },
    "MATH202": {
        "name": "Advanced Mathematics",
        "students": [],
        "professor": "professor@aiu.edu"
    },
    "ENG301": {
        "name": "Professional English",
        "students": [],
        "professor": "professor@aiu.edu"
    }
}

attendance_history = []

# Initialize face detector
face_detector = MTCNN()

# app.config['STORAGE_PATH'] = os.path.join(os.environ['USERPROFILE'], 'Desktop', 'AttendanceSystemData')   # FOR WINDOWS
app.config['STORAGE_PATH'] = os.path.join(os.path.expanduser('~'), 'Desktop', 'AttendanceSystemData')       # FOR LINUX

class EnhancedLivenessDetector:
    def __init__(self):
        # Enhanced detection parameters
        self.eye_ar_thresh = 0.22
        self.eye_ar_consec_frames = 2
        self.blink_counter = 0
        self.total_blinks = 0
        self.prev_gray = None
        self.motion_frames = 0
        self.required_motion_frames = 5
        self.min_face_confidence = 0.97
        self.min_face_size_ratio = 0.15
        
        # Warning flags
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
        
        left_eye = np.array([
            landmarks['left_eye'],
            [landmarks['left_eye'][0], landmarks['left_eye'][1] - 5],
            [landmarks['left_eye'][0] - 5, landmarks['left_eye'][1]],
            [landmarks['left_eye'][0] + 5, landmarks['left_eye'][1]],
            [landmarks['left_eye'][0], landmarks['left_eye'][1] + 5],
            landmarks['left_eye']
        ])
        
        right_eye = np.array([
            landmarks['right_eye'],
            [landmarks['right_eye'][0], landmarks['right_eye'][1] - 5],
            [landmarks['right_eye'][0] - 5, landmarks['right_eye'][1]],
            [landmarks['right_eye'][0] + 5, landmarks['right_eye'][1]],
            [landmarks['right_eye'][0], landmarks['right_eye'][1] + 5],
            landmarks['right_eye']
        ])
        
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

class AttendanceManager:
    def __init__(self):
        self.data_folder = os.path.join(app.config['STORAGE_PATH'], "attendance_data")
        os.makedirs(self.data_folder, exist_ok=True)
        self.courses = {}
        self.load_courses()

    def load_courses(self):
        courses_file = os.path.join(self.data_folder, "courses.csv")
        if os.path.exists(courses_file):
            with open(courses_file, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    self.courses[row['code']] = {
                        'name': row['name'],
                        'professor': row['professor'],
                        'students': []
                    }

    def save_course(self, code, name, professor):
        self.courses[code] = {
            'name': name,
            'professor': professor,
            'students': []
        }
        self._update_courses_file()

    def _update_courses_file(self):
        courses_file = os.path.join(self.data_folder, "courses.csv")
        with open(courses_file, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['code', 'name', 'professor'])
            for code, data in self.courses.items():
                writer.writerow([code, data['name'], data['professor']])

    def get_course_attendance(self, course_code, session_name=None):
        if session_name:
            attendance_file = os.path.join(self.data_folder, f"{course_code}_{session_name}_attendance.csv")
        else:
            attendance_file = os.path.join(self.data_folder, f"{course_code}_attendance.csv")
            
        records = []
        if os.path.exists(attendance_file):
            with open(attendance_file, mode='r') as file:
                reader = csv.DictReader(file)
                records = list(reader)
        return records

    def mark_attendance(self, course_code, student_id, session_name=None, method='face'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        record = {
            'student_id': student_id,
            'timestamp': timestamp,
            'method': method,
            'course_code': course_code
        }
        
        if session_name:
            record['session_name'] = session_name
            attendance_file = os.path.join(self.data_folder, f"{course_code}_{session_name}_attendance.csv")
        else:
            attendance_file = os.path.join(self.data_folder, f"{course_code}_attendance.csv")
        
        file_exists = os.path.exists(attendance_file)
        
        with open(attendance_file, mode='a') as file:
            fieldnames = ['student_id', 'timestamp', 'method', 'course_code']
            if session_name:
                fieldnames.append('session_name')
                
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)
        
        return record

def save_face_image(image, student_id, course_name):
    course_dir = os.path.join(app.config['STORAGE_PATH'], course_name, "faces")
    os.makedirs(course_dir, exist_ok=True)
    
    filename = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(course_dir, filename)
    cv2.imwrite(filepath, image)
    
    return filepath

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = users_db.get(email)
        
        if not user or not check_password_hash(user['password'], password):
            flash('Invalid email or password', 'danger')
            return redirect(url_for('login'))
        
        session['user_email'] = email
        session['user_name'] = user['name']
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
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('register'))
        
        if email in users_db:
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        users_db[email] = {
            "name": name,
            "password": generate_password_hash(password),
            "courses": []
        }
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('auth/forgot_password.html')
        
    if request.method == 'POST':
        email = request.form.get('email')
        user = users_db.get(email)
        
        if user:
            try:
                token = serializer.dumps(email, salt='password-reset-salt')
                reset_url = url_for('reset_password', token=token, _external=True)
                
                msg = Message(
                    'AIU Attendance - Password Reset',
                    recipients=['shrouqwaleed7@gmail.com'],
                    html=render_template('email/reset_password.html',
                                      user_name=user['name'],
                                      reset_url=reset_url)
                )
                mail.send(msg)
                flash('Password reset link sent to shrouqwaleed7@gmail.com', 'success')
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
        
        if email in users_db:
            users_db[email]['password'] = generate_password_hash(password)
            flash('Your password has been updated successfully', 'success')
            return redirect(url_for('login'))
        else:
            flash('User not found', 'danger')
            return redirect(url_for('forgot_password'))
    
    return render_template('auth/reset_password.html', token=token)

@app.route('/dashboard')
@login_required
def dashboard():
    user_email = session['user_email']
    user = users_db.get(user_email)
    
    professor_courses = {code: course for code, course in courses_db.items() 
                         if course['professor'] == user_email}
    
    return render_template('dashboard/home.html', 
                           user_name=user['name'],
                           courses=professor_courses,
                           history=attendance_history[:5])

@app.route('/dashboard/register-students', methods=['GET', 'POST'])
@login_required
def register_students():
    if request.method == 'POST':
        course_code = request.form.get('course')
        student_name = request.form.get('student_name')
        
        if course_code in courses_db and courses_db[course_code]['professor'] == session['user_email']:
            courses_db[course_code]['students'].append(student_name)
            flash(f'Student {student_name} added to {course_code}', 'success')
        else:
            flash('Invalid course selected', 'danger')
        
        return redirect(url_for('register_students'))
    
    professor_courses = {code: course for code, course in courses_db.items() 
                         if course['professor'] == session['user_email']}
    
    return render_template('dashboard/register.html', 
                           courses=professor_courses)

@app.route('/dashboard/take-attendance', methods=['GET'])
@login_required
def take_attendance_page():
    user_email = session['user_email']
    professor_courses = {code: course for code, course in courses_db.items() 
                         if course['professor'] == user_email}
    
    return render_template('dashboard/capture.html', 
                           courses=professor_courses)

@app.route('/face-registration', methods=['POST'])
@login_required
def face_registration():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    student_id = request.form.get('student_id')
    course_code = request.form.get('course_code')
    student_name = request.form.get('student_name')
    
    if not all([student_id, course_code, student_name]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Read image file
        nparr = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check image quality
        if image is None or image.size == 0:
            return jsonify({'error': 'Invalid image file'}), 400
            
        # Check image dimensions
        if image.shape[0] < 100 or image.shape[1] < 100:
            return jsonify({'error': 'Image too small (min 100x100 pixels)'}), 400
        
        # Liveness check
        liveness_detector = EnhancedLivenessDetector()
        is_live, processed_img = liveness_detector.detect_liveness(image)
        
        if not is_live:
            return jsonify({
                'error': 'Liveness check failed',
                'reason': 'photo' if liveness_detector.photo_warning else 
                          'no_face' if liveness_detector.no_face_warning else 
                          'multi_face' if liveness_detector.multi_face_warning else
                          'small_face' if liveness_detector.small_face_warning else 
                          'no_blink_or_motion'
            }), 400
        
        # Get storage path
        course_dir = os.path.join(app.config['STORAGE_PATH'], course_code, "faces")
        os.makedirs(course_dir, exist_ok=True)
        
        # Save image with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{student_id}_{timestamp}.jpg"
        filepath = os.path.join(course_dir, filename)
        cv2.imwrite(filepath, image)
        
        # Add student to course
        if course_code not in courses_db:
            courses_db[course_code] = {
                "name": course_code,
                "students": [],
                "professor": session['user_email']
            }
        
        # Add student if not already registered
        if student_id not in [s['id'] for s in courses_db[course_code]['students']]:
            courses_db[course_code]['students'].append({
                'id': student_id,
                'name': student_name,
                'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        return jsonify({
            'success': True,
            'image_path': filepath,
            'student_id': student_id,
            'student_name': student_name,
            'course_code': course_code
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mark-attendance', methods=['POST'])
@login_required
def mark_attendance_api():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    course_code = request.form.get('course_code')
    session_name = request.form.get('session_name')
    
    if not course_code:
        return jsonify({'error': 'Missing course_code'}), 400
    
    nparr = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    liveness_detector = EnhancedLivenessDetector()
    is_live, processed_img = liveness_detector.detect_liveness(image)
    
    if not is_live:
        return jsonify({
            'error': 'Liveness check failed',
            'reason': 'photo' if liveness_detector.photo_warning else 
                      'no_face' if liveness_detector.no_face_warning else 
                      'multi_face' if liveness_detector.multi_face_warning else
                      'small_face' if liveness_detector.small_face_warning else 
                      'no_blink_or_motion'
        }), 400
    
    recognized_student_id = "STUDENT_" + str(uuid.uuid4())[:8]
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    attendance_record = {
        'course_code': course_code,
        'timestamp': timestamp,
        'student_id': recognized_student_id,
        'method': 'face'
    }
    
    if session_name:
        attendance_record['session_name'] = session_name
    
    attendance_history.append(attendance_record)
    
    attendance_file = os.path.join(app.config['STORAGE_PATH'], course_code, 'attendance.csv')
    if session_name:
        attendance_file = os.path.join(app.config['STORAGE_PATH'], course_code, f'attendance_{session_name}.csv')
    
    with open(attendance_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if session_name:
            writer.writerow([recognized_student_id, timestamp, session_name])
        else:
            writer.writerow([recognized_student_id, timestamp])
    
    return jsonify({
        'success': True,
        'attendance_record': attendance_record
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

@app.route('/recognize-student', methods=['POST'])
@login_required
def recognize_student():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    course_code = request.form.get('course_code')
    if not course_code:
        return jsonify({'error': 'Missing course_code'}), 400
    
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
    
    faces_dir = os.path.join(app.config['STORAGE_PATH'], course_code, "faces")
    recognized_students = []
    
    if os.path.exists(faces_dir):
        for filename in os.listdir(faces_dir):
            if filename.startswith("STUDENT_"):
                student_id = filename.split("_")[1].split(".")[0]
                recognized_students.append(student_id)
    
    if recognized_students:
        return jsonify({
            'success': True,
            'students': recognized_students[:1],
            'image_path': f"/static/faces/{course_code}/{recognized_students[0]}.jpg"
        })
    else:
        return jsonify({
            'success': False,
            'warning': 'no_match'
        })

@app.route('/check-first-time-setup')
@login_required
def check_first_time_setup():
    # Check if storage path exists and is writable
    storage_path = app.config.get('STORAGE_PATH', 'AttendanceSystemData')
    is_first_time = not os.path.exists(storage_path) or not os.access(storage_path, os.W_OK)
    
    return jsonify({
        'is_first_time': is_first_time,
        'current_path': storage_path
    })

@app.route('/set-storage-path', methods=['POST'])
@login_required
def set_storage_path():
    try:
        storage_path = request.json.get('storage_path', 'AttendanceSystemData')
        
        # Create the directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Test if we can write to the directory
        test_file = os.path.join(storage_path, 'test.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        # Save the path in app config
        app.config['STORAGE_PATH'] = storage_path
        
        return jsonify({
            'success': True,
            'message': f'Storage path set to {storage_path}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/export-attendance/<course_code>')
@login_required
def export_attendance(course_code):
    session_name = request.args.get('session_name')
    
    try:
        if session_name:
            attendance_file = os.path.join(app.config['STORAGE_PATH'], course_code, f'attendance_{session_name}.csv')
        else:
            attendance_file = os.path.join(app.config['STORAGE_PATH'], course_code, 'attendance.csv')
        
        if not os.path.exists(attendance_file):
            return jsonify({'error': 'No attendance records found'}), 404
            
        return send_file(
            attendance_file,
            as_attachment=True,
            download_name=f'attendance_{course_code}_{session_name if session_name else "all"}_{datetime.now().strftime("%Y%m%d")}.csv',
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard/history')
@login_required
def attendance_history_page():
    user_email = session['user_email']
    
    professor_history = [record for record in attendance_history 
                         if courses_db[record['course_code']]['professor'] == user_email]
    
    return render_template('dashboard/history.html', 
                           history=professor_history)

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