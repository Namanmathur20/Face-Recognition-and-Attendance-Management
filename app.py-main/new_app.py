import os
import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from collections import defaultdict
from threading import Lock
import time

# Flask and Database Setup
app = Flask(__name__)
app.secret_key = "secret"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return f'<User {self.name}>'

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

# Global variables for face recognition
known_face_encodings = []
known_face_ids = []
known_face_names = []
face_recognition_lock = Lock()
last_face_update = 0

# Initialize Database
with app.app_context():
    db.create_all()

# Camera setup with optimized parameters
def init_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    return cap

# Load and cache known faces
def load_known_faces():
    global known_face_encodings, known_face_ids, known_face_names, last_face_update
    
    with face_recognition_lock:
        # Only reload if more than 5 minutes have passed or first load
        if time.time() - last_face_update > 300 or not known_face_encodings:
            users = User.query.all()
            new_encodings = []
            new_ids = []
            new_names = []
            
            for user in users:
                try:
                    image = face_recognition.load_image_file(user.image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        new_encodings.append(encodings[0])
                        new_ids.append(user.id)
                        new_names.append(user.name)
                except Exception as e:
                    print(f"Error loading face for {user.name}: {e}")
            
            known_face_encodings = new_encodings
            known_face_ids = new_ids
            known_face_names = new_names
            last_face_update = time.time()
            print(f"Loaded {len(known_face_encodings)} face encodings")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        image = request.files['image']
        if name and image:
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(image_path)

            new_user = User(name=name, image_path=image_path)
            db.session.add(new_user)
            db.session.commit()
            
            # Update face encodings cache
            load_known_faces()
            
            flash(f"User '{name}' registered successfully!", "success")
            return redirect(url_for('index'))
        else:
            flash("Please provide a name and an image.", "danger")
    return render_template('register.html')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        load_known_faces()
        
        if not known_face_encodings:
            flash("No registered faces found", "error")
            return redirect(url_for('index'))
        
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            flash("Camera error", "error")
            return redirect(url_for('attendance'))
        
        # Process frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        marked_today = []
        new_marked = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                user_id = known_face_ids[best_match_index]
                name = known_face_names[best_match_index]
                
                # Check if already marked today
                today = datetime.now().date()
                existing = Attendance.query.filter(
                    Attendance.user_id == user_id,
                    db.func.date(Attendance.timestamp) == today
                ).first()
                
                if existing:
                    marked_today.append(name)
                else:
                    # Mark new attendance
                    new_attendance = Attendance(user_id=user_id, name=name)
                    db.session.add(new_attendance)
                    new_marked.append(name)
        
        db.session.commit()
        
        # Prepare response message
        if new_marked:
            message = f"Attendance marked for: {', '.join(new_marked)}"
            if marked_today:
                message += f" | Already marked today: {', '.join(marked_today)}"
            flash(message, "success")
        elif marked_today:
            flash(f"Already marked today: {', '.join(marked_today)}", "info")
        else:
            flash("No known faces detected", "error")
        
        return redirect(url_for('index'))
    
    return render_template('attendance.html')
@app.route('/records')
def records():
    records = Attendance.query.order_by(Attendance.timestamp.desc()).all()
    
    # Calculate user statistics
    user_stats = defaultdict(lambda: {
        'name': '',
        'days_present': 0,
        'dates': set(),
        'first_date': None,
        'last_date': None
    })
    
    for record in records:
        user_id = record.user_id
        date = record.timestamp.date()
        user_stats[user_id]['name'] = record.name
        user_stats[user_id]['dates'].add(date)
    
    # Process the stats
    final_stats = []
    for user_id, stats in user_stats.items():
        dates = sorted(stats['dates'])
        days_present = len(dates)
        first_date = dates[0] if dates else None
        last_date = dates[-1] if dates else None
        
        # Calculate attendance rate
        total_days = (datetime.now().date() - first_date).days + 1 if first_date else 1
        attendance_rate = round((days_present / total_days) * 100) if total_days > 0 else 0
        
        final_stats.append({
            'user_id': user_id,
            'name': stats['name'],
            'days_present': days_present,
            'first_date': first_date,
            'last_date': last_date,
            'attendance_rate': attendance_rate
        })
    
    # Sort by most active users first
    final_stats.sort(key=lambda x: x['days_present'], reverse=True)
    
    return render_template('records.html', 
                         records=records[:100],  # Limit to 100 most recent records
                         user_stats=final_stats)

if __name__ == '__main__':
    # Pre-load known faces when starting the app
    with app.app_context():
        load_known_faces()
    app.run(debug=True)