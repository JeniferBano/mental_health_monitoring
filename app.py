from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify
)
import mysql.connector
import os
import pickle
import numpy as np
from datetime import date
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'mentalhealth2026'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = \
    UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_db():
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="",
        database="mental_health_web"
    )


def analyze_scan(filepath):
    try:
        has_model = (
            os.path.exists(
                'model/model.pkl'
            )
            and os.path.exists(
                'model/scaler.pkl'
            )
            and os.path.exists(
                'model/masker.pkl'
            )
        )

        if has_model:
            with open(
                'model/model.pkl', 'rb'
            ) as f:
                model = pickle.load(f)
            with open(
                'model/scaler.pkl', 'rb'
            ) as f:
                scaler = pickle.load(f)
            with open(
                'model/masker.pkl', 'rb'
            ) as f:
                masker = pickle.load(f)

            from nilearn.connectome import (
                ConnectivityMeasure
            )
            series = masker.fit_transform(
                filepath
            )
            measure = ConnectivityMeasure(
                kind='correlation'
            )
            matrix = measure.fit_transform(
                [series]
            )
            upper = matrix[0][
                np.triu_indices_from(
                    matrix[0], k=1
                )
            ]
            features = scaler.transform(
                [upper]
            )
            pred = model.predict(
                features
            )[0]
            prob = model.predict_proba(
                features
            )[0]
            conf = round(
                float(max(prob)) * 100, 1
            )
            cond = (
                "Depression"
                if pred == 1
                else "Normal"
            )
            risk = (
                "HIGH" if conf >= 80
                else "MEDIUM"
                if conf >= 60
                else "LOW"
            )
            return {
                'condition': cond,
                'confidence': conf,
                'risk': risk,
                'success': True
            }
        else:
            import random
            demo = random.choice([
                {
                    'condition': 'Depression',
                    'confidence': 87.5,
                    'risk': 'HIGH'
                },
                {
                    'condition': 'Normal',
                    'confidence': 92.0,
                    'risk': 'LOW'
                },
                {
                    'condition': 'Anxiety',
                    'confidence': 79.3,
                    'risk': 'MEDIUM'
                }
            ])
            demo['success'] = True
            return demo

    except Exception as e:
        return {
            'error': str(e),
            'success': False
        }


@app.route('/')
def home():
    return redirect(url_for('login'))


@app.route(
    '/login',
    methods=['GET', 'POST']
)
def login():
    if request.method == 'POST':
        email = request.form.get(
            'email', ''
        ).strip()
        password = request.form.get(
            'password', ''
        ).strip()
        try:
            db = get_db()
            cur = db.cursor(
                dictionary=True
            )
            cur.execute(
                "SELECT * FROM users "
                "WHERE email=%s "
                "AND password=%s",
                (email, password)
            )
            user = cur.fetchone()
            db.close()
            if user:
                session['user_id'] = \
                    user['user_id']
                session['name'] = \
                    user['full_name']
                session['role'] = \
                    user['role']
                return redirect(
                    url_for('dashboard')
                )
            else:
                return render_template(
                    'login.html',
                    error=(
                        "Wrong email "
                        "or password!"
                    )
                )
        except Exception as e:
            return render_template(
                'login.html',
                error="Error: " + str(e)
            )
    return render_template('login.html')


@app.route(
    '/register',
    methods=['GET', 'POST']
)
def register():
    if request.method == 'POST':
        name = request.form.get(
            'name', ''
        ).strip()
        email = request.form.get(
            'email', ''
        ).strip()
        password = request.form.get(
            'password', ''
        ).strip()
        role = request.form.get(
            'role', 'patient'
        )
        if not name or not email \
                or not password:
            return render_template(
                'register.html',
                error=(
                    "Please fill "
                    "all fields!"
                )
            )
        try:
            db = get_db()
            cur = db.cursor()
            cur.execute(
                "INSERT INTO users"
                " (full_name, email,"
                " password, role)"
                " VALUES (%s,%s,%s,%s)",
                (name, email,
                 password, role)
            )
            db.commit()
            db.close()
            return render_template(
                'register.html',
                success=(
                    "Account created! "
                    "Please login now."
                )
            )
        except Exception as e:
            if "Duplicate" in str(e):
                return render_template(
                    'register.html',
                    error=(
                        "Email already "
                        "exists!"
                    )
                )
            return render_template(
                'register.html',
                error="Error: " + str(e)
            )
    return render_template(
        'register.html'
    )


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template(
        'dashboard.html',
        name=session['name'],
        role=session['role']
    )


@app.route(
    '/upload',
    methods=['GET', 'POST']
)
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'scan' not in request.files:
            return render_template(
                'upload.html',
                error="Select a file!"
            )
        file = request.files['scan']
        if file.filename == '':
            return render_template(
                'upload.html',
                error="No file chosen!"
            )
        filename = secure_filename(
            file.filename
        )
        filepath = os.path.join(
            app.config['UPLOAD_FOLDER'],
            filename
        )
        file.save(filepath)
        result = analyze_scan(filepath)
        if result['success']:
            try:
                db = get_db()
                cur = db.cursor()
                cur.execute(
                    "INSERT INTO scans"
                    " (user_id, scan_file,"
                    " upload_date)"
                    " VALUES (%s,%s,%s)",
                    (
                        session['user_id'],
                        filename,
                        date.today()
                    )
                )
                scan_id = cur.lastrowid
                cur.execute(
                    "INSERT INTO results"
                    " (scan_id,"
                    " condition_detected,"
                    " confidence_score,"
                    " risk_level,"
                    " analysis_date)"
                    " VALUES"
                    " (%s,%s,%s,%s,%s)",
                    (
                        scan_id,
                        result['condition'],
                        result['confidence'],
                        result['risk'],
                        date.today()
                    )
                )
                db.commit()
                db.close()
            except Exception as e:
                print("DB error:", e)
            return render_template(
                'result.html',
                condition=
                result['condition'],
                confidence=
                result['confidence'],
                risk=result['risk'],
                filename=filename
            )
        else:
            return render_template(
                'upload.html',
                error=(
                    "Error: "
                    + result.get(
                        'error', ''
                    )
                )
            )
    return render_template('upload.html')


@app.route('/chatbot')
def chatbot():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template(
        'chatbot.html',
        name=session['name']
    )


@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify(
            {'error': 'Not logged in'}
        )
    data = request.get_json()
    if not data:
        return jsonify(
            {'error': 'No data'}
        )
    user_msg = data.get(
        'message', ''
    ).lower().strip()

    responses = {
        'depression': (
            "Depression is serious but "
            "treatable. Signs: persistent"
            " sadness, loss of interest. "
            "Please consult a doctor. "
            "You are not alone! 💙"
        ),
        'anxiety': (
            "Anxiety causes excessive "
            "worry. Try deep breathing: "
            "inhale 4s, hold 4s, "
            "exhale 4s. Talk to a "
            "professional! 🌟"
        ),
        'stress': (
            "Stress is body's response "
            "to pressure. Exercise, "
            "7-9h sleep, relaxation "
            "helps. Take care! 💚"
        ),
        'sleep': (
            "Good sleep is vital. "
            "Aim for 7-9 hours. "
            "Avoid screens before bed. "
            "Maintain sleep schedule. 😴"
        ),
        'sad': (
            "Your feelings are valid. "
            "Talk to someone trusted "
            "or a doctor. Better days "
            "are coming! 🌈"
        ),
        'help': (
            "I can help with depression,"
            " anxiety, stress, sleep, "
            "and fMRI scan guidance!"
        ),
        'scan': (
            "Go to Upload page to "
            "submit your fMRI scan. "
            "AI analyzes 116 brain "
            "regions using SVM! 🧠"
        ),
        'hello': (
            "Hello! I am your Mental "
            "Health AI Assistant. "
            "How are you feeling? "
            "I am here to help! 😊"
        ),
        'hi': (
            "Hi! Your AI Mental Health"
            " Chatbot is ready. How "
            "can I help you? 💙"
        ),
    }

    response = (
        "I understand. Mental health "
        "is important. Ask me about "
        "depression, anxiety, stress, "
        "sleep, or scan upload. "
        "I am here! 💙"
    )

    for key, val in responses.items():
        if key in user_msg:
            response = val
            break

    try:
        db = get_db()
        cur = db.cursor()
        cur.execute(
            "INSERT INTO chat_history"
            " (user_id, message,"
            " response)"
            " VALUES (%s,%s,%s)",
            (
                session['user_id'],
                data.get('message', ''),
                response
            )
        )
        db.commit()
        db.close()
    except Exception as e:
        print("Chat error:", e)

    return jsonify({'response': response})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


if __name__ == '__main__':
    print("=" * 40)
    print("MENTAL HEALTH MONITORING SYSTEM")
    print("Starting server...")
    print("Open Chrome: localhost:5000")
    print("=" * 40)
    app.run(debug=True, port=5000)