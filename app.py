# --------------------------------------------------------------
# app.py – FINAL: FACE DETECTION + USER CHECK + LOWERCASE ONLY
# --------------------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import face_recognition
import pickle
import base64
import numpy as np
import glob
import threading
import logging
import re

app = Flask(__name__, static_folder='.')
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faceapp")

_db_lock = threading.Lock()
DATABASE = 'database'
EMBEDDINGS_FILE = 'embeddings.pkl'

if not os.path.exists(DATABASE):
    os.makedirs(DATABASE)

with _db_lock:
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings_db = pickle.load(f)
    else:
        embeddings_db = {}

def save_embeddings():
    with _db_lock:
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings_db, f)

def validate_username(username):
    if not username:
        return False, "Username is empty"
    if not re.fullmatch(r"[a-z]+", username):
        return False, "Only lowercase letters (a-z) allowed"
    return True, ""

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/get_users', methods=['GET'])
def get_users():
    return jsonify(sorted(embeddings_db.keys()))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/register_image', methods=['POST'])
def register_image():
    data = request.json
    username = data.get('username', '').strip()
    image_b64 = data.get('image')

    valid, msg = validate_username(username)
    if not valid:
        return jsonify({'status': 'error', 'message': msg}), 400

    user_dir = os.path.join(DATABASE, username)
    if os.path.exists(user_dir) and len(glob.glob(os.path.join(user_dir, "*.jpg"))) > 0:
        return jsonify({'status': 'error', 'message': f'User "{username}" already registered'}), 400

    if not image_b64:
        return jsonify({'status': 'error', 'message': 'No image'}), 400
    try:
        b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        img_data = base64.b64decode(b64)
    except:
        return jsonify({'status': 'error', 'message': 'Invalid image data'}), 400

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Failed to decode image'}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model='hog')
    if not face_locations:
        return jsonify({'status': 'error', 'message': 'No face detected. Please face the camera.'}), 400

    os.makedirs(user_dir, exist_ok=True)
    count = len(glob.glob(os.path.join(user_dir, "*.jpg"))) + 1
    img_path = os.path.join(user_dir, f'image_{count}.jpg')
    cv2.imwrite(img_path, img)

    return jsonify({'status': 'success', 'count': count, 'message': f'Image {count} saved'})

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    username = data.get('username', '').strip()

    valid, msg = validate_username(username)
    if not valid:
        return jsonify({'status': 'error', 'message': msg}), 400

    user_dir = os.path.join(DATABASE, username)
    img_files = glob.glob(os.path.join(user_dir, "*.jpg"))
    if len(img_files) < 2:
        return jsonify({'status': 'error', 'message': f'Need at least 2 images, got {len(img_files)}'}), 400

    encodings = []
    for path in img_files:
        try:
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)
            if enc:
                encodings.append(enc[0])
        except Exception as e:
            log.warning(f"Encoding failed: {e}")
            continue

    if len(encodings) < 2:
        return jsonify({'status': 'error', 'message': 'Not enough valid faces'}), 400

    with _db_lock:
        embeddings_db[username] = encodings
        save_embeddings()

    return jsonify({'status': 'success', 'message': f'Trained {username}'})

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    data = request.json
    image_b64 = data.get('image')
    if not image_b64 or not embeddings_db:
        return jsonify({'status': 'error', 'message': 'No image or users'}), 400

    try:
        b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        img_data = base64.b64decode(b64)
    except:
        return jsonify({'status': 'error', 'message': 'Invalid image'}), 400

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Decode failed'}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model='hog')
    if not face_locations:
        return jsonify({'status': 'error', 'message': 'No face in query'}), 400

    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    known_encodings = []
    known_names = []
    for name, encs in embeddings_db.items():
        known_encodings.extend(encs)
        known_names.extend([name] * len(encs))

    faces = []
    for (top, right, bottom, left), enc in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.6)
        name = "Unknown"
        confidence = 0.0
        if True in matches:
            idx = matches.index(True)
            name = known_names[idx]
            distances = face_recognition.face_distance(known_encodings, enc)
            confidence = round(1.0 - min(distances), 2)
        faces.append({
            'x': left, 'y': top,
            'width': right - left, 'height': bottom - top,
            'name': name, 'confidence': confidence
        })

    message = "No match" if all(f['name'] == 'Unknown' for f in faces) else \
              f"Recognized: {', '.join([f['name'] for f in faces if f['name'] != 'Unknown'])}"

    return jsonify({'status': 'success', 'faces': faces, 'message': message})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
