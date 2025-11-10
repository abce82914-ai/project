# --------------------------------------------------------------
# app.py – OPENCV-ONLY: NO DLIB, NO ERRORS, FULL FEATURES
# --------------------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import base64
import pickle
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

# Load Haar cascade for face detection (built into OpenCV)
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with _db_lock:
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings_db = pickle.load(f)
    else:
        embeddings_db = {}  # {username: [feature_vec1, feature_vec2, ...]}

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

def extract_features(img):
    """Simple feature extraction: grayscale, resize to 64x64, flatten pixel values"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (64, 64))
    return small.flatten().astype(np.float32)

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

    # FACE DETECTION – ONLY IF FACE IS FOUND
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
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

    features = []
    for path in img_files:
        try:
            img = cv2.imread(path)
            feat = extract_features(img)
            features.append(feat)
        except Exception as e:
            log.warning(f"Feature extraction failed: {e}")
            continue

    if len(features) < 2:
        return jsonify({'status': 'error', 'message': 'Not enough valid faces'}), 400

    # Average features for the user
    avg_features = np.mean(features, axis=0).tolist()
    with _db_lock:
        embeddings_db[username] = [avg_features] * len(features)  # Simple average as "multiple encodings"
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
    if len(faces) == 0:
        return jsonify({'status': 'error', 'message': 'No face in query'}), 400

    q_feat = extract_features(img)

    best_name = "Unknown"
    best_dist = float('inf')
    threshold = 5000.0  # Tune based on your images (lower = stricter)

    for name, feats in embeddings_db.items():
        for feat in feats:
            dist = np.linalg.norm(q_feat - np.array(feat))
            if dist < best_dist:
                best_dist = dist
                best_name = name

    confidence = round(max(0.0, 1.0 - (best_dist / threshold)), 2)
    name = best_name if best_dist < threshold else "Unknown"

    # Bounding box from first face (simple, single-face assumption)
    (x, y, w, h) = faces[0]
    faces_list = [{'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h), 'name': name, 'confidence': confidence}]

    message = f"Recognized: {name}" if name != "Unknown" else "No match"

    return jsonify({'status': 'success', 'faces': faces_list, 'message': message})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
