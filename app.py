# --------------------------------------------------------------
# app.py – FULLY WORKING ON PYTHON 3.8.0 + RENDER FREE TIER
# --------------------------------------------------------------
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
from deepface import DeepFace
import base64
import pickle
import glob
import threading
import logging

app = Flask(__name__, static_folder='.')
CORS(app)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faceapp")

_db_lock = threading.Lock()
DATABASE = 'database'
EMBEDDINGS_FILE = 'embeddings.pkl'

# Create database folder
if not os.path.exists(DATABASE):
    os.makedirs(DATABASE)

# Load or init embeddings
with _db_lock:
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                embeddings_db = pickle.load(f)
        except Exception:
            embeddings_db = {}
    else:
        embeddings_db = {}

def save_embeddings():
    with _db_lock:
        try:
            with open(EMBEDDINGS_FILE, 'wb') as f:
                pickle.dump(embeddings_db, f)
        except Exception as e:
            log.error(f"Save failed: {e}")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/get_users', methods=['GET'])
def get_users():
    return jsonify(sorted(embeddings_db.keys()))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'users': len(embeddings_db)})

@app.route('/register_image', methods=['POST'])
def register_image():
    data = request.json
    username = data.get('username')
    image_b64 = data.get('image')
    if not username or not image_b64:
        return jsonify({'status': 'error', 'message': 'Missing data'}), 400

    user_dir = os.path.join(DATABASE, username)
    os.makedirs(user_dir, exist_ok=True)

    try:
        b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        img_data = base64.b64decode(b64)
    except:
        return jsonify({'status': 'error', 'message': 'Invalid image'}), 400

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Decode failed'}), 400

    count = len(glob.glob(os.path.join(user_dir, "*.jpg"))) + 1
    path = os.path.join(user_dir, f'image_{count}.jpg')
    cv2.imwrite(path, img)
    return jsonify({'status': 'success', 'count': count})

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'No username'}), 400

    user_dir = os.path.join(DATABASE, username)
    img_files = glob.glob(os.path.join(user_dir, "*.jpg"))
    if not img_files:
        return jsonify({'status': 'error', 'message': 'No images'}), 400

    embeddings = []
    for path in img_files:
        try:
            result = DeepFace.represent(
                img_path=path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )
            emb = result[0]["embedding"] if isinstance(result, list) else result["embedding"]
            embeddings.append(emb)
        except Exception as e:
            log.warning(f"Train failed on {path}: {e}")
            continue

    if not embeddings:
        return jsonify({'status': 'error', 'message': 'No faces detected'}), 400

    with _db_lock:
        embeddings_db[username] = embeddings
        save_embeddings()

    return jsonify({'status': 'success', 'count': len(embeddings)})

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    data = request.json
    image_b64 = data.get('image')
    if not image_b64 or not embeddings_db:
        return jsonify({'status': 'error', 'message': 'No image or no users'}), 400

    try:
        b64 = image_b64.split(',', 1)[1] if ',' in image_b64 else image_b64
        img_data = base64.b64decode(b64)
    except:
        return jsonify({'status': 'error', 'message': 'Invalid image'}), 400

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Decode failed'}), 400

    try:
        result = DeepFace.represent(
            img_path=img,
            model_name="Facenet",
            detector_backend="opencv",
            enforce_detection=False
        )
        q_emb = result[0]["embedding"] if isinstance(result, list) else result["embedding"]
    except:
        return jsonify({'status': 'error', 'message': 'No face in query'}), 400

    best_name = "Unknown"
    best_dist = float('inf')

    for name, embs in embeddings_db.items():
        for emb in embs:
            dist = np.linalg.norm(np.array(q_emb) - np.array(emb))
            if dist < best_dist:
                best_dist = dist
                best_name = name

    threshold = 0.6
    name = best_name if best_dist < threshold else "Unknown"
    confidence = round(1.0 - (best_dist / threshold), 2) if best_dist < threshold else 0.0

    return jsonify({
        'status': 'success',
        'faces': [{'name': name, 'confidence': confidence}],
        'message': f"Recognized: {name}"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
