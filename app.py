# --------------------------------------------------------------
# app.py   (FINAL: AUTO-CREATE DB, EMBEDDINGS IN ROOT, FULL RECOGNITION)
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
import tempfile
import logging

app = Flask(__name__, static_folder='.')
# allow restricting origins via env var CORS_ORIGINS (comma separated). Default '*' for dev.
_allowed = os.environ.get("CORS_ORIGINS", "*")
if _allowed == "*" or _allowed.strip() == "":
    CORS(app)
else:
    origins = [o.strip() for o in _allowed.split(",")]
    CORS(app, resources={r"/*": {"origins": origins}})

# basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("faceapp")

# thread lock to protect embeddings file & in-memory DB
_db_lock = threading.Lock()

# === PATHS ===
DATABASE = 'database'           # Folder for user images
EMBEDDINGS_FILE = 'embeddings.pkl'  # In same folder as app.py

# Create database folder
if not os.path.exists(DATABASE):
    os.makedirs(DATABASE)

# Load or initialize embeddings (protected)
with _db_lock:
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, 'rb') as f:
                embeddings_db = pickle.load(f)
        except Exception as e:
            log.exception("Failed to load embeddings file, starting fresh")
            embeddings_db = {}
    else:
        embeddings_db = {}  # {username: [emb1, emb2, ...]}

# Lightweight settings
MODEL_NAME = 'Facenet'
DETECTOR_BACKEND = 'opencv'
# Use cosine distance on normalized embeddings; adjust threshold if needed
THRESHOLD = 0.40

# OpenCV face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def save_embeddings():
    """Save embeddings to root folder atomically and thread-safe"""
    tmp_fd, tmp_path = tempfile.mkstemp(dir='.', prefix='embeddings_', suffix='.pkl')
    os.close(tmp_fd)
    try:
        with _db_lock:
            with open(tmp_path, 'wb') as f:
                pickle.dump(embeddings_db, f)
            # atomic replace
            os.replace(tmp_path, EMBEDDINGS_FILE)
    except Exception:
        log.exception("Failed to save embeddings")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass


# --- helper: normalize embedding to plain list ---
def _normalize_embedding(emb):
    v = np.array(emb, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(v)
    if norm == 0:
        return v.tolist()
    v = v / norm
    return v.tolist()


def _extract_embedding(rep):
    """
    Accept different DeepFace.represent return formats and return a 1D numeric vector
    or None if extraction fails.
    """
    if rep is None:
        return None
    # dict with 'embedding' key
    if isinstance(rep, dict):
        if 'embedding' in rep:
            return np.array(rep['embedding'], dtype=np.float32).reshape(-1)
        # sometimes there are other keys containing vector
        for k in ('embeddings', 'represent'):
            if k in rep:
                try:
                    return np.array(rep[k], dtype=np.float32).reshape(-1)
                except Exception:
                    pass
        return None
    # list: could be list of numbers or list of dicts (faces)
    if isinstance(rep, list):
        if len(rep) == 0:
            return None
        first = rep[0]
        if isinstance(first, dict) and 'embedding' in first:
            return np.array(first['embedding'], dtype=np.float32).reshape(-1)
        # assume list of numbers (vector)
        try:
            return np.array(rep, dtype=np.float32).reshape(-1)
        except Exception:
            return None
    # numpy array or sequence
    try:
        return np.array(rep, dtype=np.float32).reshape(-1)
    except Exception:
        return None


# ------------------- ROUTES -------------------
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/get_users', methods=['GET'])
def get_users():
    return jsonify(sorted(embeddings_db.keys()))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'users': len(embeddings_db)}), 200


# Example: protect operations that mutate embeddings_db
@app.route('/delete_user', methods=['POST'])
def delete_user():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'No username'})

    # Remove user folder
    user_dir = os.path.join(DATABASE, username)
    if os.path.isdir(user_dir):
        import shutil
        shutil.rmtree(user_dir)

    # Remove from embeddings (thread-safe)
    with _db_lock:
        if username in embeddings_db:
            del embeddings_db[username]
            save_embeddings()

    return jsonify({'status': 'success', 'message': f'User \"{username}\" deleted'})


# ------------------- REGISTER IMAGE -------------------
@app.route('/register_image', methods=['POST'])
def register_image():
    data = request.json
    username = data.get('username')
    image_b64 = data.get('image')

    if not username or not image_b64:
        return jsonify({'status': 'error', 'message': 'Missing data'})

    # Allow adding images even if user already exists (append more before training)
    user_dir = os.path.join(DATABASE, username)
    os.makedirs(user_dir, exist_ok=True)

    # Decode image
    try:
        if ',' in image_b64:
            b64 = image_b64.split(',', 1)[1]
        else:
            b64 = image_b64
        img_data = base64.b64decode(b64)
    except:
        return jsonify({'status': 'error', 'message': 'Invalid image'})

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Decode failed'})

    # Face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        # save debug copy
        dbg_path = os.path.join(user_dir, f'no_face_debug_{len(glob.glob(os.path.join(user_dir,"*.jpg")))+1}.jpg')
        try:
            cv2.imwrite(dbg_path, img)
        except Exception:
            pass
        log.info("No face detected; saved debug image to: %s", dbg_path)
        return jsonify({'status': 'error', 'message': 'No face detected (try different photo or better lighting)'}), 400

    # Save image (save full image; train step will extract embeddings)
    count = len(glob.glob(os.path.join(user_dir, "*.jpg"))) + 1
    img_path = os.path.join(user_dir, f'image_{count}.jpg')
    cv2.imwrite(img_path, img)

    return jsonify({'status': 'success', 'count': count})


# Wrap train endpoint to persist under lock
@app.route('/train', methods=['POST'])
def train():
    data = request.json
    username = data.get('username')
    if not username:
        return jsonify({'status': 'error', 'message': 'No username'})

    user_dir = os.path.join(DATABASE, username)
    img_files = glob.glob(os.path.join(user_dir, "*.jpg"))
    if len(img_files) < 1:
        return jsonify({'status': 'error', 'message': f'Need at least 1 image, got {len(img_files)}'})

    embeddings = []
    for img_path in img_files:
        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            vec = _extract_embedding(rep)
            if vec is None:
                log.warning(f"Embedding extraction returned None for {img_path}; repr type={type(rep)}")
                continue
            norm_emb = _normalize_embedding(vec)
            embeddings.append(norm_emb)
        except Exception as e:
            log.exception("Embedding failed for %s", img_path)
            continue

    if embeddings:
        with _db_lock:
            embeddings_db[username] = embeddings
            save_embeddings()
        return jsonify({'status': 'success', 'message': 'Model trained successfully', 'count': len(embeddings)})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to extract any embeddings'})


# ------------------- RECOGNIZE -------------------
@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    data = request.json
    image_b64 = data.get('image')
    if not image_b64:
        return jsonify({'status': 'error', 'message': 'No image'})

    if not embeddings_db:
        return jsonify({'status': 'error', 'message': 'No trained users in database'})

    try:
        if ',' in image_b64:
            b64 = image_b64.split(',', 1)[1]
        else:
            b64 = image_b64
        img_data = base64.b64decode(b64)
    except:
        return jsonify({'status': 'error', 'message': 'Invalid image'})

    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'status': 'error', 'message': 'Decode failed'})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    faces = []
    msg_parts = []

    temp_path = os.path.join(DATABASE, 'temp_face.jpg')

    for (x, y, w, h) in detections:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite(temp_path, face_img)

        try:
            rep = DeepFace.represent(
                img_path=temp_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False
            )
            qvec = _extract_embedding(rep)
            if qvec is None:
                raise ValueError("Failed to extract query embedding")

            q = np.array(_normalize_embedding(qvec), dtype=np.float32)

            best_name = 'Unknown'
            best_score = 0.0
            best_dist = float('inf')

            # embeddings_db stores normalized plain lists
            for name, embs in embeddings_db.items():
                for emb in embs:
                    emb_arr = np.array(emb, dtype=np.float32)
                    dot = float(np.dot(q, emb_arr))
                    if dot > 1.0: dot = 1.0
                    if dot < -1.0: dot = -1.0
                    dist = 1.0 - dot
                    if dist < best_dist:
                        best_dist = dist
                        best_name = name

            if best_dist <= THRESHOLD:
                score = round(max(0.0, (THRESHOLD - best_dist) / THRESHOLD), 2)
            else:
                score = 0.0
                best_name = 'Unknown'

            faces.append({
                'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                'name': best_name, 'confidence': score
            })
            msg_parts.append(f'{best_name} ({score})')

        except Exception as e:
            log.exception("Recognition error: %s", e)
            faces.append({
                'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h),
                'name': 'Unknown', 'confidence': 0
            })
            msg_parts.append('Unknown')

        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

    # Build message
    if not msg_parts:
        message = 'No faces detected.'
    elif len(msg_parts) == 1:
        message = 'Unknown face detected.' if 'Unknown' in msg_parts[0] else f'Identified him/her as {msg_parts[0]}'
    else:
        message = f'Identified them: {", ".join(msg_parts)}'

    return jsonify({'status': 'success', 'faces': faces, 'message': message})


# ------------------- RUN SERVER -------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))    # use platform PORT
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
    log.info(f"Server starting on 0.0.0.0:{port} debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)