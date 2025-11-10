from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import base64
import json
from PIL import Image
import io
import pickle

app = Flask(__name__)

# Configuration
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer/face_recognizer.pkl"
USERS_FILE = "users.json"
MAX_IMAGES = 12

# Initialize directories
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("trainer", exist_ok=True)

# Load registered users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def username_exists(username):
    users = load_users()
    return any(user.lower() == username.lower() for user in users)

# Face detector - handle Render environment
def get_face_cascade():
    try:
        # Try multiple paths for haarcascade
        paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
            './haarcascade_frontalface_default.xml'
        ]
        
        for path in paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if cascade.empty():
                    continue
                return cascade
        
        # Download if not found
        import urllib.request
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')
        return cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
    except Exception as e:
        print(f"Face cascade error: {e}")
        return None

face_cascade = get_face_cascade()

def detect_faces(image):
    if face_cascade is None:
        return [], image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces, gray

# Custom Face Recognizer
class CustomFaceRecognizer:
    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8):
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.histograms = []
        self.labels = []
        self.label_map = {}
        
    def lbp_histogram(self, image):
        # Simple LBP implementation
        height, width = image.shape
        lbp_image = np.zeros_like(image)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp_image[i, j] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])
        return hist / (hist.sum() + 1e-8)  # Normalize with epsilon to avoid division by zero
    
    def train(self, faces, labels):
        self.histograms = []
        self.labels = []
        
        for face, label in zip(faces, labels):
            # Ensure face is proper size
            if face.size > 0:
                hist = self.lbp_histogram(face)
                self.histograms.append(hist)
                self.labels.append(label)
        
        if not self.histograms:
            return False
            
        self.histograms = np.array(self.histograms)
        self.labels = np.array(self.labels)
        
        # Create label map
        unique_labels = np.unique(self.labels)
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
        return True
    
    def predict(self, face):
        if not self.histograms or face.size == 0:
            return -1, 0
            
        test_hist = self.lbp_histogram(face)
        
        best_distance = float('inf')
        best_label = -1
        
        for i, hist in enumerate(self.histograms):
            # Chi-squared distance
            distance = 0.5 * np.sum(((test_hist - hist) ** 2) / (test_hist + hist + 1e-10))
            
            if distance < best_distance:
                best_distance = distance
                best_label = self.labels[i]
        
        # Convert distance to confidence
        confidence = max(0, 100 - (best_distance * 100))
        return best_label, confidence
    
    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({
                'histograms': self.histograms,
                'labels': self.labels,
                'label_map': self.label_map
            }, f)
    
    def read(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.histograms = data['histograms']
                self.labels = data['labels']
                self.label_map = data['label_map']
                return True
        return False

def create_recognizer():
    return CustomFaceRecognizer()

# Routes
@app.route('/')
def index():
    return send_file('index.html')

@app.route('/get_users')
def get_users():
    try:
        users = load_users()
        return jsonify(users)
    except Exception as e:
        return jsonify([])

@app.route('/delete_user', methods=['POST'])
def delete_user():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"})
            
        username = data.get('username')
        if not username:
            return jsonify({"status": "error", "message": "Username is required"})
        
        users = load_users()
        
        if username in users:
            users.remove(username)
            save_users(users)
            
            # Remove user images
            user_folder = os.path.join(DATASET_PATH, username)
            if os.path.exists(user_folder):
                import shutil
                shutil.rmtree(user_folder)
            
            # Retrain model
            train_model()
            
            return jsonify({"status": "success", "message": f"User '{username}' deleted successfully"})
        
        return jsonify({"status": "error", "message": "User not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"})

@app.route('/register_image', methods=['POST'])
def register_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"})
            
        username = data.get('username')
        image_data = data.get('image')
        
        if not username:
            return jsonify({"status": "error", "message": "Username is required"})
        
        if not image_data:
            return jsonify({"status": "error", "message": "Image data is required"})
        
        if username_exists(username):
            return jsonify({"status": "error", "message": "Username already exists"})
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces, gray = detect_faces(image_np)
        
        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected"})
        
        # Create user folder
        user_folder = os.path.join(DATASET_PATH, username)
        os.makedirs(user_folder, exist_ok=True)
        
        existing_images = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
        if existing_images >= MAX_IMAGES:
            return jsonify({"status": "error", "message": f"Already captured {MAX_IMAGES} images"})
        
        image_count = existing_images + 1
        
        # Save face
        for i, (x, y, w, h) in enumerate(faces):
            if i >= 1:  # Save only one face per frame
                break
                
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            face_filename = f"image_{image_count}.jpg"
            face_path = os.path.join(user_folder, face_filename)
            cv2.imwrite(face_path, face_roi)
            break
        
        updated_count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
        
        return jsonify({
            "status": "success", 
            "count": updated_count,
            "message": f"Image {updated_count}/{MAX_IMAGES} captured"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"})
            
        username = data.get('username')
        
        if not username:
            return jsonify({"status": "error", "message": "Username is required"})
        
        # Add user to registered users list
        users = load_users()
        if username not in users:
            users.append(username)
            save_users(users)
        
        # Train the model
        success = train_model()
        
        if success:
            return jsonify({
                "status": "success", 
                "message": f"User '{username}' registered successfully!"
            })
        else:
            return jsonify({"status": "error", "message": "Training failed"})
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"Training error: {str(e)}"})

def train_model():
    try:
        if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
            return False
        
        recognizer = create_recognizer()
        
        faces = []
        ids = []
        id_counter = 0
        user_map = {}
        
        users = load_users()
        for username in users:
            user_map[username] = id_counter
            id_counter += 1
        
        total_images = 0
        
        for username in users:
            user_folder = os.path.join(DATASET_PATH, username)
            if os.path.exists(user_folder) and os.path.isdir(user_folder):
                user_id = user_map[username]
                
                image_files = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]
                
                for image_file in image_files:
                    image_path = os.path.join(user_folder, image_file)
                    try:
                        pil_image = Image.open(image_path).convert('L')
                        image_array = np.array(pil_image, 'uint8')
                        
                        if image_array.size > 0:
                            faces.append(image_array)
                            ids.append(user_id)
                            total_images += 1
                            
                    except Exception as e:
                        continue
        
        if len(faces) == 0:
            return False
        
        success = recognizer.train(faces, ids)
        if success:
            recognizer.save(TRAINER_PATH)
            return True
        return False
        
    except Exception as e:
        print(f"Training error: {e}")
        return False

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"})
            
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"status": "error", "message": "Image data is required"})
        
        # Check if model exists
        if not os.path.exists(TRAINER_PATH):
            return jsonify({
                "status": "error", 
                "message": "No trained model found"
            })
        
        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces, gray = detect_faces(image_np)
        
        if len(faces) == 0:
            return jsonify({
                "status": "success", 
                "faces": [],
                "message": "No faces detected"
            })
        
        # Load recognizer
        recognizer = create_recognizer()
        if not recognizer.read(TRAINER_PATH):
            return jsonify({"status": "error", "message": "Failed to load model"})
        
        users = load_users()
        user_id_map = {}
        for idx, username in enumerate(users):
            user_id_map[idx] = username
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            id_, confidence = recognizer.predict(roi_gray)
            
            if confidence > 50 and id_ in user_id_map:
                name = user_id_map[id_]
                confidence_percent = f"{confidence:.1f}%"
            else:
                name = "Unknown"
                confidence_percent = f"{confidence:.1f}%"
            
            recognized_faces.append({
                "x": int(x),
                "y": int(y), 
                "width": int(w),
                "height": int(h),
                "name": name,
                "confidence": confidence_percent
            })
        
        if recognized_faces and recognized_faces[0]['name'] != "Unknown":
            message = f"Recognized: {recognized_faces[0]['name']}"
        else:
            message = "No recognized faces"
        
        return jsonify({
            "status": "success",
            "faces": recognized_faces,
            "message": message
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Recognition error: {str(e)}"
        })

@app.route('/check_model')
def check_model():
    try:
        model_exists = os.path.exists(TRAINER_PATH)
        users = load_users()
        dataset_info = {}
        
        if os.path.exists(DATASET_PATH):
            for user in users:
                user_folder = os.path.join(DATASET_PATH, user)
                if os.path.exists(user_folder):
                    image_count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
                    dataset_info[user] = image_count
        
        return jsonify({
            "model_exists": model_exists,
            "users_count": len(users),
            "dataset_info": dataset_info
        })
    except:
        return jsonify({"model_exists": False, "users_count": 0, "dataset_info": {}})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
