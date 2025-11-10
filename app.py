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

# Face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except:
    import urllib.request
    url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
    urllib.request.urlretrieve(url, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    return faces, gray

# Custom LBPH Face Recognizer
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
        # Calculate LBP image
        lbp_image = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
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
        return hist / hist.sum()  # Normalize
    
    def train(self, faces, labels):
        self.histograms = []
        self.labels = []
        
        for face, label in zip(faces, labels):
            hist = self.lbp_histogram(face)
            self.histograms.append(hist)
            self.labels.append(label)
        
        self.histograms = np.array(self.histograms)
        self.labels = np.array(self.labels)
        
        # Create label map
        unique_labels = np.unique(labels)
        self.label_map = {i: label for i, label in enumerate(unique_labels)}
    
    def predict(self, face):
        test_hist = self.lbp_histogram(face)
        
        # Calculate chi-squared distance
        best_distance = float('inf')
        best_label = -1
        
        for i, hist in enumerate(self.histograms):
            # Chi-squared distance
            distance = 0.5 * np.sum(((test_hist - hist) ** 2) / (test_hist + hist + 1e-10))
            
            if distance < best_distance:
                best_distance = distance
                best_label = self.labels[i]
        
        # Convert distance to confidence (lower distance = higher confidence)
        confidence = max(0, 100 - (best_distance * 1000))  # Scale factor for better confidence values
        
        return best_label, confidence
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'histograms': self.histograms,
                'labels': self.labels,
                'label_map': self.label_map,
                'radius': self.radius,
                'neighbors': self.neighbors,
                'grid_x': self.grid_x,
                'grid_y': self.grid_y
            }, f)
    
    def read(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.histograms = data['histograms']
            self.labels = data['labels']
            self.label_map = data['label_map']
            self.radius = data['radius']
            self.neighbors = data['neighbors']
            self.grid_x = data['grid_x']
            self.grid_y = data['grid_y']

def create_recognizer():
    return CustomFaceRecognizer()

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/get_users')
def get_users():
    users = load_users()
    return jsonify(users)

@app.route('/delete_user', methods=['POST'])
def delete_user():
    data = request.get_json()
    username = data.get('username')
    
    users = load_users()
    
    if username in users:
        users.remove(username)
        save_users(users)
        
        user_folder = os.path.join(DATASET_PATH, username)
        if os.path.exists(user_folder):
            import shutil
            shutil.rmtree(user_folder)
        
        train_model()
        
        return jsonify({"status": "success", "message": f"User '{username}' deleted successfully"})
    
    return jsonify({"status": "error", "message": "User not found"})

@app.route('/register_image', methods=['POST'])
def register_image():
    try:
        data = request.get_json()
        username = data.get('username')
        image_data = data.get('image')
        
        if not username:
            return jsonify({"status": "error", "message": "Username is required"})
        
        if username_exists(username):
            return jsonify({"status": "error", "message": "Username already exists"})
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces, gray = detect_faces(image_np)
        
        if len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected. Please position yourself clearly in front of the camera."})
        
        user_folder = os.path.join(DATASET_PATH, username)
        os.makedirs(user_folder, exist_ok=True)
        
        existing_images = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
        image_count = existing_images + 1
        
        saved_count = 0
        for i, (x, y, w, h) in enumerate(faces):
            if saved_count >= 1:
                break
                
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            face_filename = f"image_{image_count}.jpg"
            face_path = os.path.join(user_folder, face_filename)
            cv2.imwrite(face_path, face_roi)
            saved_count += 1
            image_count += 1
        
        updated_count = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')])
        
        return jsonify({
            "status": "success", 
            "count": updated_count,
            "message": f"Image {updated_count}/{MAX_IMAGES} captured successfully"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error processing image: {str(e)}"})

@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json()
        username = data.get('username')
        
        print(f"Training started for user: {username}")
        
        users = load_users()
        if username not in users:
            users.append(username)
            save_users(users)
            print(f"Added {username} to users list")
        
        success = train_model()
        
        if success:
            print(f"Training completed successfully for {username}")
            return jsonify({
                "status": "success", 
                "message": f"User '{username}' registered successfully with {MAX_IMAGES} images! Model trained."
            })
        else:
            print(f"Training failed for {username}")
            return jsonify({"status": "error", "message": "Training failed - no valid faces found in dataset"})
            
    except Exception as e:
        print(f"Training error: {e}")
        return jsonify({"status": "error", "message": f"Training error: {str(e)}"})

def train_model():
    try:
        print("Starting model training...")
        
        if not os.path.exists(DATASET_PATH) or not os.listdir(DATASET_PATH):
            print("No dataset found for training")
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
        
        print(f"Users to train: {users}")
        
        total_images = 0
        
        for username in users:
            user_folder = os.path.join(DATASET_PATH, username)
            if os.path.exists(user_folder) and os.path.isdir(user_folder):
                user_id = user_map[username]
                print(f"Processing user: {username} (ID: {user_id})")
                
                image_files = [f for f in os.listdir(user_folder) if f.endswith('.jpg')]
                print(f"Found {len(image_files)} images for {username}")
                
                for image_file in image_files:
                    image_path = os.path.join(user_folder, image_file)
                    try:
                        pil_image = Image.open(image_path).convert('L')
                        image_array = np.array(pil_image, 'uint8')
                        
                        faces.append(image_array)
                        ids.append(user_id)
                        total_images += 1
                        
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")
                        continue
        
        print(f"Total images processed: {total_images}")
        print(f"Faces array length: {len(faces)}")
        print(f"IDs array length: {len(ids)}")
        
        if len(faces) == 0:
            print("No faces found for training")
            return False
        
        faces_np = np.array(faces)
        ids_np = np.array(ids)
        
        print(f"Training with {len(faces_np)} samples...")
        
        recognizer.train(faces_np, ids_np)
        recognizer.save(TRAINER_PATH)
        
        print(f"Model trained successfully and saved to {TRAINER_PATH}")
        print(f"Trained with {len(faces)} face samples from {len(set(ids))} users")
        
        return True
        
    except Exception as e:
        print(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/recognize_image', methods=['POST'])
def recognize_image():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not os.path.exists(TRAINER_PATH):
            return jsonify({
                "status": "error", 
                "message": "No trained model found. Please register users first."
            })
        
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
        
        recognizer = create_recognizer()
        recognizer.read(TRAINER_PATH)
        
        users = load_users()
        user_id_map = {}
        for idx, username in enumerate(users):
            user_id_map[idx] = username
        
        recognized_faces = []
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            id_, confidence = recognizer.predict(roi_gray)
            
            print(f"Recognition result - ID: {id_}, Confidence: {confidence}")
            
            if confidence > 50 and id_ in user_id_map:  # Higher confidence = better match
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
            message = f"Recognized: {recognized_faces[0]['name']} ({recognized_faces[0]['confidence']})"
        else:
            message = "No recognized faces or confidence too low"
        
        return jsonify({
            "status": "success",
            "faces": recognized_faces,
            "message": message
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"Recognition error: {str(e)}"
        })

@app.route('/check_model')
def check_model():
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

if __name__ == '__main__':
    print("Starting Face Recognition Server...")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Trainer path: {TRAINER_PATH}")
    print(f"Max images per user: {MAX_IMAGES}")
    
    recognizer = create_recognizer()
    if recognizer:
        print("Custom face recognizer initialized successfully")
    else:
        print("WARNING: Face recognizer could not be initialized")
    
    if os.path.exists(TRAINER_PATH):
        print("Pre-trained model found")
    else:
        print("No pre-trained model found")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
