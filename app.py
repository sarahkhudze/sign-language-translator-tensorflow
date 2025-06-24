import os

# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import mediapipe as mp

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# Define model architecture
def create_model():
    return Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(25, activation='softmax')
    ])

# Load model weights with fallback
model_loaded = False
try:
    model = create_model()
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'chichewa_sign_language.h5')
    model.load_weights(model_path)
    model_loaded = True
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    model = create_model()
    print("⚠️ Using fallback model with random weights")

def preprocess_frame(frame):
    try:
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            x_coords = [lm.x * w for lm in results.multi_hand_landmarks[0].landmark]
            y_coords = [lm.y * h for lm in results.multi_hand_landmarks[0].landmark]

            # Calculate bounding box with padding
            padding = 20
            x_min = max(0, int(min(x_coords)) - padding)
            x_max = min(w, int(max(x_coords)) + padding)
            y_min = max(0, int(min(y_coords)) - padding)
            y_max = min(h, int(max(y_coords)) + padding)

            # Crop hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
            hand_img = cv2.resize(hand_img, (28, 28))

            # Normalize and reshape for model
            return hand_img.reshape(1, 28, 28, 1) / 255.0

        return None

    except Exception as e:
        print(f"⚠️ Preprocessing error: {str(e)}")
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded properly, cannot predict.'})

    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data URL prefix
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        processed = preprocess_frame(frame)
        if processed is not None:
            prediction = model.predict(processed)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]

            if confidence < 0.7:
                return jsonify({'error': 'Low confidence prediction', 'confidence': float(confidence)})

            return jsonify({
                'prediction': prediction[0].tolist(),
                'predicted_class': int(predicted_index),
                'confidence': float(confidence),
                'status': 'success'
            })
        else:
            return jsonify({'error': 'No hands detected'})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
