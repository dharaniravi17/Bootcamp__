import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from React

# Load pre-trained model (update with your model's path)
MODEL_PATH = "music_genre_classifier.h5"  # Update with your actual model filename
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# Define genre labels (update based on your model training)
genres = ["classical", "jazz", "rock", "hiphop", "blues", "metal", "pop", "reggae"]

@app.route('/predict', methods=['POST'])
def predict_genre():
    if 'file' not in request.files:
        print("🚨 No file uploaded!")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    print(f"📂 Received file: {file.filename}")

    try:
        # Load audio
        signal, sr = librosa.load(file, sr=22050)
        print(f"🎵 Loaded audio file with sample rate: {sr}")

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)
        print(f"📊 Extracted MFCC shape: {mfcc.shape}")

        # Compute mean MFCCs across time axis
        mfcc = np.mean(mfcc.T, axis=0)
        print(f"📏 MFCC mean shape: {mfcc.shape}")

        # Reshape MFCCs to match model input shape
        mfcc = mfcc.reshape(1, -1)
        print(f"🔄 Reshaped MFCC shape: {mfcc.shape}")

        # Check model input shape
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize MFCCs

        print(f"🛠️ Model expects input shape: {model.input_shape}")

        # Predict genre
        prediction = model.predict(mfcc)
        genre_index = np.argmax(prediction)
        genre_name = genres[genre_index]
        print(f"🎶 Predicted Genre: {genre_name}")

        return jsonify({'genre': genre_name})

    except Exception as e:
        print(f"❌ Error predicting genre: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
