import h5py
try:
    with h5py.File("music_genre_model.h5", "r") as f:
        print("Model file is valid!")
except Exception as e:
    print(f"Error loading model: {e}")


# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import librosa
# import numpy as np
# import tensorflow as tf
# import pickle
# import h5py

# app = Flask(__name__)
# CORS(app)  # Enable CORS for React

# # Load the trained model
# MODEL_PATH = "music_genre_model.h5"
# LABEL_ENCODER_PATH = "label_encoder.pkl"

# # Check if model file exists
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train the model first.")

# try:
#     # Check if the file is a valid H5 file
#     with h5py.File(MODEL_PATH, "r") as f:
#         print("H5 file is valid. Loading model...")
#     model = tf.keras.models.load_model(MODEL_PATH)
# except Exception as e:
#     raise ValueError(f"Error loading model: {e}")

# # Load label encoder
# if not os.path.exists(LABEL_ENCODER_PATH):
#     raise FileNotFoundError(f"Label encoder file '{LABEL_ENCODER_PATH}' not found.")

# with open(LABEL_ENCODER_PATH, "rb") as f:
#     label_encoder = pickle.load(f)

# def extract_features(file_path):
#     """Extract MFCC features from an audio file."""
#     try:
#         y, sr = librosa.load(file_path, duration=5)
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         mfccs_mean = np.mean(mfccs.T, axis=0)
#         return mfccs_mean
#     except Exception as e:
#         print(f"Error processing {file_path}: {e}")
#         return None

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     file_path = "temp.au"
#     file.save(file_path)

#     features = extract_features(file_path)
#     if features is None:
#         return jsonify({"error": "Invalid audio file"}), 400

#     features = np.expand_dims(features, axis=0)  # Reshape for model
#     prediction = model.predict(features)
#     genre_index = np.argmax(prediction)
#     predicted_genre = label_encoder.inverse_transform([genre_index])[0]

#     return jsonify({"genre": predicted_genre})

# if __name__ == "__main__":
#     app.run(debug=True)

