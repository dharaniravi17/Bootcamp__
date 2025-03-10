import os
import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Load Dataset (Change path to your dataset location)
DATA_PATH = "C:\\Users\\Dharani Ravi\\Desktop\\Music\\data\\genres"

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

X, y = [], []

# Convert audio files to MFCC features
for genre in genres:
    genre_path = os.path.join(DATA_PATH, genre)
    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        signal, sr = librosa.load(file_path, sr=22050)  # Load song
        mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=20)  # Extract MFCC
        mfcc = np.mean(mfcc.T, axis=0)  # Average over time
        X.append(mfcc)
        y.append(genres.index(genre))

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN Model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')  # 10 genres
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save Model
model.save("music_genre_classifier.h5")

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')
plt.show()

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")


