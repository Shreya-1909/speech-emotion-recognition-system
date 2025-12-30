
import os, numpy as np
from utils import extract_mfcc, get_emotion_from_filename
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

DATA_DIR = "data/audio_speech_actors_01-24"
X, y = [], []

for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)
            X.append(extract_mfcc(path))
            y.append(get_emotion_from_filename(f))
            X.append(extract_mfcc(path, augment=True))
            y.append(get_emotion_from_filename(f))

max_len = max(m.shape[0] for m in X)
X = np.array([np.pad(m, ((0, max_len - m.shape[0]), (0, 0))) for m in X])

encoder = LabelEncoder()
y = to_categorical(encoder.fit_transform(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=X.shape[1:]),
    Dropout(0.3),
    LSTM(64),
    Dense(y.shape[1], activation="softmax")
])

model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))
model.save("model/ser_model.h5")
