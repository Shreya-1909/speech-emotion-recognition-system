
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from utils import extract_mfcc

model = load_model("model/ser_model.h5")
y_true, y_pred = [], []

for root, _, files in os.walk("data/audio_speech_actors_01-24"):
    for f in files:
        if f.endswith(".wav"):
            mfcc = extract_mfcc(os.path.join(root, f))
            mfcc = np.pad(mfcc, ((0, model.input_shape[1]-mfcc.shape[0]), (0,0)))
            mfcc = np.expand_dims(mfcc, axis=0)
            y_pred.append(np.argmax(model.predict(mfcc)))
            y_true.append(int(f.split("-")[2]) - 1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()
