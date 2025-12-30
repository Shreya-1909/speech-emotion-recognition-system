
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from utils import extract_mfcc

EMOTIONS = ["neutral","calm","happy","sad","angry","fearful","disgust","surprised"]
model = load_model("model/ser_model.h5")

st.title("🎤 Speech Emotion Recognition")

audio_file = st.file_uploader("Upload WAV file", type=["wav"])
if audio_file:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())
    mfcc = extract_mfcc("temp.wav")
    mfcc = np.pad(mfcc, ((0, model.input_shape[1]-mfcc.shape[0]), (0,0)))
    mfcc = np.expand_dims(mfcc, axis=0)
    emotion = EMOTIONS[np.argmax(model.predict(mfcc))]
    st.success(f"Predicted Emotion: {emotion}")
