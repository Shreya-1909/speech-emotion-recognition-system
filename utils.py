
import librosa
import numpy as np

EMOTION_MAP = {
    "01": "neutral","02": "calm","03": "happy","04": "sad",
    "05": "angry","06": "fearful","07": "disgust","08": "surprised"
}

def extract_mfcc(file_path, augment=False, n_mfcc=40):
    audio, sr = librosa.load(file_path, sr=22050)
    if augment:
        audio += 0.005 * np.random.randn(len(audio))
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=2)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T

def get_emotion_from_filename(filename):
    return EMOTION_MAP[filename.split("-")[2]]
