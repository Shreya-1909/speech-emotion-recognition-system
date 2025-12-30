
# Speech Emotion Recognition using RAVDESS (Enhanced)

## Features
- MFCC-based LSTM model
- Data augmentation (noise + pitch shift)
- Confusion matrix visualization
- Streamlit web application

## Dataset
Download RAVDESS:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Place it inside:
data/audio_speech_actors_01-24/

## Install
pip install -r requirements.txt

## Train
python train.py

## Evaluate
python evaluation.py

## Run Web App
streamlit run app.py
