**🎤 Speech Emotion Recognition System**

**End-to-End Machine Learning Pipeline**

An end-to-end Speech Emotion Recognition (SER) system that classifies human emotions from speech audio using MFCC feature extraction, an LSTM-based deep learning model, and a Streamlit web application for real-time inference.

This project demonstrates production-style ML engineering, including data preprocessing, model training, evaluation, and deployment.

**🔍 Problem Statement**

Understanding emotional context in speech is critical for applications such as virtual assistants, customer support analytics, and human-computer interaction.
This project predicts the emotional state of a speaker from short audio clips using supervised machine learning.

**✨ Key Features**
- End-to-end machine learning pipeline (training → evaluation → inference)
- MFCC-based audio feature extraction
- LSTM neural network for temporal speech modeling
- Audio data augmentation to improve robustness
- Confusion matrix–based evaluation
- Streamlit web app for interactive emotion prediction

**📊 Dataset**
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 1,400+ labeled speech samples
- 8 emotion classes:
  - Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

Dataset files are excluded from the repository for size and licensing reasons.

**🧠 Model & Approach**

**- Feature Extraction:**
  - 40-dimensional MFCC time-series features extracted from raw audio
**- Model:**
  - LSTM-based neural network to capture temporal speech patterns
**- Training Enhancements:**
  - Noise injection and pitch shifting (~2× effective data increase)
**- Evaluation Metrics:**
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix
**- Performance:**
  - Achieved ~80% multi-class accuracy on a held-out test set

**🌐 Web Application**

The trained model is deployed using Streamlit, allowing users to:
- Upload a .wav audio file
- Run emotion inference
- View predicted emotional state in near real time

**🛠️ Tech Stack**

**- Languages:** Python
**- ML / DSP:** TensorFlow (LSTM), Librosa, NumPy, Scikit-learn
**- Features:** MFCC
**- Visualization:** Matplotlib, Seaborn
**- Deployment:** Streamlit

**🚀 How to Run Locally**
# Create virtual environment (Python 3.10 recommended)
py -3.10 -m venv venv
venv\Scripts\activate

**# Install dependencies**
pip install -r requirements.txt

**# Train the model**
python train.py

**# Evaluate performance**
python evaluation.py

**# Launch web app**
streamlit run app.py

**📁 Project Structure**
Speech_Emotion_Recognition_RAVDESS_Enhanced/
├── train.py        # Model training
├── evaluation.py   # Confusion matrix & metrics
├── utils.py        # Feature extraction & augmentation
├── app.py          # Streamlit web app
├── requirements.txt
├── model/
└── data/           # (Excluded from repo)

**🧩 Engineering Highlights**

- Built a modular, maintainable ML codebase aligned with production workflows
- Handled real-world library compatibility issues (e.g., Librosa API changes)
- Applied data augmentation to address overfitting and improve generalization
- Focused on model interpretability, not just accuracy

**🔮 Future Improvements**

- Live microphone emotion detection
- CNN + LSTM hybrid architecture
- Cross-dataset generalization testing
- Containerized deployment (Docker)
