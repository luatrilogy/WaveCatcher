import os
import joblib
import librosa

import numpy as np

from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_combined_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    # Normalize audio
    y = y / np.max(np.abs(y) + 1e-6)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean(axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y).mean(axis=1)

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)

    # Root Mean Square Energy
    rmse = librosa.feature.rms(y=y).mean(axis=1)

    # Concatenate all features into one 1D array
    return np.hstack([mfcc_mean, spec_centroid, zcr, chroma, rmse])

X, y = [], []
labels = ['clap', 'whistling']

for label in labels:
    folder = f"audio/{label}"
    for file in os.listdir(folder):
        if not file.endswith(".wav"): continue
        path = os.path.join(folder, file)
        features = extract_combined_features(path)  # or extract_log_spectrogram(path)
        X.append(features)
        y.append(label)

X = np.array(X)
y= np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(classification_report(y_test, clf.predict(X_test)))
joblib.dump(clf, "sound_classifier.joblib")