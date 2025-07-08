import joblib
import librosa

import numpy as np

class PitchDetector():
    def __init__(self):
        self.NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def interpolate_peak(self, fft_vals, peak_idx):
        if 1 <= peak_idx < len(fft_vals) - 1:
            alpha = fft_vals[peak_idx - 1]
            beta = fft_vals[peak_idx]
            gamma = fft_vals[peak_idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            return peak_idx + p
        return peak_idx 

    def detect_pitch_from_fft(self, audio_chunk, rate):
        windowed = audio_chunk * np.hanning(len(audio_chunk))
        fft_vals = np.abs(np.fft.rfft(windowed))
        fft_vals = 20 * np.log10(fft_vals + 1e-10)
        freqs = np.fft.rfftfreq(len(audio_chunk), d=1 / rate)

        min_idx = np.argmax(freqs > 50)
        peak_idx = np.argmax(fft_vals[min_idx:]) + min_idx
        self.interp_idx = self.interpolate_peak(fft_vals, peak_idx)
        self.peak_freq = freqs[0] + self.interp_idx * (freqs[1] - freqs[0])
        
        return self.peak_freq
        
    def freq_to_note(self, freq):
        if freq <= 0:
            return "?"
        A4 = 440.0

        self.semitone_diff = int(round(12*np.log2(freq/A4)))
        self.note_index = (self.semitone_diff + 9) % 12
        self.octave = 4 + (self.semitone_diff + 9) // 12
        
        return f"{self.NOTE_NAMES[self.note_index]}"
    
class SoundDetector():
    def __init__(self):
        self.clf = joblib.load("sound_classifier.joblib")

    def extract_features(self, audio_data, sr=44100):
        if len(audio_data) < 2048 or np.max(np.abs(audio_data)) < 0.01:
            print("Too short or too silent — skipping pitch detection.")
            return None
        
        # Make sure audio is float and normalized
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data) + 1e-6)

        # Pad or trim to fixed length
        if len(audio_data) < 22050:
            audio_data = np.pad(audio_data, (0, 22050 - len(audio_data)))
        else:
            audio_data = audio_data[:22050]

        # Use same feature pipeline as training
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).mean(axis=1)
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y=audio_data).mean()
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr).mean(axis=1)
        rmse = librosa.feature.rms(y=audio_data).mean()

        if mfcc.size == 0 or np.isnan(mfcc).any():
            return None
        else:
            return np.hstack([mfcc, [centroid], [zcr], chroma, [rmse]])

    def predict_sound_type(self, audio_data):
        if audio_data is None or len(audio_data) == 0:
            print("Empty audio data.")
            return None

        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-6)
                                   
        if np.max(np.abs(audio_data)) < 0.01:
            print("Too silent — skipping pitch detection.")
            return None
        
        features = self.extract_features(audio_data)
        features = np.array(features, dtype=np.float32)
        if np.isnan(features).any():
            print("Features contain NaN — skipping prediction.")
            return None
        
        return self.clf.predict([features])[0]