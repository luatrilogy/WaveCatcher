import sys
import pyaudio

import numpy as np
import pyqtgraph as pg

from scipy.fft import fft
from PyQt5.QtCore import QTimer
from matplotlib.figure import Figure
from SoundDetection import PitchDetector
from scipy.ndimage import gaussian_filter1d
from scipy.signal import spectrogram, find_peaks
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel

# Harmonic Object
class HarmonicsCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor='black')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('black')
        super(HarmonicsCanvas, self).__init__(self.fig)
        self.setParent(parent)

        self.ax.set_title("Harmonics", color='lightblue')
        self.ax.set_xlabel("Frequency (Hz)", color='lightblue')
        self.ax.set_ylabel("Amplitude", color='lightblue')
        self.ax.tick_params(colors='lightblue')
        
        self.pitchdetector = PitchDetector()

    def plot_harmonics(self, freqs, amps):
        self.ax.clear()
        self.ax.set_facecolor('black')

        # Handle invalid data gracefully
        if not isinstance(freqs, (list, np.ndarray)) or not isinstance(amps, (list, np.ndarray)):
            self.ax.set_title("Invalid harmonic data", color='red')
            self.draw()
            return
        if len(freqs) == 0 or len(amps) == 0 or len(freqs) != len(amps):
            self.ax.set_title("No harmonics detected", color='red')
            self.draw()
            return

        # Plotting
        self.ax.bar(freqs, amps, width=10, color='red', edgecolor='white')
        for f, a in zip(freqs, amps):
            self.ax.text(f, a + 5, f"{self.pitchdetector.freq_to_note(f)}", ha='center', va='bottom', fontsize=8, color='white')

        self.ax.set_xlim(0, 1500)
        self.ax.set_ylim(0, 1000000)
        self.ax.set_title("Harmonics", color='lightblue')
        self.ax.set_xlabel("Frequency (Hz)", color='lightblue')
        self.ax.set_ylabel("Amplitude", color='lightblue')
        self.ax.tick_params(colors='lightblue')
        self.draw()

# AudioStream Object
class AudioStream(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

        self.PitchDetector = PitchDetector()
        self.last_pitch = None

        self.rms_histroy = []

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            output=True,
            frames_per_buffer=self.CHUNK
        )

        self.buffer_size = self.RATE * 2
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # Layout
        layout = QGridLayout()
        self.setLayout(layout)
        

        # Waveform plot
        self.waveform_plot = pg.PlotWidget(title="Waveform")
        self.waveform_plot.setYRange(-4000, 4000)
        self.waveform_curve = self.waveform_plot.plot(pen='y')
        layout.addWidget(self.waveform_plot, 0, 0)

        # FFT plot
        self.fft_plot = pg.PlotWidget(title="FFT Spectrum")
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.setYRange(0, (1000000))
        self.fft_curve = self.fft_plot.plot(pen='c')
        layout.addWidget(self.fft_plot, 0, 1)

        # FFT Harmonics
        self.harmonics_canvas = HarmonicsCanvas(self)
        layout.addWidget(self.harmonics_canvas, 0, 2) 

        # Spectrogram image
        self.spec_view = pg.ImageView()
        layout.addWidget(self.spec_view, 1, 1, 1, 2)
        self.spec_view.setPredefinedGradient("thermal")
        self.spec_view.view.setLimits(xMin=0, xMax=1000, yMin=0, yMax=1000)

        # Pitch Frequency Detection
        self.note_label = QLabel("Pitch: -")
        layout.addWidget(self.note_label, 2, 0, 1, 2)

        # Amplitude Envelope Plot
        self.envelope_plot = pg.PlotWidget(title="Amplitude Envelope")
        self.envelope_plot.setYRange(0, 200)
        self.envelope_plot.setLabel('left', 'Amplitude')
        self.envelope_curve = self.envelope_plot.plot(pen=pg.mkPen('magenta', width=2))
        layout.addWidget(self.envelope_plot, 1, 0)

        self.label_dynrange = QLabel("Dynamic Range: -- dB")
        self.label_maxamp = QLabel("Max Amplitude: --")
        self.label_dynrange.setStyleSheet("color: black; font-size: 12px;")
        self.label_maxamp.setStyleSheet("color: black; font-size: 12px;")

        layout.addWidget(self.label_dynrange, 7, 0, 1, 2)  # Replace row/col as needed
        layout.addWidget(self.label_maxamp, 8, 0, 1, 2)

        # Real-Time Spectral Centriod and Bandwidth
        self.label_spectral = QLabel("Spectral Centriod: -- Hz | Bandwidth: -- Hz")
        self.label_spectral.setStyleSheet("color:black; font-size: 12px;")
        layout.addWidget(self.label_spectral, 9, 0, 1, 2)

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def update(self):
        data = self.stream.read(self.CHUNK, exception_on_overflow=False)
        data_int = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self.stream.write(data_int.astype(np.float32).tobytes(), pyaudio.paContinue)

        # Waveform update
        self.waveform_curve.setData(data_int)

        # FFT update
        yf = np.abs(fft(data_int))[:self.CHUNK // 2]
        xf = np.fft.fftfreq(self.CHUNK, 1 / self.RATE)[:self.CHUNK // 2]
        self.fft_curve.setData(xf, yf)

        # Analyzing Harmonic Peaks from FFT
        freqs, amps = self.extract_harmonics(data_int, self.RATE)
        self.plot_harmonics(freqs, amps)

        # Spectrogram update
        self.audio_buffer = np.roll(self.audio_buffer, -len(data_int))
        self.audio_buffer[-len(data_int):] = data_int

        f, t, Sxx = spectrogram(self.audio_buffer, fs=self.RATE, nperseg=1024, noverlap=768)
        self.spec_view.setImage(Sxx, autoLevels=False, autoRange=False)

        # Envelope Calculations
        rms = np.sqrt(np.mean(data_int**2))
        self.rms_histroy = getattr(self, "rms_histroy", [])
        self.rms_histroy.append(rms)

        if len(self.rms_histroy) > 200:
            self.rms_histroy.pop(0)

        rms_smoothed = gaussian_filter1d(self.rms_histroy, sigma=2)
        normed = rms_smoothed / np.max(rms_smoothed) if np.max(rms_smoothed) > 0 else rms_smoothed
        normed *= 100

        max_amp = np.max(rms_smoothed)
        min_amp = np.min(rms_smoothed[rms_smoothed > 1e-4])
        dyn_range = 20 * np.log10(max_amp / min_amp) if min_amp > 0 else 0

        self.label_dynrange.setText(f"Dynamic Range: {dyn_range:.2f} dB")
        self.label_maxamp.setText(f"Max Amplitude: {max_amp:.2f}")

        self.envelope_curve.setData(normed)
        
        normalized_amp = max_amp / 32768.0 # for 16-bit audio

        if normalized_amp > 0.3:
            color = 'red'
        elif normalized_amp > 0.1:
            color = 'yellow'
        else:
            color = 'green'

        self.envelope_curve.setPen(pg.mkPen(color, width=2))

        # Pitch Detection 
        pitch_freq = self.PitchDetector.detect_pitch_from_fft(data_int, self.RATE)
        note = self.PitchDetector.freq_to_note(pitch_freq)

        if len(rms_smoothed) > 1 and np.min(rms_smoothed) > 0:
            dyn_range = 20 * np.log10(np.max(rms_smoothed) / np.min(rms_smoothed))
            dyn_text = f" | DynRange: {dyn_range:.1f} dB"
        else:
            dyn_text = " | DynRange: N/A"

        self.note_label.setText(f"Pitch: {note} ({pitch_freq:.1f} Hz){dyn_text}")

        alpha = 0.3

        if self.last_pitch is None:
            self.last_pitch = pitch_freq
        else:
            self.last_pitch = alpha * pitch_freq + (1 - alpha) * self.last_pitch

        # Real-Time Spectral Centriod and Bandwidth
        if np.sum(yf) > 0:
            spectral_centroid = np.sum(xf * yf) / np.sum(yf)
            bandwidth = np.sqrt(np.sum(((xf - spectral_centroid) ** 2) * yf) / np.sum(yf))
        else:
            spectral_centroid = 0
            bandwidth = 0

        self.label_spectral.setText(
            f"Spectral Centroid: {spectral_centroid:.1f} Hz | Bandwidth: {bandwidth:.1f} Hz"
        )

        if spectral_centroid > 3000:
            color = "red"
        elif spectral_centroid > 1000:
            color = "orange"
        else:
            color = "green"

        self.label_spectral.setStyleSheet(f"color: {color}; font-size: 12px;")

    def get_fundamental_frequency(self, data, rate):
    # Apply FFT
        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1 / rate)

        # Avoid silent or NaN inputs
        if np.max(fft_vals) < 1e-3:
            return None

        # Find the index of the maximum (ignoring the first bin)
        peak_idx = np.argmax(fft_vals[1:]) + 1
        f0 = freqs[peak_idx]

        if f0 < 20 or f0 > rate // 2:
            return None

        return f0
    
    def extract_harmonics(self, data, rate, num_harmonics=5):
        f0 = self.get_fundamental_frequency(data, rate)
        if not f0:
            return [], []

        fft_vals = np.abs(np.fft.rfft(data))
        freqs = np.fft.rfftfreq(len(data), 1 / rate)

        harmonics_freqs = []
        harmonics_amps = []

        for n in range(1, num_harmonics + 1):
            harmonic_freq = f0 * n
            idx = np.argmin(np.abs(freqs - harmonic_freq))
            harmonics_freqs.append(freqs[idx])
            harmonics_amps.append(fft_vals[idx])

        return harmonics_freqs, harmonics_amps
    
    def plot_harmonics(self, freqs, amps):
        self.harmonics_canvas.plot_harmonics(freqs, amps)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Audio Spectrum Analyzer")
        self.setGeometry(100, 100, 1000, 800)
        self.stream_widget = AudioStream(self)
        self.setCentralWidget(self.stream_widget)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
