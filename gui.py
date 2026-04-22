import sys
import os
import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, 
    QPlainTextEdit, QProgressBar, QTabWidget, QGroupBox, QFormLayout,
    QFileDialog, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from src.pipeline import run_separation_pipeline, run_ica_on_mixtures
from src.mixer import mix_stems, save_mixes
from src.loader import load_stems, get_stem_matrix
from config import OUTPUTS_PLOTS_DIR, STEM_NAMES, SAMPLE_RATE, OUTPUTS_STEMS_DIR, DATA_MIXED_DIR

class LogHandler(logging.Handler):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal

    def emit(self, record):
        msg = self.format(record)
        self.signal.emit(msg)

class Worker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params

    def run(self):
        try:
            if self.task_type == "mix":
                self.run_mix()
            elif self.task_type == "separate":
                results = run_separation_pipeline(**self.params)
                self.finished.emit({"type": "separate", "results": results})
            elif self.task_type == "ica_only":
                self.run_ica_only()
        except Exception as e:
            self.error.emit(str(e))

    def run_mix(self):
        duration = self.params.get('duration', 10.0)
        seed = self.params.get('seed', 42)
        channels = self.params.get('channels', 'mono')

        stems_dict = load_stems(duration=duration, channel_strategy=channels)
        S_true = get_stem_matrix(stems_dict)
        X, A_true = mix_stems(S_true, seed=seed)
        save_mixes(X)
        
        self.finished.emit({"type": "mix", "X": X, "A_true": A_true, "S_true": S_true})

    def run_ica_only(self):
        # This assumes X is already in params (from a previous mix)
        X = self.params.get('X')
        if X is None:
            raise ValueError("No mixtures found. Run Step 1 first.")
        
        S_est, W_est = run_ica_on_mixtures(
            X, 
            mode=self.params.get('mode', 'time'),
            contrast=self.params.get('contrast', 'tanh')
        )
        self.finished.emit({"type": "ica_only", "S_est": S_est, "W_est": W_est})

class StemPlayer(QFrame):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.name_label = QLabel(f"<b>{name.capitalize()}:</b>")
        self.name_label.setFixedWidth(80)
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setFixedWidth(80)
        self.play_btn.clicked.connect(self.toggle_play)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray;")
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.8)
        
        self.file_path = None
        self.player.playbackStateChanged.connect(self.on_state_changed)

    def set_file(self, path):
        self.file_path = path
        if os.path.exists(path):
            self.player.setSource(QUrl.fromLocalFile(path))
            self.status_label.setText("Loaded")
            self.status_label.setStyleSheet("color: green;")
            self.play_btn.setEnabled(True)
        else:
            self.status_label.setText("Not found")
            self.status_label.setStyleSheet("color: red;")
            self.play_btn.setEnabled(False)

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def on_state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("⏸ Pause")
        else:
            self.play_btn.setText("▶ Play")

class MartiniGUI(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Martini — Music Source Separation")
        self.setMinimumSize(1200, 850)
        
        self.current_X = None
        self.current_S_true = None
        
        self.setup_ui()
        self.setup_logging()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel: Controls
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        
        # Parameters Group
        config_group = QGroupBox("1. Configuration")
        config_form = QFormLayout(config_group)
        
        self.duration_input = QDoubleSpinBox()
        self.duration_input.setRange(1.0, 60.0)
        self.duration_input.setValue(10.0)
        config_form.addRow("Duration (s):", self.duration_input)
        
        self.mode_input = QComboBox()
        self.mode_input.addItems(["time", "frequency"])
        config_form.addRow("ICA Mode:", self.mode_input)
        
        self.contrast_input = QComboBox()
        self.contrast_input.addItems(["tanh", "kurtosis"])
        config_form.addRow("Contrast:", self.contrast_input)
        
        self.seed_input = QSpinBox()
        self.seed_input.setRange(0, 9999)
        self.seed_input.setValue(42)
        config_form.addRow("Random Seed:", self.seed_input)
        
        left_layout.addWidget(config_group)
        
        # Actions Group
        action_group = QGroupBox("2. Pipeline Steps")
        action_layout = QVBoxLayout(action_group)
        
        self.load_mix_btn = QPushButton("📂 Load External Mixes")
        self.load_mix_btn.setFixedHeight(30)
        self.load_mix_btn.clicked.connect(self.load_external_mixes)
        action_layout.addWidget(self.load_mix_btn)
        
        self.mix_btn = QPushButton("Step 1: Generate Mixes")
        self.mix_btn.setFixedHeight(40)
        self.mix_btn.clicked.connect(self.start_mix)
        action_layout.addWidget(self.mix_btn)
        
        self.ica_btn = QPushButton("Step 2: Run ICA")
        self.ica_btn.setFixedHeight(40)
        self.ica_btn.setStyleSheet("background-color: #34495e; color: white;")
        self.ica_btn.clicked.connect(self.start_ica_only)
        action_layout.addWidget(self.ica_btn)
        
        self.separate_btn = QPushButton("Full Pipeline (Auto)")
        self.separate_btn.setFixedHeight(50)
        self.separate_btn.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        self.separate_btn.clicked.connect(self.start_separation)
        action_layout.addWidget(self.separate_btn)
        
        left_layout.addWidget(action_group)

        # Audio Players Group
        player_group = QGroupBox("3. Results (Audio)")
        player_layout = QVBoxLayout(player_group)
        
        self.players = {}
        # Mixes players
        player_layout.addWidget(QLabel("<b>Mixtures:</b>"))
        for i in range(2): # Show first 2 mixes
            p = StemPlayer(f"Mix {i+1}")
            self.players[f"mix_{i}"] = p
            player_layout.addWidget(p)
            
        player_layout.addWidget(QLabel("<b>Separated Stems:</b>"))
        for name in STEM_NAMES:
            p = StemPlayer(name)
            self.players[name] = p
            player_layout.addWidget(p)
            
        left_layout.addWidget(player_group)
        left_layout.addStretch()
        
        # Progress
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Right Panel: Results & Logs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        tabs = QTabWidget()
        
        # Tab 1: Interactive Visualization
        self.viz_tab = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_tab)
        
        self.web_view = QWebEngineView()
        self.web_view.setHtml("<html><body style='background-color:#f0f0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:sans-serif;'><h2>Run Step 1 or 2 to see interactive graphs</h2></body></html>")
        self.viz_layout.addWidget(self.web_view)
        
        tabs.addTab(self.viz_tab, "Interactive Visualizer")
        
        # Tab 2: Logs
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace; background-color: #1e1e1e; color: #d4d4d4; font-size: 12px;")
        tabs.addTab(self.log_output, "Execution Logs")
        
        right_layout.addWidget(tabs)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def setup_logging(self):
        self.log_signal.connect(self.append_log)
        handler = LogHandler(self.log_signal)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    @Slot(str)
    def append_log(self, text):
        self.log_output.appendPlainText(text)
        self.log_output.ensureCursorVisible()

    def get_params(self):
        return {
            "duration": self.duration_input.value(),
            "mode": self.mode_input.currentText(),
            "contrast": self.contrast_input.currentText(),
            "seed": self.seed_input.value(),
            "X": self.current_X
        }

    def start_mix(self):
        self.set_busy(True)
        self.worker = Worker("mix", self.get_params())
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def start_ica_only(self):
        if self.current_X is None:
            self.append_log("ERROR: No mixtures loaded. Please run Step 1 first.")
            return
        self.set_busy(True)
        self.worker = Worker("ica_only", self.get_params())
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def start_separation(self):
        self.set_busy(True)
        self.worker = Worker("separate", self.get_params())
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def set_busy(self, busy):
        self.mix_btn.setEnabled(not busy)
        self.ica_btn.setEnabled(not busy)
        self.separate_btn.setEnabled(not busy)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

    def on_finished(self, result):
        self.set_busy(False)
        
        if result["type"] == "mix":
            self.current_X = result["X"]
            self.current_S_true = result["S_true"]
            self.append_log("Step 1: Mixtures generated successfully.")
            self.update_visuals(self.current_X, "Input Mixtures")
            self.update_players(mode="mix")
            
        elif result["type"] == "ica_only":
            S_est = result["S_est"]
            self.append_log("Step 2: ICA separation complete.")
            self.update_visuals(S_est, "Separated Stems (ICA)")
            self.update_players(mode="separate")
            
        elif result["type"] == "separate":
            res = result["results"]
            self.current_X = res["X"]
            S_est = res["S_est"]
            avg_sdr = res.get("avg_sdr", 0)
            self.append_log(f"Pipeline finished. Average SDR: {avg_sdr:.2f} dB")
            self.update_visuals(S_est, "Separated Stems (Full Pipeline)")
            self.update_players(mode="separate")

    def on_error(self, message):
        self.set_busy(False)
        self.append_log(f"CRITICAL ERROR: {message}")

    def update_visuals(self, signals, title):
        """Creates an interactive Plotly waveform plot."""
        n_sources, n_samples = signals.shape
        time = np.linspace(0, n_samples / SAMPLE_RATE, n_samples)
        
        # Downsample for performance in web view
        ds_factor = max(1, n_samples // 5000)
        time_ds = time[::ds_factor]
        
        fig = make_subplots(rows=n_sources, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05,
                           subplot_titles=[f"Source {i+1}" for i in range(n_sources)])
        
        for i in range(n_sources):
            fig.add_trace(
                go.Scatter(x=time_ds, y=signals[i, ::ds_factor], name=f"S{i+1}",
                          line=dict(width=1)),
                row=i+1, col=1
            )
            
        fig.update_layout(height=150 * n_sources + 100, title_text=title,
                          showlegend=False, template="plotly_white")
        fig.update_xaxes(title_text="Time (s)", row=n_sources, col=1)
        
        html = fig.to_html(include_plotlyjs='cdn')
        self.web_view.setHtml(html)

    def update_players(self, mode="mix"):
        if mode == "mix":
            for i in range(2):
                path = os.path.join(DATA_MIXED_DIR, f"mix_{i}.wav")
                self.players[f"mix_{i}"].set_file(path)
        else:
            for name in STEM_NAMES:
                path = os.path.join(OUTPUTS_STEMS_DIR, f"{name}.wav")
                self.players[name].set_file(path)

    def load_external_mixes(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Mixes", DATA_MIXED_DIR, "Audio (*.wav *.flac *.mp3)")
        if file_paths:
            try:
                import librosa
                mixes = []
                for p in file_paths:
                    y, _ = librosa.load(p, sr=SAMPLE_RATE, duration=self.duration_input.value())
                    mixes.append(y)
                self.current_X = np.stack(mixes)
                self.append_log(f"Loaded {len(self.current_X)} external mixtures.")
                self.update_visuals(self.current_X, "External Mixtures")
                # Reset players since external files might not be in our naming convention
                for p in self.players.values():
                    p.play_btn.setEnabled(False)
            except Exception as e:
                self.append_log(f"Error loading mixes: {e}")

if __name__ == "__main__":
    # Fix for WebEngine segmentation faults in some Linux/WSL environments
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
    
    # Add Chromium arguments to disable GPU if necessary
    sys.argv.append("--disable-gpu")
    sys.argv.append("--no-sandbox")
    sys.argv.append("--disable-software-rasterizer")
    sys.argv.append("--disable-dev-shm-usage")

    app = QApplication(sys.argv)
    window = MartiniGUI()
    window.show()
    sys.exit(app.exec())
