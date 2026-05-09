import sys
import os
import logging
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, 
    QPlainTextEdit, QProgressBar, QTabWidget, QGroupBox, QFormLayout,
    QFileDialog, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from src.pipeline import run_separation_pipeline, run_ica_on_mixtures
from src.mixer import mix_stems, save_mixes
from src.loader import load_stems, get_stem_matrix, load_stereo_as_mixtures
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
        X = self.params.get('X')
        if X is None:
            raise ValueError("No mixtures found. Run Step 1 first.")
        
        S_est, W_est = run_ica_on_mixtures(
            X, 
            mode=self.params.get('mode', 'time'),
            contrast=self.params.get('contrast', 'tanh'),
            n_fft=self.params.get('n_fft', 2048),
            normalize=self.params.get('normalize', 'peak')
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

        self.norm_input = QComboBox()
        self.norm_input.addItems(["peak", "lufs", "none"])
        config_form.addRow("Normalize:", self.norm_input)

        self.n_fft_input = QComboBox()
        self.n_fft_input.addItems(["1024", "2048", "4096", "8192"])
        self.n_fft_input.setCurrentText("2048")
        config_form.addRow("Resolution (FFT):", self.n_fft_input)
        
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
        for i in range(2):
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
        
        # Tab 1: Interactive Waveforms
        self.viz_tab = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_tab)
        self.web_view = QWebEngineView()
        self.web_view.setHtml("<html><body style='background-color:#f0f0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:sans-serif;'><h2>Run Step 1 or 2 to see interactive waveforms</h2></body></html>")
        self.viz_layout.addWidget(self.web_view)
        tabs.addTab(self.viz_tab, "Waveforms")

        # Tab 2: Interactive Spectrograms
        self.spec_tab = QWidget()
        self.spec_layout = QVBoxLayout(self.spec_tab)
        self.spec_web_view = QWebEngineView()
        self.spec_web_view.setHtml("<html><body style='background-color:#f0f0f0; display:flex; justify-content:center; align-items:center; height:100vh; font-family:sans-serif;'><h2>Run Step 1 or 2 to see spectrograms</h2></body></html>")
        self.spec_layout.addWidget(self.spec_web_view)
        tabs.addTab(self.spec_tab, "Spectrograms")

        # Tab 3: Quality Metrics
        self.metrics_tab = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_tab)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Stem", "SDR (dB)", "SIR (dB)", "SAR (dB)"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        self.metrics_layout.addWidget(QLabel("<h3>Separation Quality (mir_eval)</h3>"))
        self.metrics_layout.addWidget(QLabel("<i>Higher values indicate better separation quality.</i>"))
        self.metrics_layout.addWidget(self.metrics_table)
        tabs.addTab(self.metrics_tab, "Quality Metrics")
        
        # Tab 4: Logs
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
            "normalize": self.norm_input.currentText(),
            "n_fft": int(self.n_fft_input.currentText()),
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
            self.update_spectrograms(self.current_X, "Input Mixtures (Spectrograms)")
            self.update_players(mode="mix")
            
        elif result["type"] in ["ica_only", "separate"]:
            if result["type"] == "ica_only":
                S_est = result["S_est"]
                self.append_log("Step 2: ICA separation complete.")
            else:
                res = result["results"]
                self.current_X = res["X"]
                S_est = res["S_est"]
                self.current_S_true = res["S_true"]
                avg_sdr = res.get("avg_sdr", 0)
                self.append_log(f"Pipeline finished. Average SDR: {avg_sdr:.2f} dB")

            self.update_visuals(S_est, "Separated Stems")
            self.update_spectrograms(S_est, "Separated Stems (Spectrograms)")
            
            if self.current_S_true is not None:
                from src.evaluate import evaluate_separation
                sdr, sir, sar, _ = evaluate_separation(self.current_S_true, S_est)
                self.update_metrics(sdr, sir, sar)
            
            self.update_players(mode="separate")

    def on_error(self, message):
        self.set_busy(False)
        self.append_log(f"CRITICAL ERROR: {message}")

    def update_metrics(self, sdr, sir, sar):
        """Populates the Quality Metrics table."""
        self.metrics_table.setRowCount(len(sdr) + 1)
        for i in range(len(sdr)):
            name = STEM_NAMES[i] if i < len(STEM_NAMES) else f"Source {i+1}"
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name.capitalize()))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{sdr[i]:.2f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem(f"{sir[i]:.2f}"))
            self.metrics_table.setItem(i, 3, QTableWidgetItem(f"{sar[i]:.2f}"))
        row = len(sdr)
        self.metrics_table.setItem(row, 0, QTableWidgetItem("AVERAGE"))
        self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{np.mean(sdr):.2f}"))
        self.metrics_table.setItem(row, 2, QTableWidgetItem(f"{np.mean(sir):.2f}"))
        self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{np.mean(sar):.2f}"))
        for col in range(4):
            item = self.metrics_table.item(row, col)
            if item: item.setBackground(Qt.lightGray)

    def update_visuals(self, signals, title):
        """Creates an interactive Plotly waveform plot."""
        n_sources, n_samples = signals.shape
        time = np.linspace(0, n_samples / SAMPLE_RATE, n_samples)
        ds_factor = max(1, n_samples // 5000)
        time_ds = time[::ds_factor]
        fig = make_subplots(rows=n_sources, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for i in range(n_sources):
            fig.add_trace(go.Scatter(x=time_ds, y=signals[i, ::ds_factor], name=f"S{i+1}", line=dict(width=1)), row=i+1, col=1)
        fig.update_layout(height=150 * n_sources + 100, title_text=title, showlegend=False, template="plotly_white")
        html = fig.to_html(include_plotlyjs='cdn')
        self.web_view.setHtml(html)

    def update_spectrograms(self, signals, title):
        """Creates interactive Plotly spectrogram plots."""
        import librosa
        n_sources = signals.shape[0]
        fig = make_subplots(rows=n_sources, cols=1, shared_xaxes=True, vertical_spacing=0.05)
        for i in range(n_sources):
            D = librosa.stft(signals[i], n_fft=1024, hop_length=512)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            S_db_small = S_db[:512:4, ::4]
            times = np.linspace(0, len(signals[i]) / SAMPLE_RATE, S_db_small.shape[1])
            freqs = np.linspace(0, SAMPLE_RATE / 2, S_db.shape[0])[:512:4]
            fig.add_trace(go.Heatmap(z=S_db_small, x=times, y=freqs, colorscale='Magma', showscale=(i == 0)), row=i+1, col=1)
        fig.update_layout(height=200 * n_sources + 100, title_text=title, template="plotly_white")
        html = fig.to_html(include_plotlyjs='cdn')
        self.spec_web_view.setHtml(html)

    def update_players(self, mode="mix"):
        if mode == "mix":
            for i in range(2):
                path = os.path.join(DATA_MIXED_DIR, f"mix_{i}.wav")
                self.players[f"mix_{i}"].set_file(path)
        else:
            for name in STEM_NAMES:
                path = os.path.join(OUTPUTS_STEMS_DIR, f"separated_{name}.wav")
                self.players[name].set_file(path)

    def load_external_mixes(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Mixes", DATA_MIXED_DIR, "Audio (*.wav *.flac *.mp3)")
        if not file_paths: return
        try:
            if len(file_paths) == 1:
                reply = QMessageBox.question(self, 'Stereo Mode', "Treat as Stereo Mixture?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.current_X = load_stereo_as_mixtures(file_paths[0], duration=self.duration_input.value())
                    self.append_log(f"Loaded Stereo: {file_paths[0]}")
                else:
                    import librosa
                    y, _ = librosa.load(file_paths[0], sr=SAMPLE_RATE, duration=self.duration_input.value())
                    self.current_X = y[np.newaxis, :]
                    self.append_log(f"Loaded Mono: {file_paths[0]}")
            else:
                import librosa
                mixes = [librosa.load(p, sr=SAMPLE_RATE, duration=self.duration_input.value())[0] for p in file_paths]
                self.current_X = np.stack(mixes)
                self.append_log(f"Loaded {len(self.current_X)} mixtures.")
            self.update_visuals(self.current_X, "External Mixtures")
            self.update_spectrograms(self.current_X, "External Mixtures (Spectrograms)")
            for p in self.players.values(): p.play_btn.setEnabled(False); p.status_label.setText("N/A")
        except Exception as e: self.append_log(f"Error loading mixes: {e}")

if __name__ == "__main__":
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
    sys.argv.append("--disable-gpu")
    sys.argv.append("--no-sandbox")
    app = QApplication(sys.argv)
    window = MartiniGUI()
    window.show()
    sys.exit(app.exec())
