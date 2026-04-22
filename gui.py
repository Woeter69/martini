import sys
import os
import logging
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QSpinBox, 
    QPlainTextEdit, QProgressBar, QTabWidget, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QPixmap

from src.pipeline import run_separation_pipeline
from src.mixer import mix_stems, save_mixes
from src.loader import load_stems, get_stem_matrix
from src.visualize import plot_matrix
from config import OUTPUTS_PLOTS_DIR

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
    log = Signal(str)

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
        
        plot_matrix(A_true, "True Mixing Matrix", 
                   save_path=os.path.join(OUTPUTS_PLOTS_DIR, "true_mixing_matrix.png"))
        
        self.finished.emit({"type": "mix", "plot": "true_mixing_matrix.png"})

class MartiniGUI(QMainWindow):
    log_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Martini — Music Source Separation")
        self.setMinimumSize(1000, 750)
        
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
        config_group = QGroupBox("Configuration")
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
        
        # Actions
        self.mix_btn = QPushButton("Step 1: Generate Mixes")
        self.mix_btn.setFixedHeight(40)
        self.mix_btn.clicked.connect(self.start_mix)
        left_layout.addWidget(self.mix_btn)
        
        self.separate_btn = QPushButton("Step 2: Run Separation")
        self.separate_btn.setFixedHeight(50)
        self.separate_btn.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; font-size: 14px;")
        self.separate_btn.clicked.connect(self.start_separation)
        left_layout.addWidget(self.separate_btn)
        
        left_layout.addStretch()
        
        # Progress
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # Right Panel: Results & Logs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        tabs = QTabWidget()
        
        # Tab 1: Visualizations
        self.viz_tab = QWidget()
        self.viz_layout = QVBoxLayout(self.viz_tab)
        self.image_label = QLabel("Run separation to see results")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        self.viz_layout.addWidget(self.image_label)
        
        # Mini control for switching plots
        plot_selector_layout = QHBoxLayout()
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(["separated_waveforms.png", "mixed_waveforms.png", "true_mixing_matrix.png", "est_mixing_matrix.png"])
        self.plot_combo.currentTextChanged.connect(self.update_plot)
        plot_selector_layout.addWidget(QLabel("View Plot:"))
        plot_selector_layout.addWidget(self.plot_combo)
        self.viz_layout.addLayout(plot_selector_layout)
        
        tabs.addTab(self.viz_tab, "Results Visualization")
        
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
            "seed": self.seed_input.value()
        }

    def start_mix(self):
        self.set_busy(True)
        self.worker = Worker("mix", self.get_params())
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
        self.separate_btn.setEnabled(not busy)
        if busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)

    def on_finished(self, result):
        self.set_busy(False)
        self.update_plot()
        if result["type"] == "separate":
            avg_sdr = result["results"].get("avg_sdr", 0)
            self.append_log(f"Pipeline finished successfully. Average SDR: {avg_sdr:.2f} dB")

    def on_error(self, message):
        self.set_busy(False)
        self.append_log(f"CRITICAL ERROR: {message}")

    def update_plot(self):
        plot_name = self.plot_combo.currentText()
        plot_path = os.path.join(OUTPUTS_PLOTS_DIR, plot_name)
        if os.path.exists(plot_path):
            pixmap = QPixmap(plot_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.setText(f"Plot '{plot_name}' not found yet.\nRun the process first.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_plot()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Apply a simple dark theme if desired, or just use default
    window = MartiniGUI()
    window.show()
    sys.exit(app.exec())
