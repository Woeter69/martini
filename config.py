import os
import logging
import time

# Audio configuration
SAMPLE_RATE = 22050
N_SOURCES = 5
STEM_NAMES = ['vocals', 'drums', 'bass', 'guitar', 'piano']

# STFT configuration
N_FFT = 2048
HOP_LENGTH = 512

# Path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_MIXED_DIR = os.path.join(BASE_DIR, 'data', 'mixed')
OUTPUTS_STEMS_DIR = os.path.join(BASE_DIR, 'outputs', 'stems')
OUTPUTS_PLOTS_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure directories exist
for directory in [DATA_RAW_DIR, DATA_MIXED_DIR, OUTPUTS_STEMS_DIR, OUTPUTS_PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

def setup_logging(level=logging.INFO):
    """Configure logging for the project."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
