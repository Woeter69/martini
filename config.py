import os

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

# Ensure directories exist
for directory in [DATA_RAW_DIR, DATA_MIXED_DIR, OUTPUTS_STEMS_DIR, OUTPUTS_PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)
