# Changelog

## [0.1.0] - 2026-04-01
### Added
- Implemented **FastICA from scratch** in `src/ica.py` (Centering, Whitening, Deflationary Fixed-point iteration with tanh contrast function).
- Created `src/loader.py` for audio loading and resampling using librosa.
- Created `src/mixer.py` for artificial mixing of N stems with a random mixing matrix.
- Created `src/preprocessor.py` for STFT computation and bin-wise data reordering.
- Created `src/postprocessor.py` for inverse STFT and saving separated audio files.
- Created `src/evaluate.py` for SDR/SIR/SAR quality metrics using mir_eval.
- Created `src/visualize.py` for waveform, spectrogram, and matrix heatmap generation.
- Created `main.py` CLI entrypoint with support for time-domain and frequency-domain ICA.
- Added `create_dummy_data.py` for pipeline testing with synthetic signals.
- Configured global parameters in `config.py`.
