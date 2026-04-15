# Progress Tracker: Real-World Voice Separation via FastICA

This document tracks the architectural evolution of the Martini project. We are building a **professional-grade signal separation tool** to extract specific audio sources (e.g., voice from background) from real-world recordings.

---

## 🚀 Project Vision
Implement a robust, non-AI (FastICA from scratch) pipeline to separate $N$ audio sources from $M$ mixtures (where $M \geq N$). We focus on the mathematical extraction of independent components from real-world `.wav` or `.mp3` files.

---

## 🛠 Phase 1: Foundation & Observability
- [x] **1.1 Singleton Configuration:** A central `config.py` for path resolution, sample rates, and environment validation.
- [x] **1.2 Directory Sentinel:** Automatic workspace setup (`data/raw`, `outputs/separated`, `logs/`).
- [x] **1.3 Robust Logging:** A production-style logging system to track STFT bin convergence and I/O status.
- [x] **1.4 Error Handling:** Custom exceptions for "Underdetermined" cases (too few mixtures for the number of sources).

## 🎛 Phase 2: Professional Audio Ingestion
- [x] **2.1 Multi-Format Loader:** Support for `wav`, `flac`, and `mp3` using `librosa`/`soundfile`.
- [x] **2.2 Channel Manager:** Handle Mono-to-Stereo upmixing or Stereo-to-Multi-Channel splitting.
- [x] **2.3 Quality Preprocessor:** Implement STFT with adjustable windowing and overlap (Hann/Hamming) for frequency-domain ICA.

## 🧮 Phase 3: Mathematical Separation Core
- [x] **3.1 Data Whitening:** PCA-based preprocessing to decorrelate signals and unit-variance normalization.
- [x] **3.2 FastICA Engine:** Robust fixed-point iteration with `tanh` and `kurtosis` contrast functions.
- [x] **3.3 Bin-Wise Convergence:** For frequency-domain ICA, track the convergence of every frequency bin independently.
- [x] **3.4 Permutation Solver:** Use inter-bin correlation or envelope tracking to solve the "Permutation Problem" (making sure frequency bin 100 and bin 101 belong to the same source).

## 🔊 Phase 4: Reconstruction & Recovery
- [x] **4.1 Inverse Signal Path:** High-fidelity iSTFT to reconstruct time-domain waveforms.
- [x] **4.2 Signal Normalization:** Professional-grade peak normalization and DC-offset removal.
- [x] **4.3 Evaluation & BSS Metrics:** Use `mir_eval` (SDR/SIR/SAR) to mathematically prove separation quality.

## 📊 Phase 5: Professional CLI & UX
- [x] **5.1 Unified CLI:** A `click`-based command-line interface with subcommands (`mix`, `separate`, `info`).
- [x] **5.2 Visualization Studio:** Waveform comparison, spectrograms, and mixing matrix heatmaps.
- [x] **5.3 Documentation:** Detailed `README.md` explaining the ICA "Cocktail Party" theory and how to use the tool.

---

## 📅 Current Status
- [x] Initial Project Scaffolding
- [x] Development Environment Setup (`uv`)
- [x] Phase 1 (Foundation) Complete
- [x] Phase 2 (Audio Ingestion) Complete
- [x] Phase 3 (Core) Complete
- [x] Phase 4 (Normalization & Recovery) Complete
- [x] Phase 5 (CLI & Documentation) Complete
- [x] Automated Testing Suite & CI setup
