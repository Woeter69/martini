# Martini — Blind Source Separation for Music

Martini is a **professional-grade signal separation tool** designed to extract isolated audio sources (vocals, drums, bass, etc.) from mixtures using **Independent Component Analysis (ICA)** from scratch.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🍸 The Cocktail Party Problem
Imagine you are at a crowded party. Multiple people are talking simultaneously, and music is playing in the background. Your brain can effortlessly focus on a single voice, effectively "separating" it from the noise. This is the **Cocktail Party Problem**.

In digital signal processing, we model this as:
$$X = AS$$
- **$X$**: The recorded mixtures (e.g., two microphones in a room).
- **$A$**: The mixing matrix (how much of each source reaches each microphone).
- **$S$**: The isolated independent sources (the actual voices).

Martini uses the **FastICA algorithm** to recover $S$ by finding an unmixing matrix $W \approx A^{-1}$ that maximizes the statistical independence of the output signals.

---

## 🚀 Key Features
- **FastICA Engine:** Built from scratch with `tanh` and `kurtosis` contrast functions.
- **Dual-Domain Separation:** Process signals in the **Time Domain** or **Frequency Domain** (STFT bin-wise ICA).
- **Permutation Alignment:** Solves the frequency-domain permutation problem using inter-bin envelope correlation.
- **Robust Pipeline:** Includes multi-format loading, mixing simulation, and evaluation using `mir_eval`.
- **Professional CLI:** Unified command-line interface with `info`, `mix`, and `separate` subcommands.

---

## 🛠 Installation

### 1. Clone the repository
```bash
git clone https://github.com/Woeter69/martini.git
cd martini
```

### 2. Set up the environment
Martini uses `uv` for lightning-fast dependency management, but you can also use `pip`:
```bash
pip install -r requirements.txt
```

---

## 🎹 Usage

### 1. Prepare Stem Data
Martini expects 5 isolated stems in `data/raw/` (vocals, drums, bass, guitar, piano).

**Option A: Generate Synthetic Test Data**
If you don't have stems, generate synthetic signals to test the pipeline:
```bash
python downloads/fetch_tracks.py --synthetic
```

**Option B: Use Real-World Data**
Place your `.wav`, `.flac`, or `.mp3` files in `data/raw/` and check them:
```bash
python main.py info
```

### 2. Simulate Mixing
Generate $N$ mixtures from your isolated stems:
```bash
python main.py mix --duration 10.0 --seed 42
```
This saves the mixtures to `data/mixed/` and plots the mixing matrix.

### 3. Run Separation
Extract the sources from the mixtures:
```bash
# Time-domain separation
python main.py separate --mode time --duration 10.0

# Frequency-domain (STFT) separation
python main.py separate --mode frequency --duration 5.0
```
Check the `outputs/stems/` folder for the separated tracks and `outputs/plots/` for visual results.

---

## 📊 Evaluation
Martini uses the industry-standard `mir_eval` library to calculate:
- **SDR:** Signal-to-Distortion Ratio.
- **SIR:** Signal-to-Interference Ratio.
- **SAR:** Signal-to-Artifact Ratio.

Higher values (in dB) indicate better separation quality.

---

## 📂 Project Structure
```text
martini/
├── src/
│   ├── ica.py            # FastICA implementation & Permutation Solver
│   ├── preprocessor.py   # STFT and spectral handling
│   ├── loader.py         # Multi-format audio ingestion
│   ├── mixer.py          # Synthetic mixing matrix generation
│   ├── postprocessor.py  # Reconstruction & Normalization
│   └── evaluate.py       # mir_eval integration
├── main.py               # Unified CLI Entry point
└── data/
    ├── raw/              # Isolated input stems
    └── mixed/            # Generated mixtures
```

---

## 📜 License
This project is licensed under the MIT License.
