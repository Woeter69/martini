# Martini — Incomplete Work Tracker

> Auto-generated assessment of remaining work based on `PROGRESS.md` roadmap vs. actual codebase.

---

## Phase 1: Foundation & Observability

### 1.3 Robust Logging
- [x] Replace all `print()` / `click.echo()` calls with a proper `logging` module
- [x] Add log levels (DEBUG for per-bin convergence, INFO for pipeline stages, WARNING for edge cases)
- [x] Log STFT bin convergence iterations and I/O status
- [x] Add log file output (e.g. `logs/run_<timestamp>.log`)

### 1.4 Custom Error Handling
- [x] Create custom exception classes (e.g. `UnderdeterminedError`, `StemLoadError`, `ConvergenceError`)
- [x] Raise `UnderdeterminedError` when `n_mixtures < n_sources`
- [x] Raise `ConvergenceError` when FastICA doesn't converge within `n_iter`
- [x] Add input validation across all modules (e.g. shape checks, NaN detection)

---

## Phase 2: Professional Audio Ingestion

### 2.1 Multi-Format Loader
- [x] Extend `loader.py` to support `.flac` and `.mp3` in addition to `.wav`
- [x] Auto-detect format from extension or file header
- [x] Add fallback/error message for unsupported formats

### 2.2 Channel Manager
- [x] Handle stereo input files (currently forces `mono=True`)
- [x] Implement Mono → Stereo upmixing option
- [x] Implement Stereo → Multi-Channel splitting
- [x] Allow user to choose channel handling strategy via CLI

### 2.3 Quality Preprocessor — Remaining
- [x] Add window function selection (Hann, Hamming, Blackman) via config or CLI
- [x] Expose STFT overlap ratio as a configurable parameter

---

## Phase 3: Mathematical Separation Core

### 3.2 FastICA Engine — Enhancements
- [x] Add alternative contrast function: `kurtosis` (currently only `tanh`)
- [x] Allow user to select contrast function via CLI flag
- [x] Add convergence logging (iteration count per component)

### 3.3 Bin-Wise Convergence Tracking
- [x] Track and log convergence status for each frequency bin independently
- [x] Report bins that failed to converge
- [x] Add a convergence summary (% of bins converged, avg iterations)

### 3.4 Permutation Solver (**Critical for frequency-domain mode**)
- [x] Implement inter-bin correlation method to align source ordering across frequency bins
- [x] Implement envelope-based tracking for permutation alignment
- [x] **Fix `main.py` line 65**: currently `Y_stft_reordered[b] = X_bin` discards the separated output — must use `S_bin_real` to reconstruct the complex separated signal

---

## Phase 4: Reconstruction & Recovery

### 4.2 Signal Normalization — Remaining
- [x] Apply peak normalization to separated output stems (currently only done for mixes in `mixer.py`)
- [x] Add DC-offset removal to `postprocessor.py`
- [x] Add optional loudness normalization (LUFS-based)

---

## Phase 5: Professional CLI & UX

### 5.1 Unified CLI — Remaining
- [x] Add subcommands: `mix`, `separate`, `info` (currently a single flat command)
- [x] `martini info` — print stem count, duration, sample rate of input files
- [x] `martini mix` — only run the mixing step
- [x] `martini separate` — run full pipeline
- [x] Add `--verbose` / `--quiet` flags tied to the logging system

### 5.2 Visualization Studio — Remaining
- [x] Improve spectrogram plots (add labeled axes, colorbar, source titles)
- [x] Add before/after spectrogram comparison for each stem
- [x] Add interactive mode or HTML report output (optional)

### 5.3 Documentation
- [x] Write a proper `README.md` explaining:
  - The Cocktail Party Problem and ICA theory
  - How to install and run the project
  - How to prepare stem data
  - Example outputs and expected results
- [x] Add docstrings to all public functions (some are present, some are missing)
- [x] Add inline comments explaining the math in `ica.py` more thoroughly

---

## Other / Cross-Cutting

- [x] **Data**: `data/raw/` is empty — add a download script or instructions to obtain 5-stem `.wav` files (e.g. from MUSDB18 or similar dataset)
- [x] **Testing**: No unit tests exist — add `tests/` with test cases for:
  - `ica.py` (synthetic signal recovery)
  - `mixer.py` (matrix conditioning)
  - `evaluate.py` (known SDR values)
- [x] **`requirements.txt`**: Missing `torch`-free — confirm all deps are listed (currently has `librosa`, `soundfile`, `mir_eval`, `click`, `matplotlib`, `numpy`)
- [x] **CI/CD**: No GitHub Actions or pre-commit hooks

---

## Priority Recommendation

| Priority | Task | Impact |
|----------|------|--------|
| ✅ Done | ~~Fix frequency-domain ICA (permutation solver + line 65 bug)~~ | Fixed in `ica.py` + `main.py` |
| 🔴 High | Add stem data or download script | Can't run without data |
| 🟡 Medium | Proper logging & error handling | Developer experience |
| 🟡 Medium | README & documentation | Usability |
| 🟢 Low | Multi-format loader, channel manager | Nice-to-have |
| 🟢 Low | CLI subcommands, advanced visualization | Polish |
