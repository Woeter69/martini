# Martini — Incomplete Work Tracker

> Auto-generated assessment of remaining work based on `PROGRESS.md` roadmap vs. actual codebase.

---

## Phase 1: Foundation & Observability

### 1.3 Robust Logging
- [ ] Replace all `print()` / `click.echo()` calls with a proper `logging` module
- [ ] Add log levels (DEBUG for per-bin convergence, INFO for pipeline stages, WARNING for edge cases)
- [ ] Log STFT bin convergence iterations and I/O status
- [ ] Add log file output (e.g. `logs/run_<timestamp>.log`)

### 1.4 Custom Error Handling
- [ ] Create custom exception classes (e.g. `UnderdeterminedError`, `StemLoadError`, `ConvergenceError`)
- [ ] Raise `UnderdeterminedError` when `n_mixtures < n_sources`
- [ ] Raise `ConvergenceError` when FastICA doesn't converge within `n_iter`
- [ ] Add input validation across all modules (e.g. shape checks, NaN detection)

---

## Phase 2: Professional Audio Ingestion

### 2.1 Multi-Format Loader
- [ ] Extend `loader.py` to support `.flac` and `.mp3` in addition to `.wav`
- [ ] Auto-detect format from extension or file header
- [ ] Add fallback/error message for unsupported formats

### 2.2 Channel Manager
- [ ] Handle stereo input files (currently forces `mono=True`)
- [ ] Implement Mono → Stereo upmixing option
- [ ] Implement Stereo → Multi-Channel splitting
- [ ] Allow user to choose channel handling strategy via CLI

### 2.3 Quality Preprocessor — Remaining
- [ ] Add window function selection (Hann, Hamming, Blackman) via config or CLI
- [ ] Expose STFT overlap ratio as a configurable parameter

---

## Phase 3: Mathematical Separation Core

### 3.2 FastICA Engine — Enhancements
- [x] ~~Add alternative contrast function: `kurtosis`~~ → `g_kurtosis` / `g_prime_kurtosis` in `ica.py` with `CONTRAST_FUNCTIONS` registry
- [x] ~~Allow user to select contrast function via CLI flag~~ → `--contrast tanh|kurtosis` in `main.py`
- [x] ~~Add convergence logging (iteration count per component)~~ → `fast_ica` now returns `convergence_info` list; printed in time-mode output

### 3.3 Bin-Wise Convergence Tracking
- [x] ~~Track and log convergence status for each frequency bin independently~~ → `bin_convergence` list collected per bin in `main.py`
- [x] ~~Report bins that failed to converge~~ → failed bin indices printed (first 10)
- [x] ~~Add a convergence summary (% of bins converged, avg iterations)~~ → prints fully-converged %, avg iterations, max iterations

### 3.4 Permutation Solver (**Critical for frequency-domain mode**)
- [x] ~~Implement inter-bin correlation method to align source ordering across frequency bins~~ → `solve_permutation()` in `ica.py` using Hungarian algorithm
- [x] ~~Implement envelope-based tracking for permutation alignment~~ → uses inter-bin envelope correlation
- [x] ~~**Fix `main.py` line 65**: discarded separated output~~ → now applies `W_bin @ X_bin` to complex STFT + added missing `reconstruct_stft_from_ica` transpose

---

## Phase 4: Reconstruction & Recovery

### 4.2 Signal Normalization — Remaining
- [ ] Apply peak normalization to separated output stems (currently only done for mixes in `mixer.py`)
- [ ] Add DC-offset removal to `postprocessor.py`
- [ ] Add optional loudness normalization (LUFS-based)

---

## Phase 5: Professional CLI & UX

### 5.1 Unified CLI — Remaining
- [ ] Add subcommands: `mix`, `separate`, `info` (currently a single flat command)
- [ ] `martini info` — print stem count, duration, sample rate of input files
- [ ] `martini mix` — only run the mixing step
- [ ] `martini separate` — run full pipeline
- [ ] Add `--verbose` / `--quiet` flags tied to the logging system

### 5.2 Visualization Studio — Remaining
- [ ] Improve spectrogram plots (add labeled axes, colorbar, source titles)
- [ ] Add before/after spectrogram comparison for each stem
- [ ] Add interactive mode or HTML report output (optional)

### 5.3 Documentation
- [ ] Write a proper `README.md` explaining:
  - The Cocktail Party Problem and ICA theory
  - How to install and run the project
  - How to prepare stem data
  - Example outputs and expected results
- [ ] Add docstrings to all public functions (some are present, some are missing)
- [ ] Add inline comments explaining the math in `ica.py` more thoroughly

---

## Other / Cross-Cutting

- [ ] **Data**: `data/raw/` is empty — add a download script or instructions to obtain 5-stem `.wav` files (e.g. from MUSDB18 or similar dataset)
- [ ] **Testing**: No unit tests exist — add `tests/` with test cases for:
  - `ica.py` (synthetic signal recovery)
  - `mixer.py` (matrix conditioning)
  - `evaluate.py` (known SDR values)
- [ ] **`requirements.txt`**: Missing `torch`-free — confirm all deps are listed (currently has `librosa`, `soundfile`, `mir_eval`, `click`, `matplotlib`, `numpy`)
- [ ] **CI/CD**: No GitHub Actions or pre-commit hooks

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
