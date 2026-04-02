# Voice Clone Studio - Comprehensive Technical Documentation

This document provides a detailed overview of the Voice Clone Studio application architecture, modules, entry points, and technical stack. It is designed to be provided to an AI model to plan the next phase of development: converting the application into a fully working UI-driven Linux application.

## 1. Project Overview

**Voice Clone Studio** is a production-ready, local-first voice cloning pipeline built on top of [RVC (Retrieval-based Voice Conversion) via Applio](https://github.com/IAHispano/Applio).

It takes user-provided audio or video files, preprocesses them (extracting audio, isolating vocals, and segmenting based on Voice Activity Detection), calculates optimal hyperparameters via a local LLM, and trains a custom RVC voice model. It also provides an inference engine to convert audio using the newly trained models.

Currently, it operates via a CLI (`pipeline.py`), but the underlying backend is decoupled and ready to be integrated into a GUI (e.g., PyQt/PySide, CustomTkinter, or an Electron/Tauri frontend).

### Core Features
- **Preprocessing:** FFmpeg audio extraction -> Demucs vocal isolation -> WebRTC VAD segmentation.
- **LLM Hyperparameters:** Local `llama.cpp` configures RVC hyperparameters based on hardware stats.
- **Training Engine:** Manages Applio subprocesses for RVC preprocessing, feature extraction, and training.
- **Inference Engine:** Audio-to-audio voice conversion using the trained `.pth` checkpoints.
- **Session Management:** Training state is persisted, allowing resumes and paused jobs.
- **Cloud Sync:** Optional Supabase integration for synchronizing training history.

---

## 2. Directory Structure

```text
voice-clone-studio/
├── core/                       # Core pipeline modules
│   ├── inference_engine.py     # Runs voice conversion (inference)
│   ├── llm_config.py           # Uses llama.cpp to generate training hyperparameters
│   ├── logger.py               # Custom logging utility
│   ├── preprocessing.py        # FFmpeg, Demucs, VAD integration
│   └── training_engine.py      # Subprocess wrapper for Applio RVC training
├── services/                   # Application state and network services
│   ├── backend.py              # Central orchestrator / API for the CLI & UI
│   ├── session_manager.py      # Local JSON-based state persistence
│   └── supabase_client.py      # Cloud sync & Auth (Supabase)
├── rvc/                        # Submodule: Applio repository (Retrieval-based Voice Conversion)
├── pipeline.py                 # CLI entry point
├── requirements.txt            # Python dependencies
├── .env.example                # Template for environment variables (Supabase)
├── README.md                   # Setup and usage guide
└── PROGRESS.md                 # Current task progress checklist
```

---

## 3. Core Modules (`core/`)

### 3.1 `preprocessing.py`
Responsible for taking raw media files and preparing them for RVC training.
- **Step 1: FFmpeg** - Converts any input media (MP3, MP4, FLAC, etc.) into 16-bit, 44.1kHz WAV.
- **Step 2: Demucs** - Uses `htdemucs` to isolate vocals from the audio, dropping instrumentals/background noise. Serialized via a lock to prevent GPU OOM.
- **Step 3: WebRTC VAD** - Segments the isolated vocals into clean chunks of speech (aggressiveness=3). Drops chunks shorter than 1.5 seconds.
- **Validation:** Ensures there are at least 5 minutes of valid audio data.
- **Concurrency:** Uses `ThreadPoolExecutor` to process multiple input files concurrently.

### 3.2 `llm_config.py`
Determines optimal RVC hyperparameters (batch size, epochs, etc.).
- **Hardware Scan:** Detects System RAM, CPU cores, GPU VRAM (via `nvidia-smi` or `rocminfo`), and free disk space.
- **LLM Engine:** Uses `llama.cpp` (`llama-cpp-python`) with a quantized Gemma model (`gemma-3-270m-it-Q4_0.gguf`) to dynamically decide parameters based on the hardware and dataset length.
- **Fallback:** If the LLM fails or hardware isn't supported, falls back to a deterministic rule-based calculation.
- **Constraints:** Clamps outputs to safe limits (e.g., max batch size 32).

### 3.3 `training_engine.py`
Wraps the Applio (RVC) training scripts in a managed subprocess.
- **Pipeline Execution:** Runs `preprocess.py`, `extract.py`, and `train.py` from the `rvc/` directory.
- **Progress Tracking:** Parses Applio's stdout to track `epoch` and `loss`, feeding this back via callbacks.
- **Lifecycle Management:** Can gracefully stop (SIGTERM) at the end of an epoch, force stop (SIGKILL), or resume training from existing checkpoints.
- **Outputs:** Models are saved to `rvc/logs/{model_name}/`.

### 3.4 `inference_engine.py`
Uses trained RVC models for audio conversion.
- **Chdir-free Bootstrap:** Patches Applio modules at runtime to use absolute paths, ensuring it can be imported anywhere without `os.chdir()`.
- **Conversion:** Invokes Applio's `VoiceConverter` directly in Python.
- **Index matching:** Automatically locates the `.index` file associated with the `.pth` model for better voice likeness.

---

## 4. Service Modules (`services/`)

### 4.1 `backend.py` (`AppBackend`)
This is the **primary API** for the application. Any UI or CLI should interact solely with this class.
- **Methods:** `run_pipeline`, `resume_pipeline`, `stop_training`, `get_session_status`, `list_sessions`, `convert_audio`, `test_checkpoint`.
- **Threading:** Can run tasks synchronously or asynchronously (spawning daemon threads).
- **Session Linking:** Ties the core modules together, linking Preprocessing outputs -> LLM Config -> Training Engine, while updating `SessionManager`.

### 4.2 `session_manager.py`
Persists the state of training jobs to the local disk, allowing the app to close and resume later.
- **Storage Path:** `~/.voice_clone_studio/sessions/{session_id}.json`.
- **State Fields:** Tracks `model_name`, `status` (created, preprocessing, training, paused, done, error), `current_epoch`, `total_epochs`, etc.

### 4.3 `supabase_client.py`
Optional cloud integration for tracking model history across machines.
- **Features:** Email/password login, registration, JWT token storage (in memory).
- **Sync:** Upserts session metadata (`duration_seconds`, `hardware_profile`, etc.) to a `training_sessions` table in Supabase.
- **Offline Resilient:** Fails gracefully without throwing exceptions if network is down.

---

## 5. Entry Points

### CLI: `pipeline.py`
- Exposes `AppBackend` via `argparse`.
- Supported actions:
  - Run full pipeline: `--files a.mp3 --model-name myvoice`
  - Resume paused session: `--resume-session <id>`
  - List sessions: `--list-sessions`
  - Check status: `--status <id>`
  - Test a trained checkpoint: `--test-checkpoint <id> <ckpt.pth> <test.wav>`

---

## 6. Local State & File Paths

- **Models & Logs (Applio):**
  - Checkpoints: `rvc/logs/{model_name}/`
  - Global Pretrained models: `rvc/models/pretraineds/`
- **Application Sessions:**
  - Session DB: `~/.voice_clone_studio/sessions/`
  - Temporary Session Data (Dataset): `./sessions_data/{session_id}/dataset/`
- **LLM Weights:** `./models/gemma-2b-q4.gguf`

---

## 7. Next Steps: Conversion to a Full Linux GUI Application

To convert this project into a packaged Linux application with a GUI, an AI planner should consider the following phases:

### Phase A: GUI Implementation
- The UI should instantiate a singleton `AppBackend()` (`services/backend.py`).
- **Framework Choice:**
  - *PyQt6 / PySide6* (Recommended for complex desktop apps).
  - *CustomTkinter* (Simpler, pure python).
  - *Electron / Tauri with Python backend* (More modern web-based UI, requiring a local FastAPI bridge).
- **Key Screens Needed:**
  1. **Dashboard:** List recent sessions from `backend.list_sessions()`.
  2. **Create Model Wizard:** Select files, name model, trigger `backend.run_pipeline(async_mode=True)`.
  3. **Training View:** Poll `backend.get_session_status()` to display progress bars (Epoch/Loss) and a stop/pause button.
  4. **Inference View:** Select a trained model from `backend.list_models()`, select an input audio file, adjust pitch, and run `backend.convert_audio()`.
  5. **Settings/Auth:** Supabase login screen and hardware settings.

### Phase B: Asynchronous and Event-Driven Architecture
- `AppBackend` currently uses basic `threading.Thread` for async mode.
- For a GUI (like PyQt), this should be integrated with the GUI's event loop (e.g., `QThread` and `pyqtSignal`) so that `progress_callback` updates the UI thread safely without blocking.

### Phase C: System Packaging for Linux
- **FFmpeg & System Dependencies:** Ensure `ffmpeg` is available. For a flatpak/appimage, FFmpeg needs to be bundled.
- **Python Freezing:** Use `PyInstaller` or `Nuitka` to package the python code.
  - *Challenges:* `torch`, `torchaudio`, and `demucs` are large and complex to freeze. `Applio` relies on relative paths and subprocess execution, which may require special PyInstaller hooks.
- **Applio (RVC) Bundling:** Since `training_engine.py` calls `Applio` scripts via `subprocess.Popen([PYTHON, "rvc/train/train.py", ...])`, the frozen app must ship a working Python environment or rewrite the training engine to import Applio functions directly rather than using subprocesses.
- **Model Weights Download:** The GUI should include an initialization screen to download Applio pretrained weights (`f0G48k.pth`, `f0D48k.pth`, `rmvpe.pt`, etc.) and the LLM `.gguf` if they are missing.

### Phase D: Hardware Acceleration on Linux
- Ensure PyTorch uses CUDA for NVIDIA or ROCm for AMD.
- If distributing as a binary, consider how PyTorch wheels differ for CUDA vs CPU vs ROCm. Flatpak might be the best distribution method to manage these massive graphical library runtimes.
