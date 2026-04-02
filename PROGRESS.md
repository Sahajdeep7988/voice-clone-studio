# Voice Clone Studio — Build Progress

## Phase 0 — Project Scaffold
- [x] Local git repo initialized (main branch)
- [x] Folder structure created: core/, services/, models/, dataset/, logs/
- [x] .gitignore configured (models/, dataset/, logs/, .env, *.pth)
- [x] .env.example created with SUPABASE_URL and SUPABASE_KEY placeholders
- [x] core/__init__.py and services/__init__.py created

## Phase 1 — Audio Preprocessing (`core/preprocessing.py`)
- [x] AudioPreprocessor class with process(file_paths) -> dict
- [x] Step 1: FFmpeg → 16-bit WAV at 44.1kHz → dataset/raw/
- [x] Step 2: Demucs (htdemucs) → vocal isolation → dataset/vocals/
- [x] Step 3: WebRTC VAD (aggressiveness=3) → segments ≥1.5s → dataset/segments/
- [x] Step 4: Duration check — raises PreprocessingError if < 5 minutes
- [x] Step 5: Quality report dict (Poor/Acceptable/Optimal)
- [x] PreprocessingError custom exception

## Phase 2 — LLM Hyperparameter Config (`core/llm_config.py`)
- [x] LLMConfigurator class with get_hyperparameters(dataset_stats) -> dict
- [x] Hardware scanning: VRAM (nvidia-smi / rocminfo), RAM (/proc/meminfo), CPU cores
- [x] Ollama query: Gemma 3 1B at localhost:11434 with JSON context
- [x] Strict schema validation + safe limits enforcement
- [x] Rule-based fallback when Ollama unavailable or returns bad JSON

## Phase 3 — RVC Training Engine (`core/training_engine.py`)
- [x] TrainingEngine class
- [x] start_training() — managed subprocess of Applio's train.py
- [x] Real-time stdout parsing for epoch/loss (Applio stdout format)
- [x] progress_callback(epoch, loss) invoked on every parsed epoch line
- [x] Checkpoints saved to ~/VoiceClone/models/{model_name}/
- [x] stop_training() — graceful termination
- [x] get_checkpoint_list(model_name) — returns sorted .pth paths

## Phase 4 — Inference Engine (`core/inference_engine.py`)
- [x] InferenceEngine class
- [x] convert() — audio-to-audio using Applio VoiceConverter pattern
- [x] pitch_shift support
- [x] Auto-locate .index file next to .pth for FAISS matching
- [x] list_models() — scans ~/VoiceClone/models/ recursively
- [x] Lazy-load VoiceConverter (avoids heavy import at startup)

## Phase 5 — Supabase Client (`services/supabase_client.py`)
- [x] SupabaseClient class
- [x] login() and register() — JWT auth, token in memory only
- [x] sync_training_session() — upsert to training_sessions table
- [x] get_training_history() — query by user_id
- [x] Offline guard on all network calls (try/except → graceful None/False)
- [x] Credentials loaded from .env via python-dotenv

## Phase 6 — Master Pipeline Runner (`pipeline.py`)
- [x] CLI-only runner with argparse (--files, --model-name, --email, --password)
- [x] Full pipeline: preprocess → hyperparams → train → sync
- [x] on_progress callback with Epoch/Loss console output
- [x] Error handling with clear messages throughout
- [x] Supabase sync is optional (skipped if no credentials provided)

## Phase 7 — Finalize
- [x] requirements.txt with pinned/compatible versions
- [x] Cross-referenced with Applio's requirements.txt for RVC deps
- [x] README.md with setup, usage, hardware requirements
- [x] PROGRESS.md updated (this file)
