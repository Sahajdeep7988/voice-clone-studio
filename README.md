# Voice Clone Studio

A production-ready, local-first voice cloning pipeline built on [RVC (Retrieval-based Voice Conversion)](https://github.com/IAHispano/Applio).
Train a custom voice model from audio samples and convert any speech to that voice — entirely on your own machine.

---

## What it does

1. **Preprocesses** audio/video files — converts to WAV, isolates vocals with Demucs, and splits into clean segments using WebRTC VAD.
2. **Configures** training hyperparameters using a local Gemma 3 1B model (via Ollama) that reads your hardware specs — with a deterministic fallback if Ollama is unavailable.
3. **Trains** an RVC voice model as a managed subprocess, streaming epoch/loss progress in real time and saving checkpoints.
4. **Converts** any audio to the cloned voice using the trained model.
5. **Syncs** training session metadata to Supabase (optional) for cross-device history.

> **UI coming in next phase — designed in Figma.** This release is CLI-only.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM  | None (CPU training possible but slow) | 6 GB+ (NVIDIA CUDA or AMD ROCm) |
| RAM       | 8 GB    | 16 GB+      |
| Disk      | 5 GB free | 20 GB+ for datasets and models |
| CPU       | 4 cores | 8 cores+    |

- CUDA 12.x recommended for GPU training
- FFmpeg must be installed system-wide (`sudo apt install ffmpeg`)
- Ollama must be running locally for LLM-assisted hyperparameter selection

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/voice-clone-studio
cd voice-clone-studio
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA (recommended)

```bash
pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install FFmpeg

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 4. Set up Ollama (optional, for LLM hyperparameter selection)

```bash
# Install Ollama: https://ollama.com
ollama pull gemma3:1b
ollama serve
```

### 5. Configure Supabase (optional)

```bash
cp .env.example .env
# Edit .env and fill in your SUPABASE_URL and SUPABASE_KEY
```

### 6. Clone RVC dependencies (Applio)

```bash
git clone https://github.com/IAHispano/Applio rvc
pip install -r rvc/requirements.txt
# Download pretrained weights:
mkdir -p ~/VoiceClone/pretrained
# Place f0G48k.pth and f0D48k.pth in ~/VoiceClone/pretrained/
```

---

## How to run

### Full pipeline (preprocessing → training)

```bash
python pipeline.py --files audio.mp3 --model-name myvoice
```

### With multiple input files

```bash
python pipeline.py --files speech1.wav interview.mp4 podcast.mp3 \
                   --model-name myvoice
```

### With Supabase sync

```bash
python pipeline.py --files audio.mp3 --model-name myvoice \
                   --email user@example.com --password yourpassword
```

### Inference only (after training)

```python
from core.inference_engine import InferenceEngine

engine = InferenceEngine()
engine.convert(
    model_path="~/VoiceClone/models/myvoice/myvoice.pth",
    input_audio_path="source.wav",
    output_path="output/converted.wav",
    pitch_shift=0,
)
```

---

## Project structure

```
voice-clone-studio/
├── core/
│   ├── preprocessing.py     # FFmpeg + Demucs + WebRTC VAD
│   ├── llm_config.py        # Ollama Gemma 3 1B hyperparameter selection
│   ├── training_engine.py   # RVC training subprocess wrapper
│   └── inference_engine.py  # Audio-to-audio voice conversion
├── services/
│   └── supabase_client.py   # Auth + training session sync
├── pipeline.py              # CLI runner — connects all modules
├── requirements.txt
├── .env.example
└── PROGRESS.md
```

---

## Supported input formats

MP3, WAV, FLAC, MP4, MKV

---

## Notes

- Models, datasets, and logs are gitignored — they stay on your machine.
- `.env` is never committed — credentials stay local.
- Minimum dataset duration: **5 minutes** of clean speech after preprocessing.
- Recommended: 10–20 minutes for an "Acceptable" quality model, 20+ for "Optimal".
