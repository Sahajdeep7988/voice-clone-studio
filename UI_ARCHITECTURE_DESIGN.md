# Voice Clone Studio: UI Architecture & Analysis

Based on a thorough analysis of the FastAPI backend, RVC pipeline, and user requirements, here is the architectural design and product breakdown for the Voice Clone Studio desktop application.

## 1. Extracted Product Features

**User Actions**
- **Initialize App:** First-time setup auto-checks/installs prerequisites (bundled FFmpeg, LLM gguf, PyTorch/CUDA checks).
- **Create Session:** Select input media (audio/video), assign a model name, choose simple vs. advanced mode.
- **Segment Curation (Advanced):** Listen to extracted vocal segments, delete/reject bad clips, view waveform data.
- **Hyperparameter Tuning (Advanced):** Override LLM-generated hyperparameters (Batch Size, Epochs, Learning Rate).
- **Training Control:** Start, pause, resume, or forcefully stop model training.
- **Live Monitoring:** Watch epoch progress, loss metrics, and terminal-like streaming logs.
- **Inference (Conversion):** Select a trained checkpoint, upload target audio, adjust pitch shift, and run conversion.
- **Cloud Sync:** Login/register via Supabase to sync training session metadata across devices.

**Data Flows**
1. **Raw Ingestion:** Media `->` FFmpeg (`wav`) `->` Demucs (`vocals`) `->` WebRTC VAD (`segments`).
2. **Manifest Tracking:** Segments `->` `manifest.json` (tracks `duration_s` and `approved` flag).
3. **Hardware-Aware Config:** System Scan + Dataset length `->` local `llama.cpp` `->` Hyperparameters.
4. **Subprocess Training:** Hyperparameters + Segments `->` Applio Subprocess `->` Stdout parsing (Epoch/Loss) `->` `G_*.pth` Checkpoints.
5. **Session State:** Local `~/.voice_clone_studio/sessions/` JSON files `->` Sync to Supabase.

**System States**
- `created`: Files selected, waiting to start.
- `preprocessing`: FFmpeg/Demucs/VAD running.
- `preprocessing_done` (Implicit): Waiting for manual segment review (in Advanced mode).
- `training`: Applio subprocess is active; streaming epoch/loss.
- `paused`: User stopped training gracefully at an epoch boundary.
- `done`: Target epochs reached, final checkpoint saved.
- `error`: Pipeline failure (e.g., OOM, bad data).

---

## 2. Workflows

### Primary Workflows
**A. The "Fire-and-Forget" Training Flow (Default)**
Select Files -> Enter Model Name -> Click "Train" -> System preprocesses, LLM configures, and trains automatically -> User watches progress -> Done.

**B. The "Manual Curation" Training Flow (Advanced)**
Select Files -> Enter Model Name -> Click "Preprocess Only" -> Audio extraction completes -> **Curation UI:** Play clips, approve/reject -> System calculates LLM hyperparams -> User reviews/edits hyperparams -> Click "Start Training" -> Done.

**C. Voice Conversion (Inference)**
Navigate to "Inference" -> Select a completed Model -> Select Target Audio -> Adjust Pitch -> Convert -> Play back cloned audio -> Save/Export.

### Secondary Workflows
**D. Cross-Device "My Models" Sync**
User logs in -> Views "My Models" grid -> UI displays sessions trained on *other* devices (shows epoch count, duration, metadata) but clearly marks local weights as "Not found locally".
**E. Mid-Training Checkpoint Testing**
While paused or done, user selects a specific `G_50e.pth` checkpoint -> Runs quick inference test -> Decides whether to resume training.

---

## 3. Recommended UI Stack

**Choice: Tauri + React + Tailwind CSS (via Vite)**

**Justification:**
1. **Developer-Beautiful Aesthetic:** React + Tailwind allows us to build a UI heavily inspired by tools like Linear, Vercel, or Claude—dark, sleek, data-rich, with monospaced accents and subtle glows. This is hard to achieve with PySide6/Qt without fighting the framework.
2. **Performance & Weight:** Tauri provides a lightweight binary shell (`.deb`/AppImage for Linux) that leverages the system webview, consuming far less RAM than Electron. This is critical because Demucs and PyTorch need all available VRAM/RAM.
3. **Backend Separation:** Since the backend is already decoupled (using `AppBackend` as an API), Tauri can spawn a lightweight Python FastAPI server as a "sidecar" on startup, or communicate via native IPC to a packaged Python binary.
4. **Component Ecosystem:** Access to libraries like `framer-motion` (for smooth state transitions), `lucide-react` (clean icons), and audio visualization libraries (Wavesurfer.js for segment review).

---

## 4. UI Architecture & Screen Breakdown

### Component Hierarchy
```text
AppShell (Layout with Sidebar & Titlebar)
 ├── Auth Modal / Hardware Check Splash
 ├── Sidebar (Navigation: Dashboard, New Training, My Models, Inference, Settings)
 └── MainView
      ├── Dashboard (Recent active sessions, quick stats)
      ├── SessionBuilder (The Form: Files, Name, Simple/Advanced toggle)
      ├── CurationView (Audio player list, waveforms, Approve/Reject toggles)
      ├── ActiveTrainingView (Progress rings, Loss graph, Terminal log output)
      ├── ModelsGrid (Supabase synced cards, local vs remote indicators)
      └── InferenceView (Model selector, Audio input, Pitch slider, Results player)
```

### Visual "Developer" Aesthetic Guidelines
- **Theme:** Deep Dark Mode (slate/zinc backgrounds, `#09090b`).
- **Typography:** Inter/Geist for UI text; JetBrains Mono for logs, hardware specs, and hyperparameter JSONs.
- **Cards & Borders:** 1px subtle borders (`border-white/10`), slight glassmorphism.
- **States:** Use colors sparingly but effectively. Amber for `preprocessing`, Emerald for `training/done`, Rose for `error`.
- **Logs:** A dedicated, collapsible terminal window in the Training View that streams actual `stdout` from the Python backend.

---

## 5. Key Interaction Flows (Textual Wireframes)

### 1. Initialization / Splash Screen
```text
[ Logo: Voice Clone Studio ]
Verifying Environment...
[✓] NVIDIA RTX 4090 Detected (24GB VRAM)
[✓] PyTorch CUDA available
[✓] Local LLM (gemma-2b-q4.gguf) loaded
[ ] Bundle check complete... Starting app.
```

### 2. "My Models" View (Supabase synced)
```text
+-------------------------------------------------------------+
| My Models                                      [+ New Model]|
+-------------------------------------------------------------+
|                                                             |
|  [ Card: "Podcast_Voice" ]     [ Card: "Gaming_Alter_Ego" ] |
|  Status: DONE                  Status: PAUSED               |
|  Epochs: 250/250               Epochs: 100/150              |
|  Duration: 14m dataset         Duration: 8m dataset         |
|  Device: LOCAL                 Device: MACBOOK-PRO          |
|  [Test Checkpoints] [Convert]  [ Weights not found locally ]|
+-------------------------------------------------------------+
```

### 3. Advanced Manual Curation Screen
```text
Preprocessing Complete: 14 minutes of valid audio found.
Please review your segments:

[ Play All ]  [ Approve All ]  [ Reject All ]

Segment 001 (4.2s)  [==Waveform==]  [✓ Approved]
Segment 002 (1.8s)  [==Waveform==]  [X Rejected] (Too short/noisy)
Segment 003 (5.5s)  [==Waveform==]  [✓ Approved]

------------------------------------------------
LLM Recommended Hyperparameters:
{ "batch_size": 16, "epochs": 200, "learning_rate": 0.0001 }
[ Edit Params ]
                                 [ START TRAINING ]
```

### 4. Active Training Monitor
```text
Model: Podcast_Voice
Status: TRAINING (Epoch 42 / 200)

Progress: [===================>                       ] 21%
Latest Loss: 1.2045 (Graph: ↘↘↘)

[ || Pause ]  [ [x] Stop ]

>_ Terminal Logs --------------------------------------------
[RVC] epoch=41 step=1100 lowest_value=1.2104
[RVC] epoch=42 step=1150 lowest_value=1.2045
[Session] Updated session state to Supabase.
-------------------------------------------------------------
```
