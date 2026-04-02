# Audit Report

## 1. CRITICAL BUGS

```json
{
  "BLOCKER": [
    "Data Isolation Flaw: Global shared path for dataset segments",
    "Empty file processing: Silent tensor size/runtime errors during preprocessing and inference",
    "Missing Checkpoint Validation: Silent success even when checkpoint creation fails",
    "Process Leaks: Training engine leaks Applio subprocesses when python process is interrupted"
  ],
  "HIGH": [
    "Race Conditions: ThreadPool workers modify manifest and check lengths concurrently without locks",
    "Math Errors: ZeroDivisionError when standard deviation of a silent audio frame is 0 in Demucs isolation"
  ],
  "MEDIUM": [
    "Missing File Size Validation on Cached Outputs: Corrupted zero-byte artifacts from previous killed processes are returned as valid"
  ]
}
```

## 2. BREAKPOINT ANALYSIS

**Scenario 1: Corrupted / Zero-length audio segment**
* **Input**: An empty (0 byte) or non-existent `.wav` file fed to the pipeline or inference.
* **Failure Location**: `core/inference_engine.py` (InferenceEngine.convert) & `services/backend.py` (_validate_dataset)
* **Reason**: Unchecked assumptions that all returned paths are valid audio with `>0` bytes. Causes training or inference to silently fail and produce empty output, or to crash the pipeline entirely when processing tensors of size `0`.

**Scenario 2: Extremely short audio (<0.2s) & Silent Audio**
* **Input**: An audio file containing only silence.
* **Failure Location**: `core/preprocessing.py` (_demucs_isolate)
* **Reason**: Division by zero when scaling waveform: `wav = (wav - ref.mean()) / ref.std()`. If standard deviation is 0, this yields `NaN` which corrupts the entire pipeline's downstream tensor inputs, leading to `NaN` model weights.

**Scenario 3: Interrupted training & Process Leaks**
* **Input**: A Python process running `start_training` gets terminated or errors out.
* **Failure Location**: `core/training_engine.py` (start_training)
* **Reason**: Applio subprocesses are spawned using `subprocess.Popen` but the `process.wait()` and stdout reading is not wrapped in `try...finally`. If an exception occurs, the spawned training processes (which hold onto the GPU) are orphaned and continue running or hold VRAM, causing OOM on subsequent runs.

**Scenario 4: Concurrent Runs (Data Safety)**
* **Input**: Running two sessions concurrently.
* **Failure Location**: `services/backend.py` (get_segments, approve_segments, etc.)
* **Reason**: The `dataset_path` and `base_dir` were hardcoded to `os.path.join(self.base_dir, "dataset", "segments")`. Concurrent sessions would overwrite each other's audio files, segment data, and manifest files.

**Scenario 5: Checkpoints are not created but marked as success**
* **Input**: A failure during Applio training that doesn't trigger a non-zero exit code but fails to write the `.pth` file.
* **Failure Location**: `core/training_engine.py`
* **Reason**: The `start_training` function returns `final_path = self._find_latest_checkpoint(model_name)` which just guesses the path based on conventions, but never asserts that the file actually exists on disk.

## 3. EXACT FIXES

### 1. `core/preprocessing.py`
**Code Change**:
Added `self._manifest_lock` in `approve_segments`, `approve_all`, and `delete_segment`.
Added check for zero standard deviation:
```python
std = ref.std()
if std == 0.0:
    std = 1e-8
wav = (wav - ref.mean()) / std
```
Checked file sizes when reusing cached FFmpeg and Demucs outputs:
```python
if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
    return out_path
```

### 2. `services/backend.py`
**Code Change**:
Implemented Session-Specific paths instead of a global `dataset/` directory:
```python
session_base = os.path.join(self.base_dir, "sessions_data", session_id)
preprocessor = AudioPreprocessor(base_dir=session_base)
```
Used `wave` module to validate that audio files contain more than 0 frames:
```python
import wave
try:
    with wave.open(full, "rb") as wf:
        if wf.getnframes() > 0:
            valid.append(f)
        else:
            empty.append(full)
except Exception:
    empty.append(full)
```

### 3. `core/training_engine.py`
**Code Change**:
Wrapped stdout reading in `try...finally` to ensure subprocesses are cleanly terminated:
```python
try:
    for line in self._process.stdout:
        # ...
finally:
    with self._lock:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
```
Added checkpoint existence validation:
```python
if not os.path.exists(final_path):
    raise RuntimeError(f"Training finished but checkpoint was not created: {final_path}")
```

### 4. `core/inference_engine.py`
**Code Change**:
Added checks for empty input audio and verified output exists and is non-empty:
```python
if os.path.getsize(input_audio_path) == 0:
    raise ValueError(f"Input audio is empty: {input_audio_path}")

# ... (after conversion)

if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
    raise RuntimeError(f"Inference failed, output path empty or missing: {output_path}")
```

## 4. FINAL VERDICT

```json
{
  "production_ready": false,
  "reason": "The system contained multiple critical flaws regarding data isolation (concurrent runs overwriting files), silent failures handling missing/empty audio, math errors causing NaN tensors, and process leaks that would cause a production server to OOM.",
  "must_fix_before_production": [
    "Ensure session isolation for preprocessing files",
    "Properly clean up subprocesses under all error conditions",
    "Validate mathematical operations (ZeroDivisionError) on audio tensors",
    "Ensure outputs and intermediate checkpoints are validated for file size and existence"
  ]
}
```