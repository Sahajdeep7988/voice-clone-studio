"""
Audio Preprocessing Pipeline
Handles: MP3, WAV, FLAC, MP4, MKV, M4A, AAC, OGG, OPUS, AVI, MOV, WEBM
Steps: FFmpeg conversion → Demucs vocal isolation → WebRTC VAD segmentation
Extras: segment manifest, inspection API (list/delete/approve)
"""

import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path


class PreprocessingError(Exception):
    pass


class AudioPreprocessor:
    RAW_DIR      = "dataset/raw"
    VOCALS_DIR   = "dataset/vocals"
    SEGMENTS_DIR = "dataset/segments"
    MANIFEST_PATH = "dataset/segments/manifest.json"

    # All formats FFmpeg can decode
    SUPPORTED_FORMATS = {
        ".mp3", ".wav", ".flac",                    # audio
        ".m4a", ".aac", ".ogg", ".opus", ".wma",    # audio
        ".mp4", ".mkv", ".avi", ".mov", ".webm",    # video
    }

    MIN_SEGMENT_SECONDS = 1.5
    MIN_TOTAL_MINUTES   = 5.0

    def __init__(self, base_dir: str = "."):
        self.base_dir   = os.path.abspath(base_dir)
        self.raw_dir    = os.path.join(self.base_dir, self.RAW_DIR)
        self.vocals_dir = os.path.join(self.base_dir, self.VOCALS_DIR)
        self.segs_dir   = os.path.join(self.base_dir, self.SEGMENTS_DIR)
        self.manifest   = os.path.join(self.base_dir, self.MANIFEST_PATH)
        for d in [self.raw_dir, self.vocals_dir, self.segs_dir]:
            os.makedirs(d, exist_ok=True)
        # Lazy Demucs model cache — loaded once per instance
        self._demucs_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_paths: list) -> dict:
        """
        Process a list of audio/video files through the full pipeline.
        Files are processed sequentially; results are aggregated.
        Returns a quality report dict including segment manifest.
        """
        self._validate_formats(file_paths)

        segment_paths = []
        for fp in file_paths:
            raw_path   = self._ffmpeg_convert(fp)
            vocal_path = self._demucs_isolate(raw_path)
            segs       = self._vad_segment(vocal_path, source_file=fp)
            segment_paths.extend(segs)

        total_seconds  = sum(self._wav_duration(s) for s in segment_paths)
        total_minutes  = total_seconds / 60.0

        if total_minutes < self.MIN_TOTAL_MINUTES:
            raise PreprocessingError(
                f"Dataset too short: minimum 5 minutes required "
                f"(got {total_minutes:.2f} minutes)"
            )

        if total_minutes < 10:
            quality = "Poor"
        elif total_minutes <= 20:
            quality = "Acceptable"
        else:
            quality = "Optimal"

        self._write_manifest(segment_paths)

        return {
            "segment_paths":          segment_paths,
            "total_duration_minutes": round(total_minutes, 2),
            "segment_count":          len(segment_paths),
            "quality_label":          quality,
            "manifest_path":          self.manifest,
        }

    # ------------------------------------------------------------------
    # Segment inspection API (for user review before training)
    # ------------------------------------------------------------------

    def get_segments(self) -> list:
        """
        Return list of segment metadata dicts from the manifest.
        Each dict: {path, duration_s, source_file, approved, size_bytes}
        """
        if not os.path.exists(self.manifest):
            return []
        with open(self.manifest) as f:
            return json.load(f)

    def approve_segments(self, approved_paths: list) -> int:
        """
        Mark specific segment paths as approved=True, others as approved=False.
        Returns count of approved segments.
        Only approved segments are used for training.
        """
        segs = self.get_segments()
        approved_set = set(os.path.abspath(p) for p in approved_paths)
        for seg in segs:
            seg["approved"] = os.path.abspath(seg["path"]) in approved_set
        self._save_manifest(segs)
        count = sum(1 for s in segs if s["approved"])
        print(f"[Preprocess] {count}/{len(segs)} segments approved.")
        return count

    def approve_all(self) -> int:
        """Mark all existing segments as approved."""
        segs = self.get_segments()
        for seg in segs:
            seg["approved"] = True
        self._save_manifest(segs)
        return len(segs)

    def delete_segment(self, path: str) -> bool:
        """Delete a segment file and remove it from the manifest."""
        abs_path = os.path.abspath(path)
        segs = self.get_segments()
        new_segs = [s for s in segs if os.path.abspath(s["path"]) != abs_path]
        if len(new_segs) == len(segs):
            return False  # not found
        if os.path.exists(abs_path):
            os.remove(abs_path)
        self._save_manifest(new_segs)
        print(f"[Preprocess] Deleted segment: {path}")
        return True

    def get_approved_segment_paths(self) -> list:
        """Return paths of all approved segments (for passing to training)."""
        return [s["path"] for s in self.get_segments() if s.get("approved", True)]

    # ------------------------------------------------------------------
    # Step 1: FFmpeg conversion
    # ------------------------------------------------------------------

    def _ffmpeg_convert(self, input_path: str) -> str:
        # Use a hash-based name to avoid collisions between files with same stem
        stem  = Path(input_path).stem
        fhash = hashlib.md5(os.path.abspath(input_path).encode()).hexdigest()[:8]
        out_path = os.path.join(self.raw_dir, f"{stem}_{fhash}.wav")

        if os.path.exists(out_path):
            print(f"[Preprocess] Already converted (cached): {out_path}")
            return out_path

        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "44100",
            "-ac", "1",
            "-sample_fmt", "s16",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PreprocessingError(
                f"FFmpeg failed for {input_path}:\n{result.stderr}"
            )
        print(f"[Preprocess] Converted: {input_path} → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Step 2: Demucs vocal isolation (Python API)
    # ------------------------------------------------------------------

    def _demucs_isolate(self, wav_path: str) -> str:
        import torch
        import soundfile as sf
        from demucs.separate import load_track, get_model_from_args
        from demucs.apply import apply_model
        import argparse

        stem       = Path(wav_path).stem
        vocal_path = os.path.join(self.vocals_dir, f"{stem}_vocals.wav")

        if os.path.exists(vocal_path):
            print(f"[Preprocess] Vocals already isolated (cached): {vocal_path}")
            return vocal_path

        # Load model once per instance
        if self._demucs_model is None:
            print("[Preprocess] Loading htdemucs model (first call)...")
            args = argparse.Namespace(
                name="htdemucs", repo=None,
                device="cuda" if torch.cuda.is_available() else "cpu",
                shifts=1, overlap=0.25, no_split=False,
                segment=None, jobs=0, verbose=False, stem="vocals",
            )
            self._demucs_model = get_model_from_args(args)
            self._demucs_model.eval()
            if torch.cuda.is_available():
                self._demucs_model.cuda()

        model = self._demucs_model
        print(f"[Preprocess] Separating vocals: {wav_path}")
        wav = load_track(wav_path, model.audio_channels, model.samplerate)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(
                model, wav[None],
                device="cuda" if torch.cuda.is_available() else "cpu",
                shifts=1, split=True, overlap=0.25, progress=True,
            )[0]

        sources    = sources * ref.std() + ref.mean()
        vocal_idx  = model.sources.index("vocals")
        vocal_np   = sources[vocal_idx].cpu().numpy().T  # [samples, channels]

        sf.write(vocal_path, vocal_np, model.samplerate, subtype="PCM_16")
        print(f"[Preprocess] Vocals isolated: {vocal_path}")
        return vocal_path

    # ------------------------------------------------------------------
    # Step 3: WebRTC VAD segmentation
    # ------------------------------------------------------------------

    def _vad_segment(self, wav_path: str, source_file: str = "") -> list:
        import webrtcvad

        vad   = webrtcvad.Vad(3)
        audio, sample_rate, num_channels = self._read_wav(wav_path)

        # VAD requires 8k/16k/32k/48k Hz mono
        if sample_rate not in (8000, 16000, 32000, 48000) or num_channels != 1:
            wav_path  = self._resample_for_vad(wav_path)
            audio, sample_rate, num_channels = self._read_wav(wav_path)

        frame_ms    = 30
        frame_size  = int(sample_rate * frame_ms / 1000)
        frame_bytes = frame_size * 2

        frames = [
            audio[i: i + frame_bytes]
            for i in range(0, len(audio) - frame_bytes + 1, frame_bytes)
        ]

        voiced_flags = []
        for frame in frames:
            try:
                voiced_flags.append(
                    vad.is_speech(frame, sample_rate) if len(frame) == frame_bytes else False
                )
            except Exception:
                voiced_flags.append(False)

        # Merge consecutive voiced frames → segments with padding
        segments   = []
        in_seg     = False
        seg_start  = 0
        PADDING_FRAMES = 5  # keep 5 frames of silence around speech

        for i, voiced in enumerate(voiced_flags):
            if voiced and not in_seg:
                in_seg    = True
                seg_start = max(0, i - PADDING_FRAMES)
            elif not voiced and in_seg:
                in_seg = False
                segments.append((seg_start, min(len(voiced_flags), i + PADDING_FRAMES)))

        if in_seg:
            segments.append((seg_start, len(voiced_flags)))

        stem        = Path(wav_path).stem
        saved_paths = []
        for idx, (sf_start, sf_end) in enumerate(segments):
            seg_audio = audio[sf_start * frame_bytes: sf_end * frame_bytes]
            duration  = len(seg_audio) / (sample_rate * 2)
            if duration < self.MIN_SEGMENT_SECONDS:
                continue
            out_path = os.path.join(self.segs_dir, f"{stem}_seg{idx:04d}.wav")
            self._write_wav(out_path, seg_audio, sample_rate)
            saved_paths.append(out_path)

        print(
            f"[Preprocess] VAD segmented {Path(wav_path).name}: "
            f"{len(saved_paths)} segments kept"
        )
        return saved_paths

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _write_manifest(self, segment_paths: list) -> None:
        existing = {s["path"]: s for s in self.get_segments()}
        entries  = []
        for path in segment_paths:
            abs_path = os.path.abspath(path)
            entry = existing.get(path, {
                "path":       path,
                "duration_s": round(self._wav_duration(path), 3),
                "size_bytes": os.path.getsize(path),
                "approved":   True,
            })
            entries.append(entry)
        self._save_manifest(entries)

    def _save_manifest(self, entries: list) -> None:
        with open(self.manifest, "w") as f:
            json.dump(entries, f, indent=2)

    # ------------------------------------------------------------------
    # WAV / format helpers
    # ------------------------------------------------------------------

    def _read_wav(self, wav_path: str):
        with wave.open(wav_path, "rb") as wf:
            return wf.readframes(wf.getnframes()), wf.getframerate(), wf.getnchannels()

    def _resample_for_vad(self, wav_path: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = ["ffmpeg", "-y", "-i", wav_path,
               "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp.name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PreprocessingError(f"Resample for VAD failed:\n{result.stderr}")
        return tmp.name

    def _write_wav(self, path: str, raw: bytes, rate: int) -> None:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(raw)

    def _wav_duration(self, wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    def _validate_formats(self, file_paths: list) -> None:
        for fp in file_paths:
            if not os.path.exists(fp):
                raise PreprocessingError(f"File not found: {fp}")
            ext = Path(fp).suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise PreprocessingError(
                    f"Unsupported format: '{ext}'. "
                    f"Supported: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
                )
