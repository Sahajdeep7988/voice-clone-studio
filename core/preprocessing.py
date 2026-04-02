"""
Audio Preprocessing Pipeline
Handles: MP3, WAV, FLAC, MP4, MKV
Steps: FFmpeg conversion → Demucs vocal isolation → WebRTC VAD segmentation
"""

import os
import subprocess
import wave
import struct
import tempfile
from pathlib import Path


class PreprocessingError(Exception):
    pass


class AudioPreprocessor:
    RAW_DIR = "dataset/raw"
    VOCALS_DIR = "dataset/vocals"
    SEGMENTS_DIR = "dataset/segments"
    SUPPORTED_FORMATS = {".mp3", ".wav", ".flac", ".mp4", ".mkv"}
    MIN_SEGMENT_SECONDS = 1.5
    MIN_TOTAL_MINUTES = 5.0

    def __init__(self):
        for d in [self.RAW_DIR, self.VOCALS_DIR, self.SEGMENTS_DIR]:
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_paths: list) -> dict:
        """
        Process a list of audio/video files through the full pipeline.
        Returns a quality report dict.
        """
        self._validate_formats(file_paths)

        # Step 1: FFmpeg → 16-bit WAV at 44.1 kHz
        raw_paths = []
        for fp in file_paths:
            raw_path = self._ffmpeg_convert(fp)
            raw_paths.append(raw_path)

        # Step 2: Demucs → isolate vocals
        vocal_paths = []
        for rp in raw_paths:
            vocal_path = self._demucs_isolate(rp)
            vocal_paths.append(vocal_path)

        # Step 3: WebRTC VAD → segment and filter
        segment_paths = []
        for vp in vocal_paths:
            segs = self._vad_segment(vp)
            segment_paths.extend(segs)

        # Step 4: Duration check
        total_seconds = sum(self._wav_duration(s) for s in segment_paths)
        total_minutes = total_seconds / 60.0
        if total_minutes < self.MIN_TOTAL_MINUTES:
            raise PreprocessingError(
                f"Dataset too short: minimum 5 minutes required "
                f"(got {total_minutes:.2f} minutes)"
            )

        # Step 5: Quality report
        if total_minutes < 10:
            quality = "Poor"
        elif total_minutes <= 20:
            quality = "Acceptable"
        else:
            quality = "Optimal"

        return {
            "segment_paths": segment_paths,
            "total_duration_minutes": round(total_minutes, 2),
            "segment_count": len(segment_paths),
            "quality_label": quality,
        }

    # ------------------------------------------------------------------
    # Step 1: FFmpeg conversion
    # ------------------------------------------------------------------

    def _ffmpeg_convert(self, input_path: str) -> str:
        stem = Path(input_path).stem
        out_path = os.path.join(self.RAW_DIR, f"{stem}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "44100",       # 44.1 kHz
            "-ac", "1",           # mono
            "-sample_fmt", "s16", # 16-bit PCM
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
    # Step 2: Demucs vocal isolation
    # ------------------------------------------------------------------

    def _demucs_isolate(self, wav_path: str) -> str:
        stem = Path(wav_path).stem
        # Demucs outputs to: {output_dir}/htdemucs/{stem}/vocals.wav
        out_dir = self.VOCALS_DIR
        cmd = [
            "python", "-m", "demucs",
            "--two-stems", "vocals",
            "-n", "htdemucs",
            "-o", out_dir,
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PreprocessingError(
                f"Demucs failed for {wav_path}:\n{result.stderr}"
            )

        vocal_path = os.path.join(out_dir, "htdemucs", stem, "vocals.wav")
        if not os.path.exists(vocal_path):
            raise PreprocessingError(
                f"Demucs output not found at expected path: {vocal_path}"
            )

        print(f"[Preprocess] Vocals isolated: {vocal_path}")
        return vocal_path

    # ------------------------------------------------------------------
    # Step 3: WebRTC VAD segmentation
    # ------------------------------------------------------------------

    def _vad_segment(self, wav_path: str) -> list:
        import webrtcvad

        vad = webrtcvad.Vad(3)  # aggressiveness=3 (most aggressive)
        audio, sample_rate, num_channels = self._read_wav_for_vad(wav_path)

        # WebRTC VAD requires 8000, 16000, 32000, or 48000 Hz mono
        # Resample if needed
        if sample_rate not in (8000, 16000, 32000, 48000) or num_channels != 1:
            wav_path = self._resample_for_vad(wav_path)
            audio, sample_rate, num_channels = self._read_wav_for_vad(wav_path)

        frame_duration_ms = 30  # 10, 20, or 30 ms supported
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample

        frames = []
        for i in range(0, len(audio) - frame_bytes + 1, frame_bytes):
            frames.append(audio[i : i + frame_bytes])

        # Sliding window VAD
        voiced_flags = []
        for frame in frames:
            if len(frame) < frame_bytes:
                voiced_flags.append(False)
                continue
            try:
                is_speech = vad.is_speech(frame, sample_rate)
            except Exception:
                is_speech = False
            voiced_flags.append(is_speech)

        # Group consecutive voiced frames into segments
        segments = []
        in_segment = False
        seg_start = 0

        for i, voiced in enumerate(voiced_flags):
            if voiced and not in_segment:
                in_segment = True
                seg_start = i
            elif not voiced and in_segment:
                in_segment = False
                seg_end = i
                segments.append((seg_start, seg_end))

        if in_segment:
            segments.append((seg_start, len(voiced_flags)))

        stem = Path(wav_path).stem
        saved_paths = []
        for idx, (start_frame, end_frame) in enumerate(segments):
            start_byte = start_frame * frame_bytes
            end_byte = end_frame * frame_bytes
            seg_audio = audio[start_byte:end_byte]

            duration_s = len(seg_audio) / (sample_rate * 2)
            if duration_s < self.MIN_SEGMENT_SECONDS:
                continue  # discard short segments

            out_path = os.path.join(
                self.SEGMENTS_DIR, f"{stem}_seg{idx:04d}.wav"
            )
            self._write_wav(out_path, seg_audio, sample_rate)
            saved_paths.append(out_path)

        print(
            f"[Preprocess] VAD segmented {Path(wav_path).name}: "
            f"{len(saved_paths)} segments kept"
        )
        return saved_paths

    # ------------------------------------------------------------------
    # WAV helpers
    # ------------------------------------------------------------------

    def _read_wav_for_vad(self, wav_path: str):
        with wave.open(wav_path, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        return raw, sample_rate, num_channels

    def _resample_for_vad(self, wav_path: str) -> str:
        """Resample to 16kHz mono for VAD compatibility."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise PreprocessingError(
                f"Resample for VAD failed:\n{result.stderr}"
            )
        return tmp.name

    def _write_wav(self, path: str, raw_audio: bytes, sample_rate: int):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(raw_audio)

    def _wav_duration(self, wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wf:
            return wf.getnframes() / wf.getframerate()

    def _validate_formats(self, file_paths: list):
        for fp in file_paths:
            ext = Path(fp).suffix.lower()
            if ext not in self.SUPPORTED_FORMATS:
                raise PreprocessingError(
                    f"Unsupported format: {ext}. "
                    f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
                )
            if not os.path.exists(fp):
                raise PreprocessingError(f"File not found: {fp}")
