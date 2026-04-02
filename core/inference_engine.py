"""
RVC Inference Engine
Performs audio-to-audio voice conversion using trained .pth models.
Follows Applio's VoiceConverter / infer.py pattern.
"""

import os
import sys
from pathlib import Path


MODELS_BASE = os.path.expanduser("~/VoiceClone/models")

# Applio is cloned at rvc/ — add it to sys.path so `from rvc.infer.infer import ...` works
_APPLIO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rvc"))
if os.path.isdir(_APPLIO_ROOT) and _APPLIO_ROOT not in sys.path:
    sys.path.insert(0, _APPLIO_ROOT)


class InferenceEngine:

    def __init__(self):
        self._converter = None  # lazy-loaded VoiceConverter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        model_path: str,
        input_audio_path: str,
        output_path: str,
        pitch_shift: int = 0,
    ) -> str:
        """
        Convert a source audio file to the target voice.

        Args:
            model_path:        Path to the trained .pth file.
            input_audio_path:  Source audio to convert (any format).
            output_path:       Where to save the converted WAV.
            pitch_shift:       Semitones to shift pitch (0 = no shift).

        Returns:
            Absolute path to the output WAV file.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Applio uses relative paths (e.g. 'rvc/models/predictors/rmvpe.pt')
        # for lazy-loaded resources during conversion, so we must run from
        # APPLIO_ROOT for the entire duration of the convert call.
        old_cwd = os.getcwd()
        try:
            os.chdir(_APPLIO_ROOT)
            converter = self._get_converter()
            converter.convert_audio(
                audio_input_path=os.path.abspath(os.path.join(old_cwd, input_audio_path))
                    if not os.path.isabs(input_audio_path) else input_audio_path,
                audio_output_path=os.path.abspath(os.path.join(old_cwd, output_path))
                    if not os.path.isabs(output_path) else output_path,
                model_path=os.path.abspath(model_path),
                index_path=self._find_index(model_path),
                pitch=pitch_shift,
                f0_method="rmvpe",
                index_rate=0.75,
                volume_envelope=1.0,
                protect=0.5,
                hop_length=128,
                split_audio=False,
                f0_autotune=False,
                embedder_model="contentvec",
                clean_audio=True,
                export_format="WAV",
                post_process=False,
                resample_sr=0,
                sid=0,
            )
        finally:
            os.chdir(old_cwd)

        print(f"[Inference] Converted: {input_audio_path} → {output_path}")
        return os.path.abspath(output_path)

    def list_models(self) -> list:
        """
        Scan ~/VoiceClone/models/ recursively and return all .pth file paths.
        """
        if not os.path.isdir(MODELS_BASE):
            return []
        return sorted(str(p) for p in Path(MODELS_BASE).rglob("*.pth"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_converter(self):
        """Lazy-load VoiceConverter from the rvc.infer module."""
        if self._converter is not None:
            return self._converter

        # VoiceConverter's Config() opens 'rvc/configs/48000.json' as a
        # relative path, so we must run from APPLIO_ROOT (e.g. ./rvc/).
        old_cwd = os.getcwd()
        try:
            os.chdir(_APPLIO_ROOT)
            from rvc.infer.infer import VoiceConverter
            self._converter = VoiceConverter()
            return self._converter
        except ImportError:
            raise ImportError(
                "Could not import rvc.infer.infer.VoiceConverter. "
                "Ensure Applio's rvc/ directory is on PYTHONPATH or cloned "
                "alongside this project at ./rvc/"
            )
        finally:
            os.chdir(old_cwd)

    def _find_index(self, model_path: str) -> str:
        """
        Look for a .index file next to the .pth model.
        Returns empty string if not found (inference still works without it).
        """
        model_dir = os.path.dirname(model_path)
        for f in os.listdir(model_dir):
            if f.endswith(".index"):
                return os.path.join(model_dir, f)

        # Also check ~/VoiceClone/models/ root
        model_stem = Path(model_path).stem
        candidate = os.path.join(MODELS_BASE, f"{model_stem}.index")
        if os.path.exists(candidate):
            return candidate

        return ""  # no index — inference proceeds without FAISS matching
