"""
RVC Inference Engine
Performs audio-to-audio voice conversion using trained .pth models.

chdir-free: All Applio relative-path lookups are patched to absolute paths
at module import time via _bootstrap_applio(). No global os.chdir anywhere.
"""

import json
import os
import sys
from pathlib import Path

from core.logger import get_logger

MODELS_BASE  = os.path.expanduser("~/VoiceClone/models")
_APPLIO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rvc"))
_bootstrapped = False


def _bootstrap_applio() -> None:
    """
    Import Applio modules and patch every relative-path lookup to use
    absolute paths based on _APPLIO_ROOT.  Called once at module load.
    No os.chdir() is used anywhere after this function returns.
    """
    global _bootstrapped
    if _bootstrapped or not os.path.isdir(_APPLIO_ROOT):
        return

    log = get_logger(stage="bootstrap")

    if _APPLIO_ROOT not in sys.path:
        sys.path.insert(0, _APPLIO_ROOT)

    import importlib

    # ── utils.py: now_dir is captured at module import time ──────────
    try:
        utils_mod = importlib.import_module("rvc.lib.utils")
        utils_mod.now_dir   = _APPLIO_ROOT
        utils_mod.base_path = os.path.join(
            _APPLIO_ROOT, "rvc", "models", "formant", "stftpitchshift"
        )
        utils_mod.stft = utils_mod.base_path   # Linux (no .exe suffix)
        log.info("patched rvc.lib.utils paths")
    except Exception as exc:
        log.warning(f"Could not patch rvc.lib.utils: {exc}")

    # ── config.py: Config.load_config_json uses relative paths ───────
    try:
        cfg_mod = importlib.import_module("rvc.configs.config")

        # The Config class is wrapped by @singleton; extract the real class
        # from the closure so we can patch its method before first instantiation.
        config_fn    = cfg_mod.Config          # this is get_instance
        actual_cls   = None
        freevars     = list(config_fn.__code__.co_freevars)
        if "cls" in freevars and config_fn.__closure__:
            cell       = config_fn.__closure__[freevars.index("cls")]
            actual_cls = cell.cell_contents

        if actual_cls is not None:
            _root = _APPLIO_ROOT

            def _abs_load_config_json(self):
                configs = {}
                for config_file in cfg_mod.version_config_paths:
                    config_path = os.path.join(
                        _root, "rvc", "configs", config_file
                    )
                    with open(config_path, "r", encoding="utf-8") as fh:
                        configs[config_file] = json.load(fh)
                return configs

            actual_cls.load_config_json = _abs_load_config_json
            log.info("patched Config.load_config_json")

        # Force singleton creation NOW so it runs with our patched method.
        cfg_mod.Config()
        log.info("Config singleton initialised")
    except Exception as exc:
        log.warning(f"Could not patch rvc.configs.config: {exc}")

    # ── f0.py: RMVPE and FCPE use relative model paths ───────────────
    try:
        f0_mod = importlib.import_module("rvc.lib.predictors.f0")
        _root  = _APPLIO_ROOT

        def _abs_rmvpe_init(
            self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160
        ):
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor
            self.device      = device
            self.sample_rate = sample_rate
            self.hop_size    = hop_size
            self.model       = RMVPE0Predictor(
                os.path.join(_root, "rvc", "models", "predictors", model_name),
                device=self.device,
            )

        f0_mod.RMVPE.__init__ = _abs_rmvpe_init

        def _abs_fcpe_init(self, device, sample_rate=16000, hop_size=160):
            from torchfcpe import spawn_infer_model_from_pt
            self.device      = device
            self.sample_rate = sample_rate
            self.hop_size    = hop_size
            self.model       = spawn_infer_model_from_pt(
                os.path.join(_root, "rvc", "models", "predictors", "fcpe.pt"),
                self.device,
                bundled_model=True,
            )

        f0_mod.FCPE.__init__ = _abs_fcpe_init
        log.info("patched RMVPE and FCPE paths")
    except Exception as exc:
        log.warning(f"Could not patch rvc.lib.predictors.f0: {exc}")

    _bootstrapped = True
    log.success("Applio bootstrap complete — no chdir required for inference")


# Run once at module import time
_bootstrap_applio()


class InferenceEngine:

    def __init__(self):
        self._converter = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        model_path:       str,
        input_audio_path: str,
        output_path:      str,
        pitch_shift:      int = 0,
    ) -> str:
        """
        Convert source audio to target voice.  No os.chdir() used.

        Args:
            model_path:        Absolute or relative path to the .pth file.
            input_audio_path:  Source audio (any format).
            output_path:       Destination WAV path.
            pitch_shift:       Semitones (0 = no shift).

        Returns:
            Absolute path to the output WAV.
        """
        log = get_logger(stage="infer")

        # Resolve all paths to absolute before calling Applio
        model_path       = os.path.abspath(model_path)
        input_audio_path = os.path.abspath(input_audio_path)
        output_path      = os.path.abspath(output_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio not found: {input_audio_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with log.timed(f"convert {Path(input_audio_path).name}"):
            converter = self._get_converter()
            converter.convert_audio(
                audio_input_path=input_audio_path,
                audio_output_path=output_path,
                model_path=model_path,
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

        return output_path

    def list_models(self) -> list:
        if not os.path.isdir(MODELS_BASE):
            return []
        return sorted(str(p) for p in Path(MODELS_BASE).rglob("*.pth"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_converter(self):
        if self._converter is not None:
            return self._converter
        try:
            from rvc.infer.infer import VoiceConverter
            self._converter = VoiceConverter()
            return self._converter
        except ImportError:
            raise ImportError(
                "Could not import rvc.infer.infer.VoiceConverter. "
                "Ensure Applio is cloned at ./rvc/."
            )

    def _find_index(self, model_path: str) -> str:
        model_dir = os.path.dirname(model_path)
        for f in os.listdir(model_dir):
            if f.endswith(".index"):
                return os.path.join(model_dir, f)
        model_stem = Path(model_path).stem
        candidate  = os.path.join(MODELS_BASE, f"{model_stem}.index")
        if os.path.exists(candidate):
            return candidate
        return ""
