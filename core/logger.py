"""
Structured pipeline logger.
Format: HH:MM:SS [FILE: xyz.mp3] [STAGE: preprocess] [STATUS: success] time=3.2s
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

_logger = logging.getLogger("vcs.pipeline")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False


class StageLogger:
    """Logger bound to a (file, stage) pair."""

    def __init__(self, file_name: str = "", stage: str = ""):
        self.file_name = Path(file_name).name if file_name else ""
        self.stage     = stage

    def _fmt(self, status: str, msg: str = "", elapsed: float | None = None) -> str:
        parts = []
        if self.file_name:
            parts.append(f"[FILE: {self.file_name}]")
        if self.stage:
            parts.append(f"[STAGE: {self.stage}]")
        parts.append(f"[STATUS: {status}]")
        if elapsed is not None:
            parts.append(f"time={elapsed:.2f}s")
        if msg:
            parts.append(msg)
        return " ".join(parts)

    def info(self, msg: str = "", elapsed: float | None = None):
        _logger.info(self._fmt("info", msg, elapsed))

    def success(self, msg: str = "", elapsed: float | None = None):
        _logger.info(self._fmt("success", msg, elapsed))

    def warning(self, msg: str = "", elapsed: float | None = None):
        _logger.warning(self._fmt("warning", msg, elapsed))

    def error(self, msg: str, elapsed: float | None = None):
        _logger.error(self._fmt("error", msg, elapsed))

    @contextmanager
    def timed(self, label: str = ""):
        """Logs success+elapsed on exit, error on exception."""
        t0 = time.perf_counter()
        try:
            yield
            self.success(label, time.perf_counter() - t0)
        except Exception as exc:
            self.error(str(exc), time.perf_counter() - t0)
            raise


def get_logger(file_name: str = "", stage: str = "") -> StageLogger:
    return StageLogger(file_name, stage)
