from __future__ import annotations

import os
import sys
from typing import TextIO


class Tee:
    """File-like stream that mirrors stdout/stderr to a log file."""

    def __init__(self, stream: TextIO, log_path: str) -> None:
        self.stream = stream
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self._log = open(log_path, "a", encoding="utf-8", buffering=1)

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self._log.write(text)
        return written

    def flush(self) -> None:
        self.stream.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> str | None:
        return self.stream.encoding


def configure_run_logging(log_path: str) -> str:
    """Mirror all subsequent console output to ``log_path``."""

    sys.stdout = Tee(sys.stdout, log_path)
    sys.stderr = Tee(sys.stderr, log_path)
    return log_path
