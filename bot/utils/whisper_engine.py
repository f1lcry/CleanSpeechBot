"""Whisper transcription engine wrapper."""
from __future__ import annotations

import argparse
import logging
import os
import ssl
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any

import whisper

logger = logging.getLogger("bot.whisper")


@contextmanager
def _temporary_urlopen_context(context: ssl.SSLContext | None):
    """Install a temporary urllib opener that respects the provided SSL context."""

    if context is None:
        yield
        return

    handler = urllib.request.HTTPSHandler(context=context)
    opener = urllib.request.build_opener(handler)
    previous = urllib.request._opener
    urllib.request.install_opener(opener)
    try:
        yield
    finally:
        if previous is None:
            urllib.request.install_opener(urllib.request.build_opener())
        else:
            urllib.request.install_opener(previous)


class WhisperEngine:
    """Wrapper around the Whisper model used for audio transcription."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        *,
        cafile: Path | None = None,
        allow_insecure_download: bool = False,
    ) -> None:
        """Initialize the engine with the given model name and cache directory."""

        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cafile = cafile
        self.allow_insecure_download = allow_insecure_download
        self._model: Any | None = None
        self._lock = Lock()

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        if self.cafile:
            return ssl.create_default_context(cafile=str(self.cafile))
        if self.allow_insecure_download:
            logger.warning("Whisper model download will skip SSL verification.")
            return ssl._create_unverified_context()
        return None

    def _load_model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info("Loading Whisper model %s", self.model_name)
                    ssl_context = self._build_ssl_context()
                    with _temporary_urlopen_context(ssl_context):
                        self._model = whisper.load_model(self.model_name, download_root=str(self.cache_dir))
        return self._model

    def transcribe(self, audio_path: Path, *, language: str | None = None, temperature: float | None = None) -> str:
        """Transcribe the provided audio file using the configured Whisper model."""

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file for transcription not found: {audio_path}")

        model = self._load_model()
        options: dict[str, Any] = {}
        if language:
            options["language"] = language
        if temperature is not None:
            options["temperature"] = temperature

        start = time.perf_counter()
        result = model.transcribe(str(audio_path), **options)
        duration = time.perf_counter() - start
        logger.info("Whisper transcription finished in %.2fs", duration)

        text = result.get("text", "") if isinstance(result, dict) else str(result)
        return text.strip()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run Whisper transcription standalone.")
    parser.add_argument("--audio", required=True, type=Path, help="Path to the WAV audio file.")
    parser.add_argument(
        "--model",
        default=os.environ.get("WHISPER_MODEL", "medium"),
        help="Name of the Whisper model to use (default from WHISPER_MODEL env).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/whisper_cache"),
        help="Directory used to cache Whisper models.",
    )
    parser.add_argument(
        "--cafile",
        type=Path,
        help="Path to a custom CA bundle for environments with self-signed certificates.",
    )
    parser.add_argument(
        "--allow-insecure-download",
        action="store_true",
        help="Disable SSL verification for model downloads (use only in trusted networks).",
    )
    args = parser.parse_args()

    engine = WhisperEngine(
        model_name=args.model,
        cache_dir=args.cache_dir,
        cafile=args.cafile,
        allow_insecure_download=args.allow_insecure_download,
    )
    text = engine.transcribe(args.audio)
    print(text)


if __name__ == "__main__":
    _cli()
