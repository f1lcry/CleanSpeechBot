"""Whisper transcription engine wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class WhisperEngine:
    """Wrapper around the Whisper model used for audio transcription."""

    def __init__(self, model_name: str, cache_dir: Path) -> None:
        """Initialize the engine with the given model name and cache directory."""

        self.model_name = model_name
        self.cache_dir = cache_dir

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe the provided audio file using the configured Whisper model.

        TODO: Load Whisper model lazily and run transcription with caching support.
        """

        raise NotImplementedError("Whisper transcription is not yet implemented.")
