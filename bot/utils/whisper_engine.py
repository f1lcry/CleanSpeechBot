"""Whisper transcription engine wrapper."""
from __future__ import annotations

import logging
import threading
import time
import wave
from contextlib import closing
from pathlib import Path
from typing import Any

import whisper


logger = logging.getLogger("bot.whisper")


class WhisperEngine:
    """Wrapper around the Whisper model used for audio transcription."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Path,
        *,
        language: str | None = None,
        temperature: float = 0.0,
        device: str | None = None,
    ) -> None:
        """Initialize the engine configuration without loading the model eagerly."""

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.language = language
        self.temperature = temperature
        self.device = device

        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_model(self) -> Any:
        """Load the Whisper model lazily in a thread-safe fashion."""

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is None:
                try:
                    logger.info(
                        "Loading Whisper model '%s' (device=%s, cache=%s)",
                        self.model_name,
                        self.device or "auto",
                        self.cache_dir,
                    )
                    self._model = whisper.load_model(
                        name=self.model_name,
                        device=self.device,
                        download_root=str(self.cache_dir),
                    )
                except Exception as exc:  # noqa: BLE001 - surface real initialization issues
                    logger.exception("Failed to initialize Whisper model: %s", exc)
                    raise RuntimeError(f"Failed to load Whisper model '{self.model_name}'.") from exc

        return self._model

    def _read_duration(self, audio_path: Path) -> float:
        """Return duration of the wav file in seconds for logging."""

        try:
            with closing(wave.open(str(audio_path), "rb")) as wav_file:
                frames = wav_file.getnframes()
                frame_rate = wav_file.getframerate()
        except (wave.Error, OSError) as exc:  # pragma: no cover - defensive logging only
            logger.debug("Unable to read WAV metadata for %s: %s", audio_path, exc)
            return 0.0

        return frames / float(frame_rate or 1)

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe the provided audio file using the configured Whisper model."""

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        if audio_path.suffix.lower() != ".wav":
            raise ValueError("WhisperEngine expects a WAV input file.")

        duration = self._read_duration(audio_path)
        logger.info(
            "Transcribing %s (%.2fs) with model '%s'", audio_path.name, duration, self.model_name
        )

        model = self._ensure_model()
        started = time.perf_counter()

        try:
            result = model.transcribe(
                str(audio_path),
                language=self.language,
                temperature=self.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - whisper failure should be reported upstream
            logger.exception("Whisper transcription failed for %s", audio_path)
            raise RuntimeError(f"Whisper transcription failed for {audio_path.name}.") from exc

        elapsed = time.perf_counter() - started
        logger.info("Whisper transcription finished in %.2fs", elapsed)

        text = result.get("text", "")
        return text.strip()
