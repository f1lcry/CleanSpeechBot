"""Whisper transcription engine wrapper."""
from __future__ import annotations

import logging
import ssl
import threading
import time
import warnings
import wave
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import HTTPSHandler, build_opener, install_opener

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
        ssl_cert_file: Path | None = None,
        allow_insecure_ssl: bool = False,
    ) -> None:
        """Initialize the engine configuration without loading the model eagerly."""

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.language = language
        self.temperature = temperature
        self.device = device
        self.ssl_cert_file = Path(ssl_cert_file).expanduser() if ssl_cert_file else None
        self.allow_insecure_ssl = allow_insecure_ssl

        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._ssl_context_lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._ssl_context_installed = False
        self._warnings_configured = False
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.ssl_cert_file is not None and not self.ssl_cert_file.is_file():
            raise FileNotFoundError(
                f"Configured Whisper SSL certificate bundle does not exist: {self.ssl_cert_file}"
            )

    def _install_ssl_context(self) -> None:
        """Install a custom SSL context when certificates need overriding."""

        if self._ssl_context_installed:
            return

        if not (self.ssl_cert_file or self.allow_insecure_ssl):
            return

        with self._ssl_context_lock:
            if self._ssl_context_installed:
                return

            if self.allow_insecure_ssl:
                logger.warning(
                    "Using insecure SSL context for Whisper downloads. Certificates will not be verified."
                )
                context = ssl._create_unverified_context()
            else:
                logger.info("Installing custom SSL CA bundle for Whisper downloads: %s", self.ssl_cert_file)
                context = ssl.create_default_context(cafile=str(self.ssl_cert_file))

            https_handler = HTTPSHandler(context=context)
            opener = build_opener(https_handler)
            install_opener(opener)
            self._ssl_context_installed = True

    def _ensure_model(self) -> Any:
        """Load the Whisper model lazily in a thread-safe fashion."""

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is None:
                try:
                    self._install_ssl_context()
                    self._configure_runtime()
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
                    download_hint = ""
                    underlying = exc
                    if isinstance(exc, URLError):
                        underlying = exc.reason or exc
                    if isinstance(underlying, ssl.SSLError):
                        download_hint = (
                            " SSL handshake failed. Provide a trusted CA bundle via WHISPER_CA_BUNDLE "
                            "or rerun the CLI with --ca-bundle/--insecure-ssl."
                        )

                    raise RuntimeError(
                        f"Failed to load Whisper model '{self.model_name}'.{download_hint}"
                    ) from exc

        return self._model

    def _configure_runtime(self) -> None:
        """Suppress noisy warnings emitted by upstream libraries once per process."""

        if self._warnings_configured:
            return

        with self._runtime_lock:
            if self._warnings_configured:
                return

            warnings.filterwarnings(
                "ignore",
                message="TypedStorage is deprecated",
                category=UserWarning,
                module="torch._utils",
            )
            warnings.filterwarnings(
                "ignore",
                message="FP16 is not supported on CPU; using FP32 instead",
                category=UserWarning,
                module="whisper.transcribe",
            )

            self._warnings_configured = True

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
