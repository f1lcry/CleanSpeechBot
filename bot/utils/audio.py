"""Audio processing utilities."""
from __future__ import annotations

import logging
import subprocess
import uuid
import wave
from contextlib import closing
from pathlib import Path


logger = logging.getLogger("bot.audio")


class AudioProcessor:
    """Utility class responsible for audio conversion and validation tasks."""

    def __init__(self, tmp_dir: Path) -> None:
        """Initialize the processor with a temporary directory."""

        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_wav(self, source_path: Path, output_path: Path | None = None) -> Path:
        """Convert the input audio file to WAV format using FFmpeg.

        Parameters
        ----------
        source_path:
            Path to the original audio file (e.g. ``.ogg``) that needs conversion.
        output_path:
            Optional explicit destination for the converted ``.wav`` file. When omitted
            a unique filename is created within ``self.tmp_dir``.
        """

        source_path = Path(source_path)
        if not source_path.exists():
            logger.error("Audio source does not exist: %s", source_path)
            raise FileNotFoundError(source_path)

        if output_path is None:
            unique_name = f"{source_path.stem}_{uuid.uuid4().hex}.wav"
            output_path = self.tmp_dir / unique_name
        else:
            output_path = Path(output_path)
            if output_path.suffix.lower() != ".wav":
                output_path = output_path.with_name(f"{output_path.stem}.wav")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ]

        logger.info("Converting audio via FFmpeg: %s -> %s", source_path, output_path)

        try:
            subprocess.run(command, check=True, capture_output=True)
        except FileNotFoundError as exc:
            logger.exception("FFmpeg binary is missing. Please install FFmpeg.")
            raise RuntimeError("FFmpeg is required for audio conversion.") from exc
        except subprocess.CalledProcessError as exc:
            logger.exception("FFmpeg failed to convert %s", source_path)
            raise RuntimeError(
                f"FFmpeg failed to convert {source_path.name}: {exc.stderr.decode('utf-8', errors='ignore')}"
            ) from exc

        return output_path

    def validate_audio(self, audio_path: Path) -> None:
        """Validate audio file properties to ensure compatibility with Whisper."""

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.error("Audio file does not exist: %s", audio_path)
            raise FileNotFoundError(audio_path)

        if audio_path.suffix.lower() != ".wav":
            logger.error("Audio file must be in WAV format: %s", audio_path)
            raise ValueError("Audio file must be a WAV file.")

        try:
            with closing(wave.open(str(audio_path), "rb")) as wav_file:
                frame_rate = wav_file.getframerate()
                frame_count = wav_file.getnframes()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
        except (wave.Error, OSError) as exc:
            logger.exception("Unable to read WAV metadata: %s", audio_path)
            raise ValueError(f"Invalid WAV file: {audio_path}") from exc

        duration = frame_count / float(frame_rate or 1)

        if duration <= 0:
            raise ValueError("Audio duration must be greater than zero seconds.")

        if frame_rate != 16000:
            raise ValueError("Audio sample rate must be 16 kHz for Whisper.")

        if channels != 1:
            raise ValueError("Audio must be mono.")

        if sample_width * 8 != 16:
            raise ValueError("Audio must use 16-bit samples.")

        logger.debug(
            "Validated audio file: path=%s duration=%.2fs rate=%sHz channels=%s", audio_path, duration, frame_rate, channels
        )

    def cleanup(self, audio_path: Path) -> None:
        """Remove temporary audio artifacts after processing."""

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.debug("Temporary audio file already removed: %s", audio_path)
            return

        if audio_path.is_dir():
            logger.warning("Skipping cleanup for directory path: %s", audio_path)
            return

        try:
            audio_path.unlink()
            logger.debug("Removed temporary audio file: %s", audio_path)
        except OSError:
            logger.exception("Failed to remove temporary audio file: %s", audio_path)
