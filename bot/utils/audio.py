"""Audio processing utilities."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import uuid
import wave
from contextlib import closing
from pathlib import Path

logger = logging.getLogger("bot.audio")


class AudioProcessor:
    """Utility class responsible for audio conversion and validation tasks."""

    MAX_DURATION_SECONDS = 300  # Whisper inference is limited to ~5 минут.
    MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 МБ на одно сообщение.

    def __init__(self, tmp_dir: Path, ffmpeg_binary: str | Path | None = None) -> None:
        """Initialize the processor with a temporary directory and FFmpeg binary."""

        self.tmp_dir = tmp_dir
        binary = str(ffmpeg_binary) if ffmpeg_binary else os.environ.get("FFMPEG_BINARY", "ffmpeg")
        self.ffmpeg_binary = binary

    def _resolve_ffmpeg_binary(self) -> str:
        """Return the FFmpeg binary path or raise a helpful error if missing."""

        resolved = shutil.which(self.ffmpeg_binary)
        if resolved:
            return resolved

        raise RuntimeError(
            "FFmpeg не найден. Установите ffmpeg (https://ffmpeg.org/) или укажите путь через FFMPEG_BINARY."
        )

    def convert_to_wav(self, source_path: Path) -> Path:
        """Convert the input audio file to WAV format using FFmpeg."""

        if not source_path.exists():
            raise FileNotFoundError(f"Source audio file not found: {source_path}")

        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        output_name = f"{source_path.stem}-{uuid.uuid4().hex}.wav"
        output_path = self.tmp_dir / output_name

        command = [
            self._resolve_ffmpeg_binary(),
            "-y",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(output_path),
        ]

        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"FFmpeg failed to convert audio: {exc}") from exc

        return output_path

    def validate_audio(self, audio_path: Path) -> None:
        """Validate audio file properties to ensure compatibility with Whisper."""

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file missing: {audio_path}")

        if audio_path.suffix.lower() != ".wav":
            raise ValueError("Audio file must be in WAV format after conversion.")

        size = audio_path.stat().st_size
        if size <= 0:
            raise ValueError("Audio file is empty after conversion.")
        if size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError("Audio file exceeds the maximum allowed size for processing.")

        try:
            with closing(wave.open(str(audio_path), "rb")) as wav_file:
                frames = wav_file.getnframes()
                frame_rate = wav_file.getframerate()
                if frame_rate <= 0:
                    raise ValueError("Invalid sample rate detected in WAV file.")
                duration = frames / float(frame_rate)
        except wave.Error as exc:
            raise ValueError("WAV файл повреждён или имеет неверный формат.") from exc

        if duration <= 0:
            raise ValueError("Audio duration is zero after conversion.")
        if duration > self.MAX_DURATION_SECONDS:
            raise ValueError("Audio duration exceeds the allowed limit for Whisper processing.")

    def cleanup(self, audio_path: Path) -> None:
        """Remove temporary audio artifacts after processing."""

        try:
            if audio_path.exists():
                if audio_path.is_file():
                    audio_path.unlink()
                elif audio_path.is_dir():
                    shutil.rmtree(audio_path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to cleanup temporary audio path %s: %s", audio_path, exc)
        # Каталог создаётся перед началом обработки и переиспользуется.
        # Удалять его агрессивно не нужно, чтобы не мешать параллельным задачам.


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Convert audio files to WAV for Whisper testing.")
    parser.add_argument("--src", required=True, type=Path, help="Path to the input OGG/MP3/WAV file.")
    parser.add_argument("--dst", required=True, type=Path, help="Destination path for the WAV output.")
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=Path(os.environ.get("AUDIO_TMP_DIR", "/tmp/botsummarizer")),
        help="Temporary directory used for intermediate WAV files.",
    )
    parser.add_argument(
        "--ffmpeg-binary",
        type=str,
        default=os.environ.get("FFMPEG_BINARY"),
        help="Path to ffmpeg if it is not available in PATH.",
    )
    args = parser.parse_args()

    processor = AudioProcessor(tmp_dir=args.tmp_dir, ffmpeg_binary=args.ffmpeg_binary)
    try:
        converted = processor.convert_to_wav(args.src)
    except RuntimeError as exc:  # pragma: no cover - CLI helper
        parser.error(str(exc))
    try:
        processor.validate_audio(converted)
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(converted), str(args.dst))
        print(f"WAV файл сохранён: {args.dst}")
    finally:
        processor.cleanup(converted)


if __name__ == "__main__":
    _cli()
