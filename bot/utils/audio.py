"""Audio processing utilities."""
from __future__ import annotations

from pathlib import Path


class AudioProcessor:
    """Utility class responsible for audio conversion and validation tasks."""

    def __init__(self, tmp_dir: Path) -> None:
        """Initialize the processor with a temporary directory."""

        self.tmp_dir = tmp_dir

    def convert_to_wav(self, source_path: Path) -> Path:
        """Convert the input audio file to WAV format.

        TODO: Implement conversion via FFmpeg ensuring consistent sample rate and codec.
        """

        raise NotImplementedError("Audio conversion is not yet implemented.")

    def validate_audio(self, audio_path: Path) -> None:
        """Validate audio file properties to ensure compatibility with Whisper.

        TODO: Inspect duration, bitrate, and supported format.
        """

        raise NotImplementedError("Audio validation is not yet implemented.")

    def cleanup(self, audio_path: Path) -> None:
        """Remove temporary audio artifacts after processing.

        TODO: Ensure safe deletion and handle potential filesystem errors.
        """

        raise NotImplementedError("Audio cleanup is not yet implemented.")
