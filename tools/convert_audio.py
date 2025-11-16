"""Command-line utility for converting Telegram voice notes to WAV."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from tools import _env  # noqa: F401  # Ensure .env is loaded and repo root is on sys.path

from bot.utils.audio import AudioProcessor


DEFAULT_TMP_DIR = Path(os.environ.get("AUDIO_TMP_DIR", "/tmp/botsummarizer"))


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the conversion CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert Telegram .ogg voice messages into Whisper-friendly WAV files. "
            "Defaults mirror AUDIO_TMP_DIR from the bot configuration."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source .ogg (or other FFmpeg-supported) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Directory where the converted WAV should be stored.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional name for the converted file without extension. "
            "Defaults to the input stem with an auto-generated suffix."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted WAV to ensure Whisper compatibility.",
    )
    return parser


def resolve_output_path(output_dir: Path, output_name: str | None, input_path: Path) -> Path | None:
    """Derive the desired output path for conversion."""

    if output_name:
        safe_name = Path(output_name).stem
        return output_dir / f"{safe_name}.wav"
    if output_dir != DEFAULT_TMP_DIR:
        return output_dir / f"{input_path.stem}.wav"
    return None


def main() -> int:
    """Execute the conversion flow and return an exit status."""

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    processor = AudioProcessor(tmp_dir=args.output_dir)

    try:
        output_path = resolve_output_path(args.output_dir, args.output_name, args.input)
        converted = processor.convert_to_wav(source_path=args.input, output_path=output_path)
        if args.validate:
            processor.validate_audio(audio_path=converted)
    except Exception as exc:  # noqa: BLE001 - surface any operational issue to the CLI
        logging.error("Conversion failed: %s", exc)
        return 1

    logging.info("Converted file saved to: %s", converted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
