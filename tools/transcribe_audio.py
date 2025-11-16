"""Command-line utility for transcribing WAV files via WhisperEngine."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import _env  # noqa: F401  # Ensure .env is loaded and repo root is on sys.path
from tools._env import REPO_ROOT

from bot.utils.whisper_engine import WhisperEngine


DEFAULT_AUDIO_ROOT = Path(os.environ.get("AUDIO_TMP_DIR", "/tmp/botsummarizer"))
# Conversion utilities and the runtime pipeline both write WAV files directly into
# AUDIO_TMP_DIR. Keep the CLI default aligned so users can run convert ->
# transcribe without overriding flags.
DEFAULT_TMP_DIR = DEFAULT_AUDIO_ROOT
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "medium")
DEFAULT_CACHE_DIR = Path(os.environ.get("WHISPER_CACHE_DIR", REPO_ROOT / "models/whisper_cache"))
DEFAULT_CA_BUNDLE = os.environ.get("WHISPER_CA_BUNDLE")
DEFAULT_INSECURE_SSL = os.environ.get("WHISPER_INSECURE_SSL", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the transcription CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Transcribe WAV files that were prepared for Whisper. "
            "When --input is omitted the tool picks the newest .wav from the temporary directory."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to a WAV file. "
            "When omitted, the latest file in --tmp-dir (default AUDIO_TMP_DIR) is used."
        ),
    )
    parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=DEFAULT_TMP_DIR,
        help="Directory that stores WAV files (defaults to AUDIO_TMP_DIR).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Whisper model name to load (defaults to WHISPER_MODEL or 'base').",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language hint passed to Whisper (e.g. 'ru', 'en').",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Whisper decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device identifier for torch (e.g. 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging output.",
    )
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=Path(DEFAULT_CA_BUNDLE).expanduser() if DEFAULT_CA_BUNDLE else None,
        help="Optional CA bundle for verifying Whisper download SSL certificates.",
    )
    parser.add_argument(
        "--insecure-ssl",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_INSECURE_SSL,
        help="Disable SSL verification when downloading Whisper models (use as a last resort).",
    )
    return parser


def pick_latest_wav(search_dir: Path) -> Path:
    """Return the most recently modified WAV file in the directory."""

    if not search_dir.exists():
        raise FileNotFoundError(f"Temporary directory does not exist: {search_dir}")

    wav_files = sorted(search_dir.glob("*.wav"), key=lambda item: item.stat().st_mtime)
    if not wav_files:
        raise FileNotFoundError(
            f"No WAV files found in {search_dir}. Convert audio first via tools/convert_audio.py."
        )

    return wav_files[-1]


def resolve_audio_path(candidate: Path | None, tmp_dir: Path) -> Path:
    """Resolve the audio path depending on CLI arguments."""

    if candidate is not None:
        audio_path = Path(candidate)
        if not audio_path.exists():
            raise FileNotFoundError(f"Input file does not exist: {audio_path}")
        return audio_path

    return pick_latest_wav(tmp_dir)


def main() -> int:
    """Execute the transcription CLI and return the exit status."""

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        audio_path = resolve_audio_path(args.input, args.tmp_dir)
        engine = WhisperEngine(
            model_name=args.model,
            cache_dir=DEFAULT_CACHE_DIR,
            language=args.language,
            temperature=args.temperature,
            device=args.device,
            ssl_cert_file=args.ca_bundle,
            allow_insecure_ssl=args.insecure_ssl,
        )
        transcript = engine.transcribe(audio_path=audio_path)
    except Exception as exc:  # noqa: BLE001 - bubble up to the CLI output
        logging.error("Transcription failed: %s", exc)
        return 1

    print(f"[TRANSCRIPT] {transcript}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
