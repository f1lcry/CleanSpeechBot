"""Command-line interface for formatting transcripts via the local LLM."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from tools import _env  # noqa: F401  # Ensure .env is respected for defaults

from bot.utils.formatting_llm import FormattingLLMClient, load_formatter_prompt

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("FORMATTER_MODEL", "llama3.1:8b")
DEFAULT_TIMEOUT = float(os.environ.get("FORMATTER_TIMEOUT", "120"))


def build_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the formatting CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Send raw transcripts to the local Ollama instance and receive formatted text. "
            "Arguments mirror the bot runtime so results stay consistent."
        )
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--text",
        type=str,
        help="Inline transcript text that should be formatted.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a file that contains the transcript to format.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Ollama host URL. Defaults to OLLAMA_HOST or http://localhost:11434.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name registered in Ollama (defaults to FORMATTER_MODEL).",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        type=str,
        help="Inline system prompt override passed directly to the formatter.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing the system prompt override.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for the Ollama request (defaults to FORMATTER_TIMEOUT).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging for troubleshooting formatter calls.",
    )
    return parser


def _read_transcript(args: argparse.Namespace) -> str:
    """Return transcript text from CLI arguments or stdin."""

    if args.text is not None:
        return args.text
    if args.input_file is not None:
        path = Path(args.input_file)
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        return path.read_text(encoding="utf-8")
    if not sys.stdin.isatty():
        data = sys.stdin.read()
        if data.strip():
            return data
    raise RuntimeError("No transcript provided. Use --text, --input-file, or pipe stdin.")


def _resolve_prompt(args: argparse.Namespace) -> str:
    """Resolve which system prompt should be used for formatting."""

    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        path = Path(args.prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file does not exist: {path}")
        return path.read_text(encoding="utf-8")
    return load_formatter_prompt()


async def _run_formatter(args: argparse.Namespace) -> str:
    """Instantiate the client and run the formatting request."""

    transcript = _read_transcript(args)
    system_prompt = _resolve_prompt(args)
    client = FormattingLLMClient(
        host=args.host,
        model=args.model,
        system_prompt=system_prompt,
        request_timeout=args.timeout,
    )
    return await client.format_text(transcript)


def main() -> int:
    """Entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        result = asyncio.run(_run_formatter(args))
    except Exception as exc:  # noqa: BLE001 - surface CLI-friendly errors
        logging.error("Formatting failed: %s", exc)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
