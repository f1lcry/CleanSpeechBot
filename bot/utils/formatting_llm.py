"""Client for formatting text via a local LLM served by Ollama."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, MutableMapping

from ollama import AsyncClient, RequestError, ResponseError


logger = logging.getLogger("bot.formatting")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_PATH = PROJECT_ROOT / "models" / "llama_prompts" / "formatter_system_prompt.txt"
_FILE_SOURCES = {"file", "path", "filesystem"}
_INLINE_SOURCES = {"inline", "env", "environment", "text"}


def load_formatter_prompt(
    env: Mapping[str, str | None] | None = None,
    *,
    fallback_path: Path | None = None,
) -> str:
    """Load the formatter system prompt based on environment preferences.

    The lookup order:
    1. ``FORMATTER_PROMPT_SOURCE=inline`` → ``FORMATTER_PROMPT_TEXT`` content.
    2. File-based lookup via ``FORMATTER_PROMPT_PATH`` or ``fallback_path``.
    """

    values: Mapping[str, str | None]
    values = env or os.environ
    source = (values.get("FORMATTER_PROMPT_SOURCE") or "file").strip().lower()
    fallback_path = fallback_path or DEFAULT_PROMPT_PATH

    if source in _INLINE_SOURCES:
        prompt_text = values.get("FORMATTER_PROMPT_TEXT")
        if prompt_text:
            return prompt_text.strip()
        raise RuntimeError(
            "FORMATTER_PROMPT_SOURCE is set to 'inline' but FORMATTER_PROMPT_TEXT is empty."
        )

    if source in _FILE_SOURCES:
        prompt_path_raw = values.get("FORMATTER_PROMPT_PATH")
        prompt_path = Path(prompt_path_raw).expanduser() if prompt_path_raw else fallback_path
        if prompt_path and not prompt_path.is_absolute():
            prompt_path = PROJECT_ROOT / prompt_path
        try:
            content = prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise RuntimeError(f"Formatter prompt file does not exist: {prompt_path}") from exc
        return content.strip()

    raise RuntimeError(
        "Unknown FORMATTER_PROMPT_SOURCE. Use 'file' or 'inline'."
    )


class FormattingLLMClient:
    """A lightweight client that formats transcribed text using Ollama-hosted models."""

    def __init__(
        self,
        host: str,
        model: str,
        *,
        system_prompt: str | None = None,
        prompt_path: Path | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        """Store configuration required for invoking the Ollama endpoint."""

        self.host = host.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout
        self.system_prompt = system_prompt or load_formatter_prompt(fallback_path=prompt_path)
        self._client = AsyncClient(host=self.host)

    async def _stream_response(self, stream: AsyncIterator[Mapping[str, Any]]) -> str:
        """Consume the streaming generator from Ollama and build the response text."""

        chunks: list[str] = []
        received = 0
        last_logged = 0
        async for part in stream:
            chunk = part.get("response", "") or ""
            if chunk:
                chunks.append(chunk)
                received += len(chunk)
                if received - last_logged >= 500:
                    logger.debug("Formatting stream received %d characters", received)
                    last_logged = received

            if part.get("done"):
                metrics: MutableMapping[str, Any] | None = part.get("metrics")
                if metrics:
                    logger.info(
                        "Formatter completed (eval_count=%s, total_duration_ms=%.2f)",
                        metrics.get("eval_count"),
                        (metrics.get("total_duration") or 0) / 1_000_000,
                    )

        return "".join(chunks)

    async def format_text(self, text: str) -> str:
        """Format the provided text using the configured LLM instance."""

        transcript = text.strip()
        if not transcript:
            logger.info("Empty transcript provided. Skipping formatting step.")
            return ""

        logger.info("Formatting transcript via %s using model '%s'", self.host, self.model)
        try:
            stream = await self._client.generate(
                model=self.model,
                system=self.system_prompt,
                prompt=transcript,
                stream=True,
            )
        except (RequestError, ResponseError) as exc:
            logger.error("Failed to start Ollama generation: %s", exc)
            raise RuntimeError("Unable to start formatting request via Ollama.") from exc

        try:
            formatted = await asyncio.wait_for(
                self._stream_response(stream),
                timeout=self.request_timeout,
            )
        except asyncio.TimeoutError as exc:
            logger.error("Formatting timed out after %.1fs", self.request_timeout)
            raise TimeoutError(
                f"Formatting request exceeded the {self.request_timeout:.1f}s timeout."
            ) from exc
        except ResponseError as exc:
            logger.error("Ollama stream failed: %s", exc)
            raise RuntimeError("Formatting request failed while streaming the response.") from exc

        result = formatted.strip()
        logger.debug("Formatting complete. Output length: %d characters", len(result))
        return result
