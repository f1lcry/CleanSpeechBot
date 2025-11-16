"""Client for formatting text via a local LLM served by Ollama."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
from typing import Any
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger("bot.formatting_llm")

LLAMA_PROMPT = (
    "Ты получаешь неотформатированный текст транскрипта голосового сообщения.\n"
    "Текст может содержать неточные распознавания, отсутствие пунктуации, разговорные вставки и повторы.\n\n"
    "Требования к результату:\n"
    "1. Привести текст в литературно-нормальный вид: расставить пунктуацию, поправить ошибки распознавания, улучшить речевые обороты.\n"
    "2. Удалить явные повторы, паразитные слова и фрагменты, не несущие смысловой нагрузки.\n"
    "3. Слегка суммаризировать: сократить лишнее и сделать текст более плотным, но сохранить всю основную информацию и структуру мысли.\n"
    "4. Не добавлять новых фактов, не менять смысл сказанного.\n"
    "5. Итоговый текст должен быть чистым, связным и аккуратно оформленным.\n\n"
    "Вход: транскрипт.\n"
    "Выход: отформатированный связный текст."
)


class FormattingLLMClient:
    """A lightweight client that formats transcribed text using Ollama-hosted models."""

    def __init__(self, host: str, model: str) -> None:
        """Store configuration required for invoking the Ollama endpoint."""

        parsed = urlparse(host)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Ollama host must include the scheme (http/https).")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama host must point to a local interface for privacy reasons.")

        self.host = host.rstrip("/")
        self.model = model

    async def format_text(self, text: str) -> str:
        """Format the provided text using the configured LLM instance."""

        if not text.strip():
            return ""

        payload = {
            "model": self.model,
            "prompt": f"{LLAMA_PROMPT}\n\n{text.strip()}",
            "stream": False,
        }

        job_id = uuid.uuid4().hex
        start = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=180)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(f"{self.host}/api/generate", json=payload) as response:
                    response.raise_for_status()
                    data: Any = await response.json()
            except aiohttp.ClientError as exc:
                logger.exception("Formatting task %s failed during HTTP request: %s", job_id, exc)
                raise RuntimeError("Не удалось выполнить форматирование текста.") from exc

        duration = time.perf_counter() - start
        logger.info("Formatting task %s finished in %.2fs", job_id, duration)

        formatted = data.get("response") if isinstance(data, dict) else None
        if not formatted:
            raise RuntimeError("Ollama ответил без текста форматирования.")
        return str(formatted).strip()


async def _cli_async(text: str, host: str, model: str) -> None:
    client = FormattingLLMClient(host=host, model=model)
    formatted = await client.format_text(text)
    print(formatted)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Format raw text locally via Ollama.")
    parser.add_argument("--text", required=True, help="Raw text that should be formatted.")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--model", default="llama3.1-8b", help="Name of the Ollama model to use.")
    args = parser.parse_args()

    asyncio.run(_cli_async(text=args.text, host=args.host, model=args.model))


if __name__ == "__main__":
    _cli()
