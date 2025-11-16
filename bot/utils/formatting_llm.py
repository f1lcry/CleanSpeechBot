"""Client for formatting text via a local LLM served by Ollama."""
from __future__ import annotations

from typing import Any


class FormattingLLMClient:
    """A lightweight client that formats transcribed text using Ollama-hosted models."""

    def __init__(self, host: str, model: str) -> None:
        """Store configuration required for invoking the Ollama endpoint."""

        self.host = host
        self.model = model

    async def format_text(self, text: str) -> str:
        """Format the provided text using the configured LLM instance.

        TODO: Implement asynchronous HTTP calls to the Ollama API with streaming support.
        """

        raise NotImplementedError("Formatting LLM integration is not yet implemented.")
