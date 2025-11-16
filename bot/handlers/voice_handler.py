"""Handlers for processing voice messages."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from aiogram import Router
from aiogram.types import Message

from bot.services.pipeline import VoicePipeline

voice_router = Router(name="voice_router")
_PIPELINE: Optional[VoicePipeline] = None


def configure_pipeline(pipeline: VoicePipeline) -> None:
    """Attach a pipeline instance used for handling voice messages."""

    global _PIPELINE
    _PIPELINE = pipeline


@voice_router.message(lambda message: message.voice is not None)
async def handle_voice_message(message: Message) -> None:
    """Handle incoming voice messages by delegating to the voice pipeline."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    # TODO: Save the voice file locally and pass the actual path to the pipeline.
    fake_audio_path = Path(message.voice.file_id)  # type: ignore[arg-type]
    formatted_text = await _PIPELINE.process_voice(audio_path=fake_audio_path)
    await message.answer(
        "Голосовое сообщение принято и обрабатывается (заглушка).\n"
        f"Результат: {formatted_text}"
    )
