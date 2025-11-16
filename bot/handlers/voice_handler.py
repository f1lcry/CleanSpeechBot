"""Handlers for processing voice messages."""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from aiogram import Router
from aiogram.types import Message

from bot.services.pipeline import VoicePipeline

voice_router = Router(name="voice_router")
_PIPELINE: Optional[VoicePipeline] = None

logger = logging.getLogger("bot.voice_handler")


def configure_pipeline(pipeline: VoicePipeline) -> None:
    """Attach a pipeline instance used for handling voice messages."""

    global _PIPELINE
    _PIPELINE = pipeline


@voice_router.message(lambda message: message.voice is not None)
async def handle_voice_message(message: Message) -> None:
    """Handle incoming voice messages by delegating to the voice pipeline."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    voice = message.voice
    if voice is None:
        return

    bot = message.bot
    if bot is None:
        raise RuntimeError("Bot instance is not available in message context.")

    status_message = await message.answer("Обрабатываю голосовое сообщение…")

    tmp_dir = _PIPELINE.audio_processor.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_filename = f"voice-{voice.file_unique_id}-{uuid.uuid4().hex}.oga"
    local_path = tmp_dir / local_filename

    try:
        file = await bot.get_file(voice.file_id)
        with local_path.open("wb") as destination:
            await bot.download_file(file_path=file.file_path, destination=destination)
    except Exception as exc:  # pragma: no cover - network/Telegram errors.
        logger.exception("Failed to download voice file: %s", exc)
        await status_message.edit_text("Не удалось получить голосовое сообщение, попробуйте снова.")
        _PIPELINE.audio_processor.cleanup(local_path)
        return

    try:
        formatted_text = await _PIPELINE.process_voice(audio_path=local_path)
        await status_message.edit_text(formatted_text)
    except Exception as exc:  # pragma: no cover - pipeline safeguards errors.
        logger.exception("Voice pipeline raised for chat %s: %s", message.chat.id, exc)
        await status_message.edit_text("Техническая ошибка. Попробуйте ещё раз позже.")
    finally:
        _PIPELINE.audio_processor.cleanup(local_path)
