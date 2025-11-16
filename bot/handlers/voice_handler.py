"""Handlers for processing voice messages."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from uuid import uuid4

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Message

from bot.services.pipeline import (
    AudioConversionError,
    AudioValidationError,
    FormattingError,
    TranscriptionError,
    VoicePipeline,
    VoicePipelineError,
)

voice_router = Router(name="voice_router")
_PIPELINE: Optional[VoicePipeline] = None
logger = logging.getLogger("bot.voice_handler")


def configure_pipeline(pipeline: VoicePipeline) -> None:
    """Attach a pipeline instance used for handling voice messages."""

    global _PIPELINE
    _PIPELINE = pipeline


def _detect_extension(mime_type: str | None) -> str:
    mapping = {
        "audio/ogg": ".ogg",
        "audio/oga": ".oga",
        "audio/x-opus+ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    if not mime_type:
        return ".ogg"
    mime_type = mime_type.lower()
    return mapping.get(mime_type, ".ogg")


@voice_router.message(F.voice)
async def handle_voice_message(message: Message) -> None:
    """Handle incoming voice messages by delegating to the voice pipeline."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    voice = message.voice
    if voice is None:  # pragma: no cover - filter already ensures this
        return

    bot = message.bot
    chat_id = message.chat.id
    logger.info(
        "Voice message received: chat_id=%s message_id=%s duration=%ss size=%s",
        chat_id,
        message.message_id,
        voice.duration,
        voice.file_size,
    )

    tmp_dir = _PIPELINE.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extension = _detect_extension(voice.mime_type)
    download_path = tmp_dir / f"voice_{chat_id}_{message.message_id}_{uuid4().hex}{extension}"

    try:
        await bot.download(voice.file_id, destination=download_path)
        logger.info("Voice saved to %s", download_path)
    except TelegramAPIError as exc:
        logger.exception("Failed to download voice %s: %s", voice.file_id, exc)
        await message.answer("Не удалось скачать голосовое сообщение. Попробуйте позже.")
        return
    except Exception as exc:  # noqa: BLE001 - unexpected download issues
        logger.exception("Unexpected error while downloading voice %s", voice.file_id)
        await message.answer("Произошла ошибка при загрузке голосового сообщения.")
        return

    pipeline_error_reply = (
        "Не получилось обработать голосовое. Попробуйте ещё раз через пару минут."
    )
    response: str | None = None

    try:
        logger.info("Queueing voice pipeline task for %s", download_path.name)
        response = await _PIPELINE.process_voice(audio_path=download_path)
        logger.info("Voice pipeline completed for %s", download_path.name)
    except (AudioConversionError, AudioValidationError) as exc:
        logger.warning("Audio preparation failed for %s: %s", download_path.name, exc)
        await message.answer("Аудио-файл нельзя обработать. Попробуйте записать новое сообщение.")
        return
    except (TranscriptionError, FormattingError, VoicePipelineError) as exc:
        logger.warning("Pipeline failed for %s: %s", download_path.name, exc)
        await message.answer(pipeline_error_reply)
        return
    except Exception:
        logger.exception("Unhandled pipeline error for %s", download_path.name)
        await message.answer(pipeline_error_reply)
        return
    finally:
        await asyncio.to_thread(_PIPELINE.audio_processor.cleanup, download_path)

    text = (response or "").strip()
    if not text:
        text = "Похоже, голосовое пустое. Запиши сообщение ещё раз."

    await message.answer(text)
