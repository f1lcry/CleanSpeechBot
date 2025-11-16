"""Handlers for processing voice messages."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional
from uuid import uuid4

from pathlib import Path

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.types import Document, Message

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
    """Handle Telegram voice messages."""

    voice = message.voice
    if voice is None:  # pragma: no cover - safeguarded by filter
        return

    await _handle_audio_content(
        message=message,
        file_id=voice.file_id,
        mime_type=voice.mime_type,
        log_label="voice",
    )


@voice_router.message(F.audio)
async def handle_audio_file(message: Message) -> None:
    """Handle Telegram audio files."""

    audio = message.audio
    if audio is None:  # pragma: no cover - safeguarded by filter
        return

    extension_hint = Path(audio.file_name).suffix if audio.file_name else None
    await _handle_audio_content(
        message=message,
        file_id=audio.file_id,
        mime_type=audio.mime_type,
        log_label="audio",
        extension_hint=extension_hint,
    )


@voice_router.message(F.document)
async def handle_audio_document(message: Message) -> None:
    """Handle audio files sent as generic documents."""

    document = message.document
    if document is None:  # pragma: no cover - safeguarded by filter
        return

    if not _is_audio_document(document):
        return

    extension_hint = Path(document.file_name).suffix if document.file_name else None
    await _handle_audio_content(
        message=message,
        file_id=document.file_id,
        mime_type=document.mime_type,
        log_label="document_audio",
        extension_hint=extension_hint,
    )


def _is_audio_document(document: Document) -> bool:
    """Return True when the document mime type or extension indicates audio content."""

    mime_type = (document.mime_type or "").lower()
    if mime_type.startswith("audio/"):
        return True
    if document.file_name:
        suffix = Path(document.file_name).suffix.lower()
        return suffix in {".mp3", ".wav", ".ogg", ".oga", ".m4a", ".flac"}
    return False


async def _handle_audio_content(
    *,
    message: Message,
    file_id: str,
    mime_type: str | None,
    log_label: str,
    extension_hint: str | None = None,
) -> None:
    """Download and process a Telegram media file representing audio content."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    bot = message.bot
    chat_id = message.chat.id
    logger.info(
        "%s received: chat_id=%s message_id=%s",
        log_label,
        chat_id,
        message.message_id,
    )

    tmp_dir = _PIPELINE.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extension = _normalize_extension(extension_hint) or _detect_extension(mime_type)
    download_path = tmp_dir / (
        f"{log_label}_{chat_id}_{message.message_id}_{uuid4().hex}{extension}"
    )

    try:
        await bot.download(file_id, destination=download_path)
        logger.info("Audio saved to %s", download_path)
    except TelegramAPIError as exc:
        logger.exception("Failed to download %s %s", log_label, file_id)
        await message.answer("Не удалось скачать аудио. Попробуйте позже.")
        return
    except Exception as exc:  # noqa: BLE001 - unexpected download issues
        logger.exception("Unexpected error while downloading %s %s", log_label, file_id)
        await message.answer("Произошла ошибка при загрузке аудио-файла.")
        return

    await _process_downloaded_audio(message, download_path)


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.strip()
    if not extension:
        return None
    if not extension.startswith("."):
        return f".{extension}"
    return extension


async def _process_downloaded_audio(message: Message, download_path: Path) -> None:
    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

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
