"""Handlers for processing voice messages."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from aiogram import Bot, F, Router
from aiogram.enums import ChatType, ContentType
from aiogram.exceptions import TelegramAPIError
from aiogram.types import (
    CallbackQuery,
    Document,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

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

_MIN_VOICE_DURATION_SECONDS = 1


@dataclass
class PendingVoiceRequest:
    chat_id: int
    message_id: int
    file_id: str
    mime_type: str | None
    extension_hint: str | None
    log_label: str
    created_at: float
    in_progress: bool = False


_PENDING_REQUESTS: Dict[str, PendingVoiceRequest] = {}
_PENDING_LOCK = asyncio.Lock()
_REQUEST_TTL_SECONDS = 900  # 15 minutes
_SUMMARY_PREFIX = "summary:"
_SUMMARY_DONE_DATA = "summary_done"
_SUMMARY_BUTTON_TEXT = "Сделать саммари"
_SUMMARY_LOCKED_TEXT = "Уже обрабатывается/готово"


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
async def handle_voice_messages(message: Message) -> None:
    """Handle Telegram voice messages (opus-in-ogg)."""

    logger.info(
        "Voice message received event: chat_id=%s message_id=%s content_type=%s",
        message.chat.id,
        message.message_id,
        message.content_type,
    )
    voice = message.voice
    if voice is None:
        logger.debug(
            "Voice handler invoked without payload: chat_id=%s message_id=%s content_type=%s",
            message.chat.id,
            message.message_id,
            message.content_type,
        )
        return

    duration = voice.duration or 0
    logger.info(
        "Received voice: chat_id=%s message_id=%s file_id=%s duration=%ss",
        message.chat.id,
        message.message_id,
        voice.file_id,
        duration,
    )

    if duration < _MIN_VOICE_DURATION_SECONDS:
        logger.info(
            "Voice rejected because it is too short: chat_id=%s message_id=%s duration=%s",
            message.chat.id,
            message.message_id,
            duration,
        )
        await message.reply("Слишком короткое сообщение, попробуйте записать длиннее.")
        return

    if _is_group_chat(message):
        logger.info(
            "Using inline-button flow for group voice: chat_id=%s message_id=%s",
            message.chat.id,
            message.message_id,
        )
        await _queue_summary_request(
            message=message,
            file_id=voice.file_id,
            mime_type=voice.mime_type,
            log_label="voice",
        )
    else:
        logger.info(
            "Processing private voice immediately: chat_id=%s message_id=%s",
            message.chat.id,
            message.message_id,
        )
        await _process_immediate_request(
            message=message,
            file_id=voice.file_id,
            mime_type=voice.mime_type,
            log_label="voice",
        )


@voice_router.message(
    F.content_type.in_({ContentType.AUDIO, ContentType.DOCUMENT})
)
async def handle_audio_like_messages(message: Message) -> None:
    """Handle message types that can contain audio needing a summary."""

    logger.info(
        "Audio-like message received: chat_id=%s message_id=%s content_type=%s",
        message.chat.id,
        message.message_id,
        message.content_type,
    )
    is_group = _is_group_chat(message)

    if message.content_type == ContentType.AUDIO and message.audio:
        audio = message.audio
        extension_hint = Path(audio.file_name).suffix if audio.file_name else None
        logger.info(
            "Detected audio payload: chat_id=%s message_id=%s file_id=%s",
            message.chat.id,
            message.message_id,
            audio.file_id,
        )
        if is_group:
            logger.info(
                "Queuing audio from group chat for inline flow: chat_id=%s message_id=%s",
                message.chat.id,
                message.message_id,
            )
            await _queue_summary_request(
                message=message,
                file_id=audio.file_id,
                mime_type=audio.mime_type,
                log_label="audio",
                extension_hint=extension_hint,
            )
        else:
            logger.info(
                "Processing private audio immediately: chat_id=%s message_id=%s",
                message.chat.id,
                message.message_id,
            )
            await _process_immediate_request(
                message=message,
                file_id=audio.file_id,
                mime_type=audio.mime_type,
                log_label="audio",
                extension_hint=extension_hint,
            )
        return

    if message.content_type == ContentType.DOCUMENT and message.document:
        document = message.document
        if not _is_audio_document(document):
            logger.debug(
                "Ignoring non-audio document in chat_id=%s message_id=%s",
                message.chat.id,
                message.message_id,
            )
            return
        extension_hint = Path(document.file_name).suffix if document.file_name else None
        logger.info(
            "Detected audio document: chat_id=%s message_id=%s file_id=%s",
            message.chat.id,
            message.message_id,
            document.file_id,
        )
        if is_group:
            logger.info(
                "Queuing audio document from group chat: chat_id=%s message_id=%s",
                message.chat.id,
                message.message_id,
            )
            await _queue_summary_request(
                message=message,
                file_id=document.file_id,
                mime_type=document.mime_type,
                log_label="document_audio",
                extension_hint=extension_hint,
            )
        else:
            logger.info(
                "Processing private audio document immediately: chat_id=%s message_id=%s",
                message.chat.id,
                message.message_id,
            )
            await _process_immediate_request(
                message=message,
                file_id=document.file_id,
                mime_type=document.mime_type,
                log_label="document_audio",
                extension_hint=extension_hint,
            )
        return

    logger.debug(
        "Unhandled content_type=%s for chat_id=%s message_id=%s",
        message.content_type,
        message.chat.id,
        message.message_id,
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


async def _queue_summary_request(
    *,
    message: Message,
    file_id: str,
    mime_type: str | None,
    log_label: str,
    extension_hint: str | None = None,
) -> None:
    """Store a pending request and show an inline button to trigger processing."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    chat_id = message.chat.id
    logger.info(
        "Queueing summary request: chat_id=%s message_id=%s file_id=%s log_label=%s",
        chat_id,
        message.message_id,
        file_id,
        log_label,
    )
    request = PendingVoiceRequest(
        chat_id=chat_id,
        message_id=message.message_id,
        file_id=file_id,
        mime_type=mime_type,
        extension_hint=extension_hint,
        log_label=log_label,
        created_at=time.time(),
    )
    key = _make_request_key(chat_id, message.message_id)

    async with _PENDING_LOCK:
        _cleanup_expired_requests_locked(now=request.created_at)
        _PENDING_REQUESTS[key] = request

    logger.info(
        "Pending summary request stored: chat_id=%s message_id=%s log_label=%s",
        chat_id,
        message.message_id,
        log_label,
    )
    keyboard = _build_summary_keyboard(key)
    try:
        reply = await message.reply(
            "Нажмите кнопку, чтобы получить саммари",
            reply_markup=keyboard,
        )
        logger.info(
            "Prompt with inline button sent: chat_id=%s message_id=%s reply_id=%s",
            chat_id,
            message.message_id,
            reply.message_id,
        )
    except TelegramAPIError:
        logger.exception(
            "Failed to send inline button prompt: chat_id=%s message_id=%s",
            chat_id,
            message.message_id,
        )
        async with _PENDING_LOCK:
            _PENDING_REQUESTS.pop(key, None)
        await message.answer(
            "Не получилось показать кнопку саммари. Попробуйте отправить аудио ещё раз.",
        )
        logger.info(
            "Fallback message sent after inline keyboard failure: chat_id=%s message_id=%s",
            chat_id,
            message.message_id,
        )


def _make_request_key(chat_id: int, message_id: int) -> str:
    return f"{chat_id}:{message_id}"


def _cleanup_expired_requests_locked(*, now: float | None = None) -> None:
    now = now or time.time()
    expired_keys = [
        key
        for key, request in _PENDING_REQUESTS.items()
        if now - request.created_at > _REQUEST_TTL_SECONDS
    ]
    for key in expired_keys:
        request = _PENDING_REQUESTS.pop(key, None)
        if request is None:
            continue
        logger.info(
            "Expired pending request: chat_id=%s message_id=%s",
            request.chat_id,
            request.message_id,
        )


def _build_summary_keyboard(key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_SUMMARY_BUTTON_TEXT,
                    callback_data=f"{_SUMMARY_PREFIX}{key}",
                )
            ]
        ]
    )


def _build_locked_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_SUMMARY_LOCKED_TEXT,
                    callback_data=_SUMMARY_DONE_DATA,
                )
            ]
        ]
    )


async def _mark_button_in_progress(message: Message | None) -> None:
    if message is None:
        return
    try:
        await message.edit_reply_markup(reply_markup=_build_locked_keyboard())
        logger.info(
            "Inline keyboard locked: chat_id=%s message_id=%s",
            message.chat.id if message.chat else "unknown",
            message.message_id,
        )
    except TelegramAPIError:
        logger.debug("Failed to update inline keyboard for message_id=%s", message.message_id)


async def _remove_keyboard(message: Message | None) -> None:
    if message is None:
        return
    try:
        await message.edit_reply_markup(reply_markup=None)
        logger.info(
            "Inline keyboard removed: chat_id=%s message_id=%s",
            message.chat.id if message.chat else "unknown",
            message.message_id,
        )
    except TelegramAPIError:
        logger.debug("Failed to remove inline keyboard for message_id=%s", message.message_id)


@voice_router.callback_query(F.data == _SUMMARY_DONE_DATA)
async def handle_locked_summary(callback: CallbackQuery) -> None:
    """Gracefully acknowledge callbacks on disabled keyboards."""

    logger.info(
        "Locked summary callback received: user_id=%s message_id=%s",
        callback.from_user.id if callback.from_user else "unknown",
        callback.message.message_id if callback.message else "unknown",
    )
    await callback.answer("Запрос уже обработан.")


@voice_router.callback_query(F.data.startswith(_SUMMARY_PREFIX))
async def handle_summary_callback(callback: CallbackQuery) -> None:
    """Trigger processing when the inline button is pressed."""

    key = (callback.data or "")[len(_SUMMARY_PREFIX) :]
    message = callback.message
    if not key or message is None:
        logger.warning(
            "Invalid callback received: callback_id=%s user_id=%s",
            callback.id,
            callback.from_user.id if callback.from_user else "unknown",
        )
        await callback.answer("Сообщение не найдено.")
        return

    async with _PENDING_LOCK:
        _cleanup_expired_requests_locked()
        request = _PENDING_REQUESTS.get(key)
        if request is None:
            logger.info("Missing pending request for key=%s", key)
            await callback.answer("Запрос не найден или истёк.")
            await _mark_button_in_progress(message)
            return
        if request.in_progress:
            logger.info(
                "Request already running: chat_id=%s message_id=%s",
                request.chat_id,
                request.message_id,
            )
            await callback.answer("Этот запрос уже обрабатывается.")
            await _mark_button_in_progress(message)
            return
        request.in_progress = True

    await _mark_button_in_progress(message)
    logger.info(
        "Callback acknowledged: key=%s chat_id=%s user_id=%s",
        key,
        request.chat_id,
        callback.from_user.id if callback.from_user else "unknown",
    )
    await callback.answer("Запускаю обработку...")
    try:
        await _process_voice_request(bot=callback.bot, request=request)
    finally:
        async with _PENDING_LOCK:
            removed = _PENDING_REQUESTS.pop(key, None)
            if removed is not None:
                logger.info(
                    "Request completed: chat_id=%s message_id=%s",
                    removed.chat_id,
                    removed.message_id,
                )
    await _remove_keyboard(message)


async def _process_voice_request(*, bot: Bot, request: PendingVoiceRequest) -> None:
    """Download the file and run the voice pipeline."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    await _download_and_process_audio(
        bot=bot,
        chat_id=request.chat_id,
        reply_to_message_id=request.message_id,
        file_id=request.file_id,
        mime_type=request.mime_type,
        log_label=request.log_label,
        extension_hint=request.extension_hint,
    )


async def _process_immediate_request(
    *,
    message: Message,
    file_id: str,
    mime_type: str | None,
    log_label: str,
    extension_hint: str | None = None,
) -> None:
    """Download and process an audio file immediately (private chats)."""

    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    bot = message.bot
    if bot is None:
        logger.error(
            "Message has no bot instance attached: chat_id=%s message_id=%s",
            message.chat.id if message.chat else "unknown",
            message.message_id,
        )
        return

    await _download_and_process_audio(
        bot=bot,
        chat_id=message.chat.id,
        reply_to_message_id=message.message_id,
        file_id=file_id,
        mime_type=mime_type,
        log_label=log_label,
        extension_hint=extension_hint,
    )


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.strip()
    if not extension:
        return None
    if not extension.startswith("."):
        return f".{extension}"
    return extension


async def _process_downloaded_audio(
    *, bot: Bot, chat_id: int, reply_to_message_id: int, download_path: Path
) -> None:
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
        await bot.send_message(
            chat_id,
            "Аудио-файл нельзя обработать. Попробуйте записать новое сообщение.",
            reply_to_message_id=reply_to_message_id,
        )
        return
    except (TranscriptionError, FormattingError, VoicePipelineError) as exc:
        logger.warning("Pipeline failed for %s: %s", download_path.name, exc)
        await bot.send_message(
            chat_id,
            pipeline_error_reply,
            reply_to_message_id=reply_to_message_id,
        )
        return
    except Exception:
        logger.exception("Unhandled pipeline error for %s", download_path.name)
        await bot.send_message(
            chat_id,
            pipeline_error_reply,
            reply_to_message_id=reply_to_message_id,
        )
        return
    finally:
        await asyncio.to_thread(_PIPELINE.audio_processor.cleanup, download_path)

    text = (response or "").strip()
    if not text:
        text = "Похоже, голосовое пустое. Запиши сообщение ещё раз."

    await bot.send_message(
        chat_id,
        text,
        reply_to_message_id=reply_to_message_id,
    )
    logger.info(
        "Summary delivered: chat_id=%s reply_to=%s text_length=%s",
        chat_id,
        reply_to_message_id,
        len(text),
    )


def _is_group_chat(message: Message) -> bool:
    chat = message.chat
    if chat is None:
        return False
    chat_type = chat.type
    if isinstance(chat_type, ChatType):
        return chat_type in {ChatType.GROUP, ChatType.SUPERGROUP}
    return chat_type in {ChatType.GROUP.value, ChatType.SUPERGROUP.value, "group", "supergroup"}


async def _download_and_process_audio(
    *,
    bot: Bot,
    chat_id: int,
    reply_to_message_id: int,
    file_id: str,
    mime_type: str | None,
    log_label: str,
    extension_hint: str | None = None,
) -> None:
    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    logger.info(
        "Downloading %s for processing: chat_id=%s message_id=%s",
        log_label,
        chat_id,
        reply_to_message_id,
    )

    tmp_dir = _PIPELINE.tmp_dir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    extension = _normalize_extension(extension_hint) or _detect_extension(mime_type)
    download_path = tmp_dir / (
        f"{log_label}_{chat_id}_{reply_to_message_id}_{uuid4().hex}{extension}"
    )

    try:
        await bot.download(file_id, destination=download_path)
        logger.info("Audio saved to %s", download_path)
    except TelegramAPIError:
        logger.exception("Failed to download %s %s", log_label, file_id)
        await bot.send_message(
            chat_id,
            "Не удалось скачать аудио. Попробуйте позже.",
            reply_to_message_id=reply_to_message_id,
        )
        return
    except Exception:  # noqa: BLE001 - unexpected download issues
        logger.exception("Unexpected error while downloading %s %s", log_label, file_id)
        await bot.send_message(
            chat_id,
            "Произошла ошибка при загрузке аудио-файла.",
            reply_to_message_id=reply_to_message_id,
        )
        return

    await _process_downloaded_audio(
        bot=bot,
        chat_id=chat_id,
        reply_to_message_id=reply_to_message_id,
        download_path=download_path,
    )
