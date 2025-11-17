"""Handlers for processing voice messages."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional
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

PendingVoiceStatus = Literal["pending", "processing"]


@dataclass
class PendingGroupVoice:
    """Metadata describing a pending voice message in a group chat."""

    id: str
    chat_id: int
    voice_message_id: int
    reply_to_message_id: int | None
    thread_id: int | None
    file_path: Path
    initiator_id: int | None
    status: PendingVoiceStatus = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self, ttl: timedelta) -> bool:
        return datetime.now(timezone.utc) - self.created_at >= ttl


class PendingVoiceStore:
    """In-memory storage for delayed group voice processing."""

    def __init__(self, *, ttl: timedelta, cleanup_interval: int = 60) -> None:
        self._ttl = ttl
        self._cleanup_interval = cleanup_interval
        self._items: dict[str, PendingGroupVoice] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def ttl(self) -> timedelta:
        return self._ttl

    async def add(self, entry: PendingGroupVoice) -> None:
        async with self._lock:
            self._items[entry.id] = entry
        logger.info(
            "Pending voice stored: entry_id=%s chat_id=%s message_id=%s",
            entry.id,
            entry.chat_id,
            entry.voice_message_id,
        )

    async def get(self, entry_id: str) -> PendingGroupVoice | None:
        async with self._lock:
            return self._items.get(entry_id)

    async def pop(self, entry_id: str) -> PendingGroupVoice | None:
        async with self._lock:
            entry = self._items.pop(entry_id, None)
        if entry:
            logger.info(
                "Pending voice removed: entry_id=%s chat_id=%s",
                entry.id,
                entry.chat_id,
            )
        return entry

    async def mark_processing(self, entry_id: str) -> bool:
        async with self._lock:
            entry = self._items.get(entry_id)
            if entry is None or entry.status != "pending":
                return False
            entry.status = "processing"
            return True

    async def cleanup_expired(self) -> list[PendingGroupVoice]:
        threshold = datetime.now(timezone.utc) - self._ttl
        expired: list[PendingGroupVoice] = []
        async with self._lock:
            for entry_id, entry in list(self._items.items()):
                if entry.created_at <= threshold and entry.status != "processing":
                    expired.append(self._items.pop(entry_id))
        if expired:
            logger.info("Expired pending voices: %s", len(expired))
        return expired

    def ensure_cleanup_task(self) -> None:
        if self._cleanup_task and not self._cleanup_task.done():
            return
        loop = asyncio.get_running_loop()
        self._cleanup_task = loop.create_task(self._cleanup_loop())
        logger.info("Started pending voice cleanup loop")

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                expired = await self.cleanup_expired()
                for entry in expired:
                    logger.info(
                        "Pending voice %s expired in chat %s", entry.id, entry.chat_id
                    )
                    await _cleanup_file(entry.file_path)
        except asyncio.CancelledError:  # pragma: no cover - graceful shutdown
            logger.info("Pending voice cleanup loop cancelled")
            raise


_PENDING_STORE = PendingVoiceStore(ttl=timedelta(minutes=10), cleanup_interval=90)
_CALLBACK_PREFIX = "gv:"


def configure_pipeline(pipeline: VoicePipeline) -> None:
    """Attach a pipeline instance used for handling voice messages."""

    global _PIPELINE
    _PIPELINE = pipeline
    _PENDING_STORE.ensure_cleanup_task()
    logger.info("Voice pipeline configured")


@voice_router.callback_query(F.data.startswith(_CALLBACK_PREFIX))
async def handle_voice_callback(call: CallbackQuery) -> None:
    """Process inline button callbacks requesting a summary in group chats."""

    if _PIPELINE is None:  # pragma: no cover - configured during bootstrap
        raise RuntimeError("Voice pipeline is not configured.")

    data = call.data or ""
    entry_id = data[len(_CALLBACK_PREFIX) :]
    message = call.message
    logger.info(
        "Callback received: entry_id=%s chat_id=%s user_id=%s",
        entry_id,
        message.chat.id if message and message.chat else None,
        call.from_user.id if call.from_user else None,
    )
    if not entry_id or message is None or message.chat is None:
        logger.warning("Invalid callback payload=%s", data)
        await call.answer("Кнопка больше не работает.", show_alert=True)
        return

    chat = message.chat
    entry = await _PENDING_STORE.get(entry_id)
    if entry is None or entry.chat_id != chat.id:
        logger.info("Pending entry missing or mismatched for %s", entry_id)
        await call.answer("Файл больше не доступен. Запишите новое голосовое.", show_alert=True)
        return

    if entry.is_expired(_PENDING_STORE.ttl):
        logger.info("Pending entry %s expired before callback", entry_id)
        await _remove_pending_entry(entry.id)
        await call.answer("Файл больше не доступен. Запишите новое голосовое.", show_alert=True)
        return

    if entry.status == "processing":
        logger.info("Pending entry %s already processing", entry_id)
        await call.answer("Саммари уже готовится.", show_alert=True)
        return

    marked = await _PENDING_STORE.mark_processing(entry.id)
    if not marked:
        logger.info("Pending entry %s cannot switch to processing", entry_id)
        await call.answer("Саммари уже запускается.", show_alert=True)
        return

    await call.answer("Начинаю обработку…")
    try:
        await message.edit_text(
            "Пожалуйста, подождите — готовлю саммари…",
            reply_markup=None,
        )
    except TelegramAPIError:
        logger.warning("Failed to update prompt message for entry %s", entry_id)

    try:
        summary = await _execute_voice_pipeline(entry.file_path, log_context=entry.id)
    except VoiceProcessingFailure as exc:
        await _finalize_prompt_response(
            prompt_message=message,
            bot=message.bot,
            chat_id=entry.chat_id,
            reply_to_message_id=entry.voice_message_id,
            thread_id=entry.thread_id,
            text=exc.user_message,
        )
    else:
        await _finalize_prompt_response(
            prompt_message=message,
            bot=message.bot,
            chat_id=entry.chat_id,
            reply_to_message_id=entry.voice_message_id,
            thread_id=entry.thread_id,
            text=summary,
        )
    finally:
        await _remove_pending_entry(entry.id)


@voice_router.message(F.content_type == ContentType.VOICE)
async def handle_voice_message(message: Message) -> None:
    """Handle Telegram voice messages."""

    voice = message.voice
    if voice is None:  # pragma: no cover - safeguarded by filter
        return

    logger.info(
        "Voice message received: chat_id=%s message_id=%s type=%s",
        message.chat.id,
        message.message_id,
        message.chat.type,
    )

    await _handle_audio_content(
        message=message,
        file_id=voice.file_id,
        mime_type=voice.mime_type,
        log_label="voice",
    )


@voice_router.message(F.content_type == ContentType.AUDIO)
async def handle_audio_file(message: Message) -> None:
    """Handle Telegram audio files."""

    audio = message.audio
    if audio is None:  # pragma: no cover - safeguarded by filter
        return

    logger.info(
        "Audio file received: chat_id=%s message_id=%s type=%s",
        message.chat.id,
        message.message_id,
        message.chat.type,
    )
    extension_hint = Path(audio.file_name).suffix if audio.file_name else None
    await _handle_audio_content(
        message=message,
        file_id=audio.file_id,
        mime_type=audio.mime_type,
        log_label="audio",
        extension_hint=extension_hint,
    )


@voice_router.message(F.content_type == ContentType.DOCUMENT)
async def handle_audio_document(message: Message) -> None:
    """Handle audio files sent as generic documents."""

    document = message.document
    if document is None:  # pragma: no cover - safeguarded by filter
        return

    if not _is_audio_document(document):
        logger.debug(
            "Document ignored: chat_id=%s message_id=%s filename=%s",
            message.chat.id,
            message.message_id,
            document.file_name,
        )
        return

    logger.info(
        "Audio document received: chat_id=%s message_id=%s type=%s",
        message.chat.id,
        message.message_id,
        message.chat.type,
    )

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
    except TelegramAPIError:
        logger.exception("Failed to download %s %s", log_label, file_id)
        await message.answer("Не удалось скачать аудио. Попробуйте позже.")
        return
    except Exception:  # noqa: BLE001 - unexpected download issues
        logger.exception("Unexpected error while downloading %s %s", log_label, file_id)
        await message.answer("Произошла ошибка при загрузке аудио-файла.")
        return

    if message.chat.type == ChatType.PRIVATE:
        logger.info(
            "Processing private %s message_id=%s immediately",
            log_label,
            message.message_id,
        )
        await _process_downloaded_audio(message, download_path)
        return

    logger.info(
        "Scheduling group %s message_id=%s for deferred processing",
        log_label,
        message.message_id,
    )
    await _schedule_group_voice(message, download_path)


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


def _normalize_extension(extension: str | None) -> str | None:
    if not extension:
        return None
    extension = extension.strip()
    if not extension:
        return None
    if not extension.startswith("."):
        return f".{extension}"
    return extension


class VoiceProcessingFailure(Exception):
    """Wrap predictable pipeline failures with a user-friendly message."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


async def _execute_voice_pipeline(download_path: Path, *, log_context: str) -> str:
    if _PIPELINE is None:
        raise RuntimeError("Voice pipeline is not configured.")

    pipeline_error_reply = (
        "Не получилось обработать голосовое. Попробуйте ещё раз через пару минут."
    )
    response: str | None = None

    try:
        logger.info("Queueing voice pipeline task for %s", log_context)
        response = await _PIPELINE.process_voice(audio_path=download_path)
        logger.info("Voice pipeline completed for %s", log_context)
    except (AudioConversionError, AudioValidationError) as exc:
        logger.warning("Audio preparation failed for %s: %s", log_context, exc)
        raise VoiceProcessingFailure(
            "Аудио-файл нельзя обработать. Попробуйте записать новое сообщение."
        ) from exc
    except (TranscriptionError, FormattingError, VoicePipelineError) as exc:
        logger.warning("Pipeline failed for %s: %s", log_context, exc)
        raise VoiceProcessingFailure(pipeline_error_reply) from exc
    except Exception as exc:  # noqa: BLE001 - unexpected pipeline issues
        logger.exception("Unhandled pipeline error for %s", log_context)
        raise VoiceProcessingFailure(pipeline_error_reply) from exc

    text = (response or "").strip()
    if not text:
        text = "Похоже, голосовое пустое. Запиши сообщение ещё раз."
    return text


async def _process_downloaded_audio(message: Message, download_path: Path) -> None:
    prompt_message: Message | None = None
    try:
        prompt_message = await message.answer(
            "Пожалуйста, подождите — готовлю саммари…",
        )
    except TelegramAPIError:
        logger.warning(
            "Failed to send waiting prompt for private message_id=%s", message.message_id
        )
    try:
        logger.info(
            "Starting pipeline for private message_id=%s",
            message.message_id,
        )
        text = await _execute_voice_pipeline(download_path, log_context=download_path.name)
    except VoiceProcessingFailure as exc:
        await _finalize_prompt_response(
            prompt_message=prompt_message,
            bot=message.bot,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            thread_id=message.message_thread_id,
            text=exc.user_message,
        )
    else:
        await _finalize_prompt_response(
            prompt_message=prompt_message,
            bot=message.bot,
            chat_id=message.chat.id,
            reply_to_message_id=message.message_id,
            thread_id=message.message_thread_id,
            text=text,
        )
    finally:
        await _cleanup_file(download_path)
        logger.info("Private message_id=%s finished", message.message_id)


async def _schedule_group_voice(message: Message, download_path: Path) -> None:
    entry_id = uuid4().hex
    pending = PendingGroupVoice(
        id=entry_id,
        chat_id=message.chat.id,
        voice_message_id=message.message_id,
        reply_to_message_id=(
            message.reply_to_message.message_id if message.reply_to_message else None
        ),
        thread_id=message.message_thread_id,
        file_path=download_path,
        initiator_id=message.from_user.id if message.from_user else None,
    )
    await _PENDING_STORE.add(pending)

    button = InlineKeyboardButton(
        text="Сделать саммари",
        callback_data=f"{_CALLBACK_PREFIX}{entry_id}",
    )
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[button]])

    await message.bot.send_message(
        chat_id=message.chat.id,
        text="Нажмите кнопку, чтобы получить саммари.",
        reply_to_message_id=message.message_id,
        message_thread_id=message.message_thread_id,
        reply_markup=keyboard,
    )
    logger.info(
        "Prompted chat_id=%s message_id=%s with callback button entry_id=%s",
        message.chat.id,
        message.message_id,
        entry_id,
    )


async def _finalize_prompt_response(
    *,
    prompt_message: Message | None,
    bot: Bot,
    chat_id: int,
    reply_to_message_id: int | None,
    thread_id: int | None,
    text: str,
) -> None:
    """Replace the waiting prompt text or fall back to a new reply message."""

    if prompt_message is not None:
        try:
            await prompt_message.edit_text(text, reply_markup=None)
            return
        except TelegramAPIError:
            logger.warning(
                "Failed to edit prompt message chat_id=%s message_id=%s",
                chat_id,
                prompt_message.message_id,
                exc_info=True,
            )

    await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_to_message_id=reply_to_message_id,
        message_thread_id=thread_id,
    )


async def _remove_pending_entry(entry_id: str) -> None:
    entry = await _PENDING_STORE.pop(entry_id)
    if entry is None:
        return
    logger.info(
        "Cleaning up pending entry %s (chat_id=%s)",
        entry.id,
        entry.chat_id,
    )
    await _cleanup_file(entry.file_path)


async def _cleanup_file(path: Path) -> None:
    try:
        if _PIPELINE is not None:
            await asyncio.to_thread(_PIPELINE.audio_processor.cleanup, path)
        elif path.exists():  # pragma: no cover - safeguard for shutdown
            path.unlink(missing_ok=True)
        logger.info("Temporary file %s removed", path)
    except FileNotFoundError:  # pragma: no cover - concurrent cleanup
        return
    except Exception:  # pragma: no cover - log but do not raise
        logger.warning("Failed to cleanup file %s", path, exc_info=True)
