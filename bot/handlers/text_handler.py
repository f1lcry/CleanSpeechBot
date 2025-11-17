"""Handlers for processing text messages."""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

text_router = Router(name="text_router")
logger = logging.getLogger("bot.text_handler")


@text_router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Describe the bot workflow when /start is received."""

    logger.info(
        "Handling /start: chat_id=%s user_id=%s is_forum=%s",
        message.chat.id,
        message.from_user.id if message.from_user else "unknown",
        getattr(message.chat, "is_forum", False),
    )
    await message.answer(
        "Этот бот превращает голосовые и аудио-сообщения в краткие саммари."
        " После отправки записи нажмите кнопку \"Сделать саммари\" под сообщением —"
        " только тогда начнётся обработка."
    )


@text_router.message(F.text)
async def handle_text(message: Message) -> None:
    """Encourage users to send an audio message instead of plain text."""

    logger.info(
        "Handling text message: chat_id=%s user_id=%s message_id=%s",
        message.chat.id,
        message.from_user.id if message.from_user else "unknown",
        message.message_id,
    )
    await message.answer(
        "Чтобы получить summary, пришли голосовое или аудио-файл."
        " Текстовые сообщения бот не обрабатывает."
    )
