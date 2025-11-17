"""Handlers for processing text messages."""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.enums import ChatType
from aiogram.filters import CommandStart
from aiogram.types import Message

logger = logging.getLogger("bot.text_handler")

text_router = Router(name="text_router")


@text_router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Describe the bot workflow when /start is received."""

    chat_type = message.chat.type
    logger.info(
        "/start received: chat_id=%s type=%s user_id=%s",
        message.chat.id,
        chat_type,
        message.from_user.id if message.from_user else None,
    )
    if chat_type == ChatType.PRIVATE:
        await message.answer("Привет! Пришли голосовое — получишь summary.")
        return

    await message.answer(
        "Я делаю саммари для голосовых. Отправьте запись и дождитесь кнопки «Сделать"
        " саммари» под сообщением, чтобы запустить обработку."
    )


@text_router.message(F.text)
async def handle_text(message: Message) -> None:
    """Encourage users to send an audio message instead of plain text."""

    logger.info(
        "Text message ignored: chat_id=%s type=%s user_id=%s",
        message.chat.id,
        message.chat.type,
        message.from_user.id if message.from_user else None,
    )

    await message.answer(
        "Чтобы получить summary, пришли голосовое или аудио-файл."
        " Текстовые сообщения бот не обрабатывает."
    )
