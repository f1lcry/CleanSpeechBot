"""Handlers for processing text messages."""
from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import CommandStart
from aiogram.types import Message

text_router = Router(name="text_router")


@text_router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Describe the bot workflow when /start is received."""

    await message.answer("Привет! Пришли голосовое — получишь summary.")


@text_router.message(F.text)
async def handle_text(message: Message) -> None:
    """Encourage users to send an audio message instead of plain text."""

    await message.answer(
        "Чтобы получить summary, пришли голосовое или аудио-файл."
        " Текстовые сообщения бот не обрабатывает."
    )
