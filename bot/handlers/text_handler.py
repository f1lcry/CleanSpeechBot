"""Handlers for processing text messages."""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

text_router = Router(name="text_router")


@text_router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Describe the bot workflow when /start is received."""

    await message.answer(
        "Привет! Пришли голосовое — получишь summary.\n"
        "Алгоритм простой: скачиваем файл, конвертируем, распознаём Whisper и"
        " форматируем текст локальной Llama."
    )


@text_router.message()
async def handle_text(message: Message) -> None:
    """Encourage users to send an audio message instead of plain text."""

    await message.answer(
        "Чтобы получить краткий конспект, отправь голосовое или аудио-файл."
        " Текстовые сообщения бот игнорирует."
    )
