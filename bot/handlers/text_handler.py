"""Handlers for processing text messages."""
from __future__ import annotations

from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message

text_router = Router(name="text_router")


@text_router.message(CommandStart())
async def handle_start(message: Message) -> None:
    """Reply to the /start command with a placeholder message."""

    await message.answer("Привет! Отправьте голосовое сообщение для обработки (заглушка).")


@text_router.message()
async def handle_text(message: Message) -> None:
    """Reply to any text message with a placeholder response."""

    await message.answer("Текстовые сообщения пока не обрабатываются (заглушка).")
