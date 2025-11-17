"""Handlers for chat member updates (bot added to groups)."""
from __future__ import annotations

import logging

from aiogram import Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.types import ChatMemberUpdated

chat_member_router = Router(name="chat_member_router")
logger = logging.getLogger("bot.chat_member_handler")

_GREETING_TEXT = (
    "Спасибо, что добавили CleanSpeechBot! Бот делает краткие саммари голосовых"
    " и аудио-сообщений прямо в групповом чате. Пришлите запись, дождитесь ответа"
    " бота с кнопкой \"Сделать саммари\" под оригинальным сообщением и нажмите её —"
    " только после этого начнётся обработка."
)


def _should_greet(update: ChatMemberUpdated) -> bool:
    chat_type = update.chat.type
    if chat_type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return False

    new_status = update.new_chat_member.status
    if new_status != ChatMemberStatus.MEMBER:
        return False

    bot_user_id = update.bot.id
    return update.new_chat_member.user.id == bot_user_id


async def _send_greeting(update: ChatMemberUpdated) -> None:
    if not _should_greet(update):
        return

    logger.info(
        "Bot joined chat_id=%s (title=%s)",
        update.chat.id,
        update.chat.title,
    )
    await update.bot.send_message(update.chat.id, _GREETING_TEXT)


@chat_member_router.my_chat_member()
async def handle_my_chat_member(update: ChatMemberUpdated) -> None:
    """React when the bot is added to a group via my_chat_member updates."""

    await _send_greeting(update)


@chat_member_router.chat_member()
async def handle_chat_member(update: ChatMemberUpdated) -> None:
    """Handle chat_member updates (fallback for some group administrators)."""

    await _send_greeting(update)
