"""Handlers for group-specific lifecycle events."""
from __future__ import annotations

import logging

from aiogram import F, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.types import ChatMemberUpdated

logger = logging.getLogger("bot.group_handler")

group_router = Router(name="group_router")
group_router.my_chat_member.filter(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))

greeting_text = (
    "Привет! Я CleanSpeechBot и делаю саммари для голосовых сообщений.\n"
    "Отправьте голосовое, дождитесь кнопки «Сделать саммари» и нажмите её,"
    " чтобы получить краткое изложение прямо в чате.\n"
    "В личных сообщениях бот работает сразу, без кнопок."
)


@group_router.my_chat_member()
async def handle_bot_added(event: ChatMemberUpdated) -> None:
    """Send a greeting when the bot becomes a member/administrator of a group."""

    new_status = event.new_chat_member.status
    old_status = event.old_chat_member.status

    if new_status not in {ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR}:
        return
    if old_status in {ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR}:
        return

    chat = event.chat
    if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return

    logger.info("Bot added to chat_id=%s (type=%s)", chat.id, chat.type)

    await event.bot.send_message(chat_id=chat.id, text=greeting_text)
