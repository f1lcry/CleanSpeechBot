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
    chat = event.chat

    logger.info(
        "my_chat_member update: chat_id=%s type=%s old=%s new=%s",
        chat.id,
        chat.type,
        old_status,
        new_status,
    )

    if new_status not in {ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR}:
        logger.debug("Ignoring update because new_status=%s", new_status)
        return
    if old_status in {ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR}:
        logger.debug("Bot already had access to chat_id=%s", chat.id)
        return

    if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        logger.debug("Skipping greeting because chat type=%s is not a group", chat.type)
        return

    logger.info("Bot added to chat_id=%s (type=%s). Sending greeting.", chat.id, chat.type)

    await event.bot.send_message(chat_id=chat.id, text=greeting_text)
