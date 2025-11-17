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
    "Чтобы я видел сообщения в группе, выдайте мне права администратора."
)

ready_text = (
    "Отлично! У меня теперь есть права администратора, поэтому я готов работать.\n"
    "Отправьте голосовое, дождитесь кнопки «Сделать саммари» и нажмите её,"
    " чтобы получить краткое изложение прямо в чате. В личных сообщениях"
    " бот продолжает отвечать сразу, без кнопок."
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

    member_like_statuses = {ChatMemberStatus.MEMBER, ChatMemberStatus.ADMINISTRATOR}
    if new_status not in member_like_statuses:
        logger.debug("Ignoring update because new_status=%s", new_status)
        return
    if chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        logger.debug("Skipping greeting because chat type=%s is not a group", chat.type)
        return

    if old_status not in member_like_statuses:
        logger.info("Bot added to chat_id=%s (type=%s). Sending greeting.", chat.id, chat.type)
        await event.bot.send_message(chat_id=chat.id, text=greeting_text)

    if new_status == ChatMemberStatus.ADMINISTRATOR and old_status != ChatMemberStatus.ADMINISTRATOR:
        logger.info("Bot promoted to admin in chat_id=%s. Sending readiness notice.", chat.id)
        await event.bot.send_message(chat_id=chat.id, text=ready_text)
