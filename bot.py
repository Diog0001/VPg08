"""Telegram-бот командного чата: память в Pinecone через пайплайны Haystack."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone

import telebot
from telebot.types import Message

from config import Settings, setup_logging
from pipelines import PipelineManager

setup_logging()
logger = logging.getLogger("bot")

QUESTION_MARKERS = (
    "что думаешь",
    "как думаешь",
    "как считаешь",
    "твое мнение",
    "твоё мнение",
    "что скажешь",
    "что думаете",
    "как вам кажется",
    "какое мнение",
)

COMMANDS = {"start_listening", "stop_listening", "start", "help"}


def display_name(message: Message) -> str:
    user = message.from_user
    if user is None:
        return "участник"
    if user.username:
        return f"@{user.username}"
    parts = [user.first_name or "", user.last_name or ""]
    name = " ".join(part for part in parts if part).strip()
    return name or f"id{user.id}"


def format_record(username: str, date_iso: str, text: str) -> str:
    return f"[{date_iso}] {username}: {text}"


def message_date_iso(message: Message) -> str:
    ts = getattr(message, "date", None)
    if ts:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def command_name(message: Message) -> str | None:
    if not message.text:
        return None
    first = message.text.split()[0]
    if not first.startswith("/"):
        return None
    return first[1:].split("@")[0].lower()


def is_addressed_to_bot(message: Message, bot_username: str, bot_id: int) -> bool:
    text = (message.text or "").lower()
    chat_type = message.chat.type
    if chat_type == "private":
        return True
    if bot_username and f"@{bot_username.lower()}" in text:
        return True
    reply = message.reply_to_message
    if reply and reply.from_user and reply.from_user.id == bot_id:
        return True
    return any(marker in text for marker in QUESTION_MARKERS)


def split_telegram_text(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        chunks.append(rest[:limit])
        rest = rest[limit:]
    return chunks


def main() -> None:
    settings = Settings.load()
    logger.info(
        "Старт бота: model=%s embedding=%s dim=%s top_k=%s",
        settings.openai_model,
        settings.embedding_model,
        settings.embedding_dimensions,
        settings.retrieval_top_k,
    )
    manager = PipelineManager(settings)
    bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode=None)
    me = bot.get_me()
    bot_username = me.username or ""
    bot_id = me.id
    logger.info("Telegram: @%s id=%s", bot_username, bot_id)
    logger.info(
        "Для групповых чатов в BotFather выключите Group Privacy, иначе бот не увидит обычные сообщения."
    )

    listening_sessions: dict[int, list[str]] = {}
    recent_messages: dict[int, deque[str]] = defaultdict(
        lambda: deque(maxlen=settings.chat_session_max_messages)
    )

    def reply(message: Message, text: str) -> None:
        for chunk in split_telegram_text(text):
            bot.reply_to(message, chunk)

    def index_user_message(message: Message) -> str | None:
        text = (message.text or "").strip()
        if not text or message.from_user is None:
            return None
        if message.from_user.is_bot:
            return None
        username = display_name(message)
        date_iso = message_date_iso(message)
        record = format_record(username, date_iso, text)
        recent_messages[message.chat.id].append(record)
        if message.chat.id in listening_sessions:
            listening_sessions[message.chat.id].append(record)
        try:
            manager.index_message(
                chat_id=message.chat.id,
                message_id=message.message_id,
                user_id=message.from_user.id,
                username=username,
                text=text,
                date_iso=date_iso,
            )
        except Exception:
            logger.exception("Не удалось проиндексировать сообщение %s", message.message_id)
        return record

    def dialogue_for(chat_id: int) -> str:
        if chat_id in listening_sessions and listening_sessions[chat_id]:
            return "\n".join(listening_sessions[chat_id])
        recent = list(recent_messages[chat_id])
        return "\n".join(recent) if recent else "(пока нет сохранённых реплик)"

    @bot.message_handler(commands=["start", "help"])
    def handle_start(message: Message) -> None:
        reply(
            message,
            "Я командный помощник: слушаю чат, помню реплики и помогаю подвести итог.\n\n"
            f"/start_listening — начать запись обсуждения\n"
            f"/stop_listening — резюме сессии\n"
            f"Упомяните @{bot_username} или спросите «Что думаешь?» — отвечу по контексту.",
        )

    @bot.message_handler(commands=["start_listening"])
    def handle_start_listening(message: Message) -> None:
        chat_id = message.chat.id
        if chat_id in listening_sessions:
            reply(message, "Я уже слушаю этот чат. Чтобы закончить — /stop_listening")
            return
        listening_sessions[chat_id] = []
        logger.info("Listening ON chat_id=%s", chat_id)
        reply(
            message,
            "Слушаю чат: все сообщения до /stop_listening попадут в резюме и в память.",
        )

    @bot.message_handler(commands=["stop_listening"])
    def handle_stop_listening(message: Message) -> None:
        chat_id = message.chat.id
        session = listening_sessions.pop(chat_id, None)
        if session is None:
            reply(message, "Сейчас я не слушаю этот чат. Сначала /start_listening")
            return
        logger.info("Listening OFF chat_id=%s messages=%s", chat_id, len(session))
        if not session:
            reply(message, "Сессия пустая: за время записи никто ничего не написал.")
            return
        dialogue = "\n".join(session)
        try:
            summary = manager.summarize(dialogue=dialogue, chat_id=chat_id)
        except Exception:
            logger.exception("Ошибка суммаризации chat_id=%s", chat_id)
            reply(message, "Не смог подвести итог. Ошибка в логах, попробуйте ещё раз.")
            return
        reply(message, summary)

    @bot.message_handler(content_types=["text"])
    def handle_text(message: Message) -> None:
        cmd = command_name(message)
        if cmd in COMMANDS:
            return
        index_user_message(message)
        if message.from_user is None or message.from_user.is_bot:
            return
        if not is_addressed_to_bot(message, bot_username, bot_id):
            return
        query = (message.text or "").strip()
        logger.info("Запрос к боту chat_id=%s: %r", message.chat.id, query[:120])
        try:
            answer = manager.answer(
                query=query,
                dialogue=dialogue_for(message.chat.id),
                chat_id=message.chat.id,
            )
        except Exception:
            logger.exception("Ошибка ответа chat_id=%s", message.chat.id)
            reply(message, "Не смог ответить по контексту. Посмотрите логи и повторите вопрос.")
            return
        reply(message, answer)

    logger.info("Polling запущен. python bot.py")
    bot.infinity_polling(skip_pending=True, timeout=60)


if __name__ == "__main__":
    main()
