"""Application entry point for the BotSummarizer project."""
from __future__ import annotations

import asyncio
from pathlib import Path

from aiogram import Bot, Dispatcher

from bot.config import BotConfig, load_config
from bot.handlers.text_handler import text_router
from bot.handlers.voice_handler import configure_pipeline, voice_router
from bot.services.logger import setup_logging
from bot.services.pipeline import VoicePipeline
from bot.utils.audio import AudioProcessor
from bot.utils.formatting_llm import FormattingLLMClient
from bot.utils.whisper_engine import WhisperEngine
from bot.utils.workers import TaskQueueManager


async def bootstrap() -> None:
    """Initialize application components and log the bootstrap sequence."""

    config: BotConfig = load_config()
    logger = setup_logging()
    logger.info("Configuration loaded successfully.")

    bot = Bot(token=config.bot_token)
    dispatcher = Dispatcher()

    audio_processor = AudioProcessor(tmp_dir=config.audio_tmp_dir)
    whisper_engine = WhisperEngine(model_name=config.whisper_model, cache_dir=Path("models/whisper_cache"))
    formatting_client = FormattingLLMClient(host=config.ollama_host, model="llama3.1-8b")
    task_queue = TaskQueueManager(maxsize=config.task_queue_limit)
    pipeline = VoicePipeline(
        audio_processor=audio_processor,
        whisper_engine=whisper_engine,
        formatting_client=formatting_client,
        task_queue=task_queue,
    )

    configure_pipeline(pipeline)
    dispatcher.include_router(text_router)
    dispatcher.include_router(voice_router)

    logger.info("Bot bootstrap completed (stub). Aiogram polling not started in this skeleton.")

    await bot.session.close()


async def main() -> None:
    """Main entry point when executing ``python -m bot.main``."""

    await bootstrap()


if __name__ == "__main__":
    asyncio.run(main())
