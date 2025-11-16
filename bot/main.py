"""Application entry point for the BotSummarizer project."""
from __future__ import annotations

import asyncio

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
    whisper_engine = WhisperEngine(
        model_name=config.whisper_model,
        cache_dir=config.audio_tmp_dir,
        language=config.whisper_language,
        temperature=config.whisper_temperature,
        device=config.whisper_device,
        ssl_cert_file=config.whisper_ca_bundle,
        allow_insecure_ssl=config.whisper_insecure_ssl,
    )
    formatting_client = FormattingLLMClient(
        host=config.ollama_host,
        model=config.formatter_model,
        system_prompt=config.formatter_prompt,
        request_timeout=config.formatter_timeout,
    )
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

    worker_count = max(1, config.task_queue_limit)
    logger.info("Starting %s task queue workers", worker_count)
    await task_queue.start_workers(worker_count)

    logger.info("Bot bootstrap completed. Starting polling loop.")
    try:
        await dispatcher.start_polling(bot)
    except Exception:  # noqa: BLE001 - log and propagate unexpected shutdowns
        logger.exception("Dispatcher polling stopped due to an error.")
        raise
    finally:
        logger.info("Polling finished. Waiting for task queue to drain.")
        await task_queue.queue.join()
        await task_queue.shutdown()
        await bot.session.close()
        logger.info("Bot shutdown completed.")


async def main() -> None:
    """Main entry point when executing ``python -m bot.main``."""

    await bootstrap()


if __name__ == "__main__":
    asyncio.run(main())
