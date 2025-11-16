"""Voice message processing pipeline."""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path

from bot.utils.audio import AudioProcessor
from bot.utils.formatting_llm import FormattingLLMClient
from bot.utils.whisper_engine import WhisperEngine
from bot.utils.workers import TaskQueueManager

logger = logging.getLogger("bot.pipeline")

_FALLBACK_RESPONSE = "Не удалось обработать голосовое сообщение, попробуйте позже."


class VoicePipeline:
    """Orchestrate audio processing, transcription, and formatting tasks."""

    def __init__(
        self,
        audio_processor: AudioProcessor,
        whisper_engine: WhisperEngine,
        formatting_client: FormattingLLMClient,
        task_queue: TaskQueueManager,
    ) -> None:
        """Store dependencies required for the voice processing pipeline."""

        self.audio_processor = audio_processor
        self.whisper_engine = whisper_engine
        self.formatting_client = formatting_client
        self.task_queue = task_queue

    async def process_voice(self, audio_path: Path) -> str:
        """Process a voice message through the entire pipeline and return formatted text."""

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[str] = loop.create_future()

        async def runner() -> None:
            try:
                result = await self._run_job(audio_path)
            except Exception:  # pragma: no cover - safeguarded by _run_job.
                logger.exception("Voice pipeline job raised unexpectedly.")
                result = _FALLBACK_RESPONSE
            if not result_future.done():
                result_future.set_result(result)

        await self.task_queue.enqueue(lambda: runner())
        return await result_future

    async def _run_job(self, audio_path: Path) -> str:
        job_id = uuid.uuid4().hex
        start = time.perf_counter()
        converted_path: Path | None = None

        try:
            converted_path = self.audio_processor.convert_to_wav(source_path=audio_path)
            self.audio_processor.validate_audio(audio_path=converted_path)
            transcript = self.whisper_engine.transcribe(audio_path=converted_path)
            formatted_text = await self.formatting_client.format_text(text=transcript)
            duration = time.perf_counter() - start
            logger.info("Voice pipeline job %s finished in %.2fs", job_id, duration)
            return formatted_text or _FALLBACK_RESPONSE
        except Exception as exc:
            logger.exception("Voice pipeline job %s failed: %s", job_id, exc)
            return _FALLBACK_RESPONSE
        finally:
            if converted_path is not None:
                self.audio_processor.cleanup(converted_path)
