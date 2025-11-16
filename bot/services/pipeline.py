"""Voice message processing pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

from bot.utils.audio import AudioProcessor
from bot.utils.formatting_llm import FormattingLLMClient
from bot.utils.whisper_engine import WhisperEngine
from bot.utils.workers import TaskQueueManager

logger = logging.getLogger("bot.pipeline")


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
        """Process a voice message through the entire pipeline and return formatted text.

        TODO: Implement proper task scheduling, error handling, and resource cleanup.
        """

        logger.info("Starting voice processing for: %s", audio_path)

        cleanup_candidate: Path | None = None
        converted_path = audio_path
        try:
            converted_path = self.audio_processor.convert_to_wav(source_path=audio_path)
            self.audio_processor.validate_audio(audio_path=converted_path)
            if converted_path != audio_path:
                cleanup_candidate = converted_path
        except NotImplementedError:
            logger.debug("Audio processing is not yet implemented.")
            converted_path = audio_path

        try:
            transcript = self.whisper_engine.transcribe(audio_path=converted_path)
        except NotImplementedError:
            logger.debug("Whisper transcription is not yet implemented.")
            transcript = ""
        except Exception as exc:
            logger.error("Whisper transcription failed: %s", exc)
            raise
        else:
            if cleanup_candidate is not None:
                self.audio_processor.cleanup(audio_path=cleanup_candidate)

        try:
            formatted_text = await self.formatting_client.format_text(text=transcript)
        except NotImplementedError:
            logger.debug("LLM formatting is not yet implemented.")
            formatted_text = transcript or "[transcription pending]"

        # TODO: Integrate TaskQueueManager to process the above steps asynchronously.
        logger.info("Voice processing completed with placeholder result.")
        return formatted_text
