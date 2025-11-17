"""Voice message processing pipeline."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from bot.utils.audio import AudioProcessor
from bot.utils.formatting_llm import FormattingLLMClient
from bot.utils.whisper_engine import WhisperEngine
from bot.utils.workers import TaskQueueManager

logger = logging.getLogger("bot.pipeline")


class VoicePipelineError(RuntimeError):
    """Base exception for voice pipeline failures."""


class AudioConversionError(VoicePipelineError):
    """Audio conversion step failed."""


class AudioValidationError(VoicePipelineError):
    """Audio validation step failed."""


class TranscriptionError(VoicePipelineError):
    """Whisper transcription step failed."""


class FormattingError(VoicePipelineError):
    """LLM formatting step failed."""


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

    @property
    def tmp_dir(self) -> Path:
        """Expose the temporary directory configured for audio artifacts."""

        return self.audio_processor.tmp_dir

    async def process_voice(self, audio_path: Path) -> str:
        """Process a voice message through the entire pipeline and return formatted text."""

        validated_path = self._ensure_within_tmp_dir(audio_path)
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[str] = loop.create_future()

        async def runner() -> None:
            try:
                result = await self._run_pipeline(validated_path)
            except Exception as exc:  # noqa: BLE001 - propagate upstream with context
                if not result_future.done():
                    result_future.set_exception(exc)
            else:
                if not result_future.done():
                    result_future.set_result(result)

        await self.task_queue.enqueue(runner)
        logger.info("Voice task enqueued for %s", validated_path.name)
        return await result_future

    def _ensure_within_tmp_dir(self, audio_path: Path) -> Path:
        """Ensure the provided audio path is inside the configured temporary directory."""

        path = Path(audio_path).resolve()
        tmp_dir = self.audio_processor.tmp_dir.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        try:
            path.relative_to(tmp_dir)
        except ValueError as exc:  # pragma: no cover - defensive check
            raise AudioValidationError(
                "Audio path must reside inside the configured temporary directory."
            ) from exc
        return path

    async def _run_pipeline(self, source_path: Path) -> str:
        """Execute the conversion → transcription → formatting pipeline."""

        logger.info("Pipeline started for %s", source_path.name)
        wav_path: Path | None = None
        transcript: str = ""

        try:
            wav_path = await self._convert(source_path)
            await self._validate(wav_path)
            transcript = await self._transcribe(wav_path)
            formatted_text = ""
            try:
                formatted_text = await self._format(transcript)
            except FormattingError as exc:
                logger.warning(
                    "Formatting skipped for %s due to formatter failure. Using raw transcript.",
                    source_path.name,
                    exc_info=exc,
                )
                result = transcript.strip()
            except Exception as exc:  # pragma: no cover - defensive fallback for Ollama errors
                logger.warning(
                    "Formatting skipped for %s due to unexpected formatter error. Using raw transcript.",
                    source_path.name,
                    exc_info=exc,
                )
                result = transcript.strip()
            else:
                result = formatted_text.strip()

            transcript = ""
            formatted_text = ""
            logger.info("Pipeline finished for %s", source_path.name)
            return result
        finally:
            if wav_path is not None:
                await asyncio.to_thread(self.audio_processor.cleanup, wav_path)
            transcript = ""

    async def _convert(self, source_path: Path) -> Path:
        try:
            wav_path = await asyncio.to_thread(
                self.audio_processor.convert_to_wav,
                source_path,
            )
        except Exception as exc:  # noqa: BLE001 - ffmpeg or filesystem failure
            logger.exception("Audio conversion failed for %s", source_path)
            raise AudioConversionError("Не удалось подготовить аудио к обработке.") from exc

        logger.debug("Converted %s to %s", source_path.name, wav_path.name)
        return wav_path

    async def _validate(self, wav_path: Path) -> None:
        try:
            await asyncio.to_thread(self.audio_processor.validate_audio, wav_path)
        except Exception as exc:  # noqa: BLE001 - invalid wav
            logger.exception("Audio validation failed for %s", wav_path)
            raise AudioValidationError("Аудио-файл не подходит для распознавания.") from exc

        logger.debug("Audio validation passed for %s", wav_path.name)

    async def _transcribe(self, wav_path: Path) -> str:
        try:
            transcript = await asyncio.to_thread(self.whisper_engine.transcribe, wav_path)
        except Exception as exc:  # noqa: BLE001 - whisper failure
            logger.exception("Transcription failed for %s", wav_path)
            raise TranscriptionError("Не удалось распознать голосовое сообщение.") from exc

        logger.info("Transcription complete for %s", wav_path.name)
        return transcript

    async def _format(self, transcript: str) -> str:
        try:
            formatted_text = await self.formatting_client.format_text(text=transcript)
        except Exception as exc:  # noqa: BLE001 - ollama failure
            logger.exception("Formatting failed for transcript")
            raise FormattingError("Не удалось отформатировать расшифровку.") from exc

        logger.info("Formatting complete")
        return formatted_text
