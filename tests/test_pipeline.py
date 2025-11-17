"""Tests for the voice processing pipeline."""
from __future__ import annotations

import asyncio
import logging
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _DummyAsyncClient:
    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - simple stub
        return None

    async def generate(self, *args, **kwargs):  # pragma: no cover - simple stub
        raise RuntimeError("Not implemented")


def _dummy_whisper_model(*args, **kwargs):  # pragma: no cover - simple stub
    return types.SimpleNamespace(transcribe=lambda *t, **k: {"text": ""})


sys.modules.setdefault(
    "ollama",
    types.SimpleNamespace(
        AsyncClient=_DummyAsyncClient,
        RequestError=RuntimeError,
        ResponseError=RuntimeError,
    ),
)

sys.modules.setdefault(
    "whisper",
    types.SimpleNamespace(load_model=_dummy_whisper_model),
)

from bot.services.pipeline import VoicePipeline


class DummyAudioProcessor:
    """Minimal audio processor stub used for pipeline tests."""

    def __init__(self, tmp_dir: Path) -> None:
        self.tmp_dir = tmp_dir
        self.cleanup_calls: list[Path] = []

    def convert_to_wav(self, source_path: Path) -> Path:
        wav_path = self.tmp_dir / f"{Path(source_path).stem}.wav"
        wav_path.write_text("converted", encoding="utf-8")
        return wav_path

    def validate_audio(self, wav_path: Path) -> None:  # pragma: no cover - stub
        return None

    def cleanup(self, wav_path: Path) -> None:
        self.cleanup_calls.append(Path(wav_path))


class DummyWhisperEngine:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, wav_path: Path) -> str:
        return self.transcript


class SuccessfulFormatter:
    def __init__(self, response: str) -> None:
        self.response = response

    async def format_text(self, text: str) -> str:
        return self.response


class FailingFormatter:
    async def format_text(self, text: str) -> str:  # pragma: no cover - stub
        raise RuntimeError("formatter unavailable")


class ImmediateTaskQueue:
    async def enqueue(self, task_factory):
        await task_factory()


def test_process_voice_success_keeps_final_log(tmp_path, caplog):
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    source_path = tmp_dir / "voice.ogg"
    source_path.write_bytes(b"data")

    pipeline = VoicePipeline(
        audio_processor=DummyAudioProcessor(tmp_dir),
        whisper_engine=DummyWhisperEngine("hello there"),
        formatting_client=SuccessfulFormatter("  formatted text  "),
        task_queue=ImmediateTaskQueue(),
    )

    caplog.set_level(logging.INFO, logger="bot.pipeline")

    result = asyncio.run(pipeline.process_voice(source_path))

    assert result == "formatted text"
    assert any("Pipeline finished" in record.message for record in caplog.records)


def test_process_voice_falls_back_to_transcript_on_formatting_error(tmp_path, caplog):
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    source_path = tmp_dir / "voice.ogg"
    source_path.write_bytes(b"data")

    audio_processor = DummyAudioProcessor(tmp_dir)
    pipeline = VoicePipeline(
        audio_processor=audio_processor,
        whisper_engine=DummyWhisperEngine("  raw transcript  "),
        formatting_client=FailingFormatter(),
        task_queue=ImmediateTaskQueue(),
    )

    caplog.set_level(logging.INFO, logger="bot.pipeline")

    result = asyncio.run(pipeline.process_voice(source_path))

    assert result == "raw transcript"
    assert any("Formatting skipped" in record.message for record in caplog.records)
    assert any("Pipeline finished" in record.message for record in caplog.records)
    assert audio_processor.cleanup_calls, "cleanup should still be invoked"
