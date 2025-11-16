"""Configuration management for the bot application."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values

from bot.utils.formatting_llm import load_formatter_prompt


@dataclass(slots=True)
class BotConfig:
    """Application configuration loaded from environment variables or a .env file."""

    bot_token: str
    ollama_host: str
    whisper_model: str
    audio_tmp_dir: Path
    task_queue_limit: int
    whisper_language: Optional[str]
    whisper_temperature: float
    whisper_device: Optional[str]
    whisper_ca_bundle: Optional[Path]
    whisper_insecure_ssl: bool
    formatter_model: str
    formatter_prompt: str
    formatter_timeout: float


def load_config(env_file: Optional[str | Path] = None) -> BotConfig:
    """Load the application configuration, optionally from a specific ``.env`` file."""

    env_path = Path(env_file) if env_file is not None else Path(".env")
    file_values = dotenv_values(env_path)
    values = {**file_values, **os.environ}

    try:
        bot_token = values["BOT_TOKEN"]
        ollama_host = values["OLLAMA_HOST"]
        whisper_model = values["WHISPER_MODEL"]
        audio_tmp_dir_raw = values["AUDIO_TMP_DIR"]
        task_queue_limit_raw = values["TASK_QUEUE_LIMIT"]
    except KeyError as missing:
        raise RuntimeError(f"Missing configuration key: {missing}") from missing

    if not all([bot_token, ollama_host, whisper_model, audio_tmp_dir_raw, task_queue_limit_raw]):
        raise RuntimeError("Configuration values must not be empty.")

    audio_tmp_dir = Path(str(audio_tmp_dir_raw))
    task_queue_limit = int(task_queue_limit_raw)
    whisper_language = values.get("WHISPER_LANGUAGE") or None
    whisper_temperature = float(values.get("WHISPER_TEMPERATURE", "0.0"))
    whisper_device = values.get("WHISPER_DEVICE") or None
    whisper_ca_bundle_raw = values.get("WHISPER_CA_BUNDLE")
    whisper_ca_bundle = Path(whisper_ca_bundle_raw).expanduser() if whisper_ca_bundle_raw else None
    whisper_insecure_ssl = _parse_bool(values.get("WHISPER_INSECURE_SSL", "false"))
    formatter_model = values.get("FORMATTER_MODEL", "llama3.1:8b")
    formatter_timeout = float(values.get("FORMATTER_TIMEOUT", "120"))
    formatter_prompt = load_formatter_prompt(values)

    return BotConfig(
        bot_token=str(bot_token),
        ollama_host=str(ollama_host),
        whisper_model=str(whisper_model),
        audio_tmp_dir=audio_tmp_dir,
        task_queue_limit=task_queue_limit,
        whisper_language=whisper_language,
        whisper_temperature=whisper_temperature,
        whisper_device=whisper_device,
        whisper_ca_bundle=whisper_ca_bundle,
        whisper_insecure_ssl=whisper_insecure_ssl,
        formatter_model=str(formatter_model),
        formatter_prompt=formatter_prompt,
        formatter_timeout=formatter_timeout,
    )


def _parse_bool(value: Optional[str]) -> bool:
    """Parse typical truthy string values into a boolean."""

    if value is None:
        return False

    return str(value).strip().lower() in {"1", "true", "yes", "on"}
