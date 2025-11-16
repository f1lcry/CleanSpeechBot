"""Configuration management for the bot application."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values


@dataclass(slots=True)
class BotConfig:
    """Application configuration loaded from environment variables or a .env file."""

    bot_token: str
    ollama_host: str
    whisper_model: str
    audio_tmp_dir: Path
    task_queue_limit: int


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

    return BotConfig(
        bot_token=str(bot_token),
        ollama_host=str(ollama_host),
        whisper_model=str(whisper_model),
        audio_tmp_dir=audio_tmp_dir,
        task_queue_limit=task_queue_limit,
    )
