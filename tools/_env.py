"""Shared helper to expose repo root and load .env for CLI tools."""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from .env so CLI defaults mirror the bot runtime.
load_dotenv()
