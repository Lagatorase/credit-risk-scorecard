"""Settings for the whole project, read from environment variables or a .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

# I load .env once here so every module sees the same settings
load_dotenv()

# project root, so relative paths work no matter where I run a script from
ROOT = Path(__file__).resolve().parent.parent

# one seed for every split, fold and model so runs are reproducible
SEED = 42


def _env(name: str, default: str) -> str:
    # I treat an empty value in .env the same as not setting it
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _path(name: str, default: str) -> Path:
    # relative paths are resolved against the project root
    p = Path(_env(name, default))
    return p if p.is_absolute() else ROOT / p


@dataclass(frozen=True)
class Settings:
    # llm backend choice and per-provider details
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "ollama"))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:3b"))
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-5-5"))
    # the customer gets the plain template unless this is "llm"; the officer always gets the checked LLM draft
    customer_explanation: str = field(default_factory=lambda: _env("CUSTOMER_EXPLANATION", "template"))

    # folders
    raw_dir: Path = field(default_factory=lambda: _path("RAW_DIR", "data/raw"))
    processed_dir: Path = field(default_factory=lambda: _path("PROCESSED_DIR", "data/processed"))
    models_dir: Path = field(default_factory=lambda: _path("MODELS_DIR", "models"))
    reports_dir: Path = field(default_factory=lambda: _path("REPORTS_DIR", "reports"))

    def with_overrides(self, **changes) -> "Settings":
        # tests use this to point the folders at a temporary synthetic dataset
        return replace(self, **changes)


def get_settings() -> Settings:
    return Settings()
