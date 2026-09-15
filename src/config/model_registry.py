"""Compatibility shim: all agents use the single GPT-5.5 configured in .env."""

from __future__ import annotations

import os
from typing import Optional


SUPPORTED_MODELS: dict[str, dict] = {
    "gpt-5.5": {"label": "GPT-5.5", "selected": True},
}


def get_model_config(model_key: Optional[str]) -> Optional[dict]:
    """Return only the system model; legacy alternative model keys are rejected."""
    if not model_key or model_key.strip().lower() != "gpt-5.5":
        return None
    return {
        "api_key": os.getenv("LLM_API_KEY", ""),
        "base_url": os.getenv("LLM_HOST", ""),
        "model": os.getenv("LLM_MODEL", "gpt-5.5"),
        "label": "GPT-5.5",
        "temperature": None,
        "binding": os.getenv("LLM_BINDING", "openai"),
    }


def agent_kwargs(model_key: Optional[str]) -> dict:
    """Keep the old call shape without enabling a second model configuration."""
    config = get_model_config(model_key)
    if not config:
        return {}
    return {
        "api_key": config["api_key"],
        "base_url": config["base_url"],
        "model": config["model"],
        "binding": config["binding"],
    }


__all__ = ["SUPPORTED_MODELS", "get_model_config", "agent_kwargs"]
