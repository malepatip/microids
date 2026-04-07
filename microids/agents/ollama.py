"""Ollama Agent — local LLM via Ollama Docker.

Thin wrapper: only implements the API call. All prompt engineering
lives in BaseAgent.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import aiohttp

from microids.agents.base import BaseAgent


class OllamaAgent(BaseAgent):

    def __init__(self, model: str = "qwen3:8b", base_url: Optional[str] = None) -> None:
        self._model = model
        self._base_url = base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")

    async def _call_llm(self, messages: list[dict], **kwargs) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 2048},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Ollama error {resp.status}: {text[:200]}")
                data = await resp.json()

        return data.get("message", {}).get("content", "")

    def model_name(self) -> str:
        return f"ollama/{self._model}"
