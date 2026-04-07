"""Groq Agent — Llama 3.3 70B via Groq's LPU hardware.

Thin wrapper: only implements the API call. All prompt engineering
lives in BaseAgent.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import aiohttp

from microids.agents.base import BaseAgent


class GroqAgent(BaseAgent):

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self._model = model or os.environ.get("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._base_url = "https://api.groq.com/openai/v1"

    async def _call_llm(self, messages: list[dict], **kwargs) -> str:
        if not self._api_key:
            raise RuntimeError("No GROQ_API_KEY set. Get one at console.groq.com")

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/chat/completions",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Groq API error {resp.status}: {text[:300]}")
                data = await resp.json()

        return data["choices"][0]["message"]["content"]

    def model_name(self) -> str:
        return f"groq/{self._model}"
