"""Bedrock Agent — Claude via AWS Bedrock Converse API.

Thin wrapper: only implements the API call. All prompt engineering
lives in BaseAgent.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from microids.agents.base import BaseAgent


class BedrockAgent(BaseAgent):

    def __init__(
        self,
        model_id: Optional[str] = None,
        aws_profile: Optional[str] = None,
        aws_region: Optional[str] = None,
    ) -> None:
        self._model_id = model_id or os.environ.get(
            "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0"
        )
        self._aws_profile = aws_profile or os.environ.get("BEDROCK_AWS_PROFILE", "")
        self._aws_region = aws_region or os.environ.get("AWS_REGION", "us-east-1")

    async def _call_llm(self, messages: list[dict], **kwargs) -> str:
        import boto3

        session_kwargs = {"region_name": self._aws_region}
        if self._aws_profile:
            session_kwargs["profile_name"] = self._aws_profile
        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock-runtime")

        model_id = self._model_id
        if model_id.startswith("bedrock/"):
            model_id = model_id[len("bedrock/"):]

        # Convert chat messages to Bedrock Converse format
        system_text = ""
        converse_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        resp = client.converse(
            modelId=model_id,
            messages=converse_messages,
            system=[{"text": system_text.strip()}],
            inferenceConfig={"maxTokens": 2048, "temperature": 0.1},
        )

        content = ""
        for block in resp.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                content += block["text"]
        return content

    def model_name(self) -> str:
        return f"bedrock/{self._model_id}"
