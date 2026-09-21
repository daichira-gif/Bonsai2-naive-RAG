from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI


@dataclass(frozen=True)
class Bonsai2Config:
    base_url: str = "http://127.0.0.1:8000/v1"
    health_url: str = "http://127.0.0.1:8000/health"
    model: str = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
    temperature: float = 0.0
    max_tokens: int = 512
    enable_thinking: bool = False
    thinking_budget: int = 1024


class Bonsai2Generator:
    def __init__(self, config: Bonsai2Config) -> None:
        self.config = config
        self.client = OpenAI(base_url=config.base_url, api_key="dummy")

    def health(self, timeout: float = 3.0) -> dict:
        with urllib.request.urlopen(self.config.health_url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _extra_body(self) -> dict:
        body = {"enable_thinking": self.config.enable_thinking}
        if self.config.enable_thinking:
            body["thinking_budget"] = self.config.thinking_budget
        return body

    def generate(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=False,
            extra_body=self._extra_body(),
        )
        return response.choices[0].message.content or ""

    def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
            extra_body=self._extra_body(),
        )

        for event in response:
            if not event.choices:
                continue
            content = event.choices[0].delta.content
            if content:
                yield content
