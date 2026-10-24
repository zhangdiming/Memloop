from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field


def request_key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params}, sort_keys=True,
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class ChatLLM:
    model: str
    base_url: str = field(default_factory=lambda: os.environ.get(
        "OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"))
    temperature: float = 0.0
    seed: int | None = 0
    max_tokens: int = 256
    log_path: str | None = None
    use_cache: bool = False
    max_retries: int = 5

    def __post_init__(self):
        from openai import OpenAI
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._lock = threading.Lock()
        self._cache: dict[str, str] = {}
        self.calls = self.cache_hits = self.replays_differ = 0
        if self.log_path and os.path.exists(self.log_path):
            with open(self.log_path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["out"]

    def __call__(self, messages: list[dict] | str, **overrides) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        params = {"temperature": self.temperature, "seed": self.seed,
                  "max_tokens": self.max_tokens, **overrides}
        key = request_key(self.model, messages, params)
        with self._lock:
            self.calls += 1
            if self.use_cache and key in self._cache:
                self.cache_hits += 1
                return self._cache[key]
        out = self._complete(messages, params)
        with self._lock:
            prev = self._cache.get(key)
            if prev is not None and prev != out:
                self.replays_differ += 1
            self._cache[key] = out
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "params": params,
                                        "messages": messages, "out": out},
                                       ensure_ascii=False) + "\n")
        return out

    def _complete(self, messages: list[dict], params: dict) -> str:
        for attempt in range(self.max_retries):
            try:
                r = self._client.chat.completions.create(model=self.model, messages=messages,
                                                         **params)
                return r.choices[0].message.content or ""
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.cache_hits,
                "replays_differ": self.replays_differ}
