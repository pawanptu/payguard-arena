"""Persistent JSON-lines cache of validated LLM replies, keyed by sha256(model, prompt version, payload).

One line per entry: {"key", "model", "prompt_version", "payload", "response"}. Loaded fully at start-up and
appended on every put (a torn last line is skipped on the next load). Thread-safe.
Default path is data/llm_cache.jsonl (eval/ is frozen); override with PAYGUARD_LLM_CACHE.
The Gemini judge has its own file, data/llm_cache_gemini.jsonl (override PAYGUARD_LLM_CACHE_GEMINI), so the
two providers' answers are never mixed.
A cache with path=None lives in memory only (tests).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "llm_cache.jsonl"
DEFAULT_GEMINI_PATH = Path(__file__).resolve().parents[2] / "data" / "llm_cache_gemini.jsonl"


def default_cache_path() -> Path:
    env = os.environ.get("PAYGUARD_LLM_CACHE", "").strip()
    return Path(env) if env else DEFAULT_PATH


def default_gemini_cache_path() -> Path:
    env = os.environ.get("PAYGUARD_LLM_CACHE_GEMINI", "").strip()
    return Path(env) if env else DEFAULT_GEMINI_PATH


def canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def cache_key(model: str, prompt_version: str, payload: dict) -> str:
    raw = canonical({"model": model, "prompt_version": prompt_version, "payload": payload})
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class LLMCache:
    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._needs_newline = False
        if self.path is not None and self.path.exists():
            raw = self.path.read_bytes()
            self._needs_newline = bool(raw) and not raw.endswith(b"\n")
            for line in raw.decode("utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                    self._data[rec["key"]] = rec["response"]
                except (ValueError, KeyError, TypeError):
                    continue  # torn or foreign line

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def get(self, key: str) -> dict | None:
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, response: dict, *, model: str, prompt_version: str, payload: dict) -> None:
        rec = {"key": key, "model": model, "prompt_version": prompt_version, "payload": payload,
               "response": response}
        with self._lock:
            self._data[key] = response
            if self.path is None:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                if self._needs_newline:
                    fh.write("\n")
                    self._needs_newline = False
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
