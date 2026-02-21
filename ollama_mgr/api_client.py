"""Ollama API client: health check, list models, generate, delete."""

from __future__ import annotations

import json
import requests
from typing import Any, Generator, Optional


class OllamaClient:
    """Minimal client for Ollama HTTP API (default base http://localhost:11434)."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, **kwargs: Any) -> requests.Response:
        return requests.get(
            f"{self.base_url}{path}",
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def _post(
        self,
        path: str,
        json: Optional[dict] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> requests.Response:
        return requests.post(
            f"{self.base_url}{path}",
            json=json,
            stream=stream,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def _delete(self, path: str, json: Optional[dict] = None, **kwargs: Any) -> requests.Response:
        return requests.delete(
            f"{self.base_url}{path}",
            json=json,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )

    def health(self) -> tuple[bool, Optional[str], Optional[dict]]:
        """
        Check if Ollama is reachable. Returns (ok, error_message, version_info).
        Uses GET /api/version or GET /api/tags as health probe.
        """
        try:
            r = self._get("/api/version", timeout=5)
            r.raise_for_status()
            return True, None, r.json()
        except requests.RequestException as e:
            return False, str(e), None

    def list_models(self) -> tuple[bool, Optional[str], list[dict]]:
        """
        List all models. Returns (ok, error_message, list of model dicts).
        Each dict has at least 'name', and may have 'details', 'size', etc.
        """
        try:
            r = self._get("/api/tags", timeout=10)
            r.raise_for_status()
            data = r.json()
            models = data.get("models") or []
            return True, None, models
        except requests.RequestException as e:
            return False, str(e), []

    def chat(
        self,
        model: str,
        messages: list[dict],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> tuple[bool, Optional[str], Optional[dict], Optional[Generator[dict, None, None]]]:
        """
        Chat completion (POST /api/chat). Use for models that expect a message format.
        messages: list of {"role": "user"|"assistant"|"system", "content": "..."}.
        Yields/returns chunks with "response" (content delta) and eval_* so callers can reuse generate logic.
        """
        payload: dict = {"model": model, "messages": messages}
        if options:
            payload["options"] = options
        try:
            r = self._post("/api/chat", json=payload, stream=stream)
            r.raise_for_status()
            if stream:
                def gen():
                    for line in r.iter_lines():
                        if line:
                            obj = json.loads(line)
                            msg = obj.get("message") or {}
                            content = msg.get("content") if isinstance(msg, dict) else None
                            out: dict = {}
                            if isinstance(content, str):
                                out["response"] = content
                            if "eval_count" in obj:
                                out["eval_count"] = obj["eval_count"]
                            if "eval_duration" in obj:
                                out["eval_duration"] = obj["eval_duration"]
                            if out:
                                yield out
                return True, None, None, gen()
            data = r.json()
            msg = data.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            merged = {"response": content if isinstance(content, str) else ""}
            for k in ("eval_count", "eval_duration"):
                if k in data:
                    merged[k] = data[k]
            return True, None, merged, None
        except requests.RequestException as e:
            return False, str(e), None, None

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        options: Optional[dict] = None,
        use_chat_api: bool = False,
    ) -> tuple[bool, Optional[str], Optional[dict], Optional[Generator[dict, None, None]]]:
        """
        Generate completion. If use_chat_api=True, sends prompt as a user message via /api/chat
        (for models like DeepSeek-R1 that expect chat format). Otherwise uses /api/generate.
        If stream=False, returns (ok, err, response_json, None).
        If stream=True, returns (ok, err, None, stream_generator).
        """
        if use_chat_api:
            return self.chat(
                model,
                [{"role": "user", "content": prompt}],
                stream=stream,
                options=options,
            )
        payload: dict = {"model": model, "prompt": prompt}
        if options:
            payload["options"] = options
        try:
            r = self._post("/api/generate", json=payload, stream=stream)
            r.raise_for_status()
            if stream:
                def gen():
                    for line in r.iter_lines():
                        if line:
                            yield json.loads(line)
                return True, None, None, gen()
            # Server may send NDJSON even with stream=False; merge all lines into one dict
            try:
                data = r.json()
                return True, None, data, None
            except json.JSONDecodeError:
                merged: dict = {}
                for line in r.text.splitlines():
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            delta = obj.get("response")
                            if isinstance(delta, str):
                                merged["response"] = (merged.get("response") or "") + delta
                            for k, v in obj.items():
                                if k != "response":
                                    merged[k] = v
                        except json.JSONDecodeError:
                            pass
                return True, None, merged if merged else None, None
        except requests.RequestException as e:
            return False, str(e), None, None

    def delete_model(self, name: str) -> tuple[bool, Optional[str]]:
        """Delete a model by name (e.g. 'llama2' or 'llama2:7b'). Returns (ok, error_message)."""
        try:
            r = self._delete("/api/delete", json={"name": name}, timeout=60)
            r.raise_for_status()
            return True, None
        except requests.RequestException as e:
            return False, str(e)
