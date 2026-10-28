"""Pinned GPU inference adapters for the study.

The module deliberately imports only the standard library at import time.  A
base installation can inspect configuration without installing PyTorch,
Transformers, Hugging Face Hub, requests, or vLLM.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import re
import threading
import time
from pathlib import PurePosixPath
from typing import Any, Mapping
from urllib.parse import urlsplit


class BackendError(RuntimeError):
    """A backend could not safely verify or complete a model operation."""

    def __init__(self, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.details = copy.deepcopy(dict(details)) if details is not None else None


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_THINK_RE = re.compile(r"<\s*/?\s*think\s*>", re.IGNORECASE)
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)


def _model_identity(model: Mapping[str, Any]) -> tuple[str, str, str]:
    repo = model.get("repo")
    revision = model.get("revision")
    if not isinstance(repo, str) or not repo.strip() or any(c.isspace() for c in repo):
        raise BackendError("Model config must contain a valid repo identifier")
    if not isinstance(revision, str) or not _SHA_RE.fullmatch(revision):
        raise BackendError("Model config must contain a pinned 40-character commit SHA")
    return repo, revision, f"{repo}@{revision}"


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise BackendError(f"{label} must be a {qualifier} integer")
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BackendError(f"{label} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise BackendError(f"{label} is outside the supported range")
    if maximum is not None and number > maximum:
        raise BackendError(f"{label} is outside the supported range")
    return number


def _json_copy(value: Any, label: str) -> Any:
    try:
        # Reject NaN/Infinity and non-JSON payloads before they reach a backend.
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError):
        raise BackendError(f"{label} must be JSON-compatible") from None


def _validate_timeout(timeout: Any) -> float:
    return _finite_number(timeout, "timeout", minimum=0.001)


class VLLMGenerator:
    """OpenAI-compatible vLLM client for one fixed Qwen generator checkpoint.

    Construction is side-effect free. Call :meth:`verify` once the server is
    ready and before token counting or generation. The request path performs no
    retries and refuses redirects so the verified model and API key cannot be
    silently forwarded elsewhere.
    """

    def __init__(self, model: dict, service: dict, session: Any = None):
        if not isinstance(model, Mapping) or not isinstance(service, Mapping):
            raise BackendError("Model and service config must be objects")
        self.model = copy.deepcopy(dict(model))
        self.service = copy.deepcopy(dict(service))
        self.repo, self.revision, self.served_model_name = _model_identity(self.model)

        base_url = self.service.get("base_url")
        if not isinstance(base_url, str):
            raise BackendError("Service config must contain a base_url ending in /v1")
        parsed = urlsplit(base_url.rstrip("/"))
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or not parsed.path.rstrip("/").endswith("/v1")
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise BackendError("Service base_url must be an HTTP(S) URL ending in /v1")
        self.base_url = base_url.rstrip("/")
        # vLLM's OpenAI routes live under /v1, while /tokenize is a native
        # server-root endpoint. Preserve any deployment prefix before /v1.
        self.server_root = self.base_url[:-3]

        api_key_env = self.service.get("api_key_env")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise BackendError("api_key_env must be a non-empty environment variable name")

        template_kwargs = self.model.get("chat_template_kwargs")
        if self.model.get("thinking") is not True or not isinstance(template_kwargs, Mapping) or template_kwargs.get("enable_thinking") is not True:
            raise BackendError("Generator config must explicitly enable native thinking")
        self.chat_template_kwargs = _json_copy(dict(template_kwargs), "chat_template_kwargs")

        self._session = session
        self._verified = False

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                requests = importlib.import_module("requests")
                self._session = requests.Session()
            except Exception:
                raise BackendError("The HTTP client dependency is unavailable") from None
        return self._session

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        api_key_env = self.service.get("api_key_env")
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise BackendError("Configured API key environment variable is unset")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        timeout: Any,
        payload: Any = None,
        server_root: bool = False,
    ) -> dict[str, Any]:
        timeout_value = _validate_timeout(timeout)
        endpoint_base = self.server_root if server_root else self.base_url
        url = f"{endpoint_base}/{path.lstrip('/')}"
        session = self._get_session()
        try:
            if method == "GET":
                response = session.get(
                    url,
                    headers=self._headers(),
                    timeout=timeout_value,
                    allow_redirects=False,
                )
            else:
                response = session.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout_value,
                    allow_redirects=False,
                )
        except BackendError:
            raise
        except Exception:
            raise BackendError("Model service request failed") from None

        status = getattr(response, "status_code", None)
        if not isinstance(status, int) or not 200 <= status < 300:
            label = f"HTTP {status}" if isinstance(status, int) else "an invalid HTTP status"
            raise BackendError(f"Model service returned {label}")
        try:
            body = response.json()
        except Exception:
            raise BackendError("Model service returned invalid JSON") from None
        if not isinstance(body, dict):
            raise BackendError("Model service returned an unsupported response")
        return body

    def verify(self, timeout: float = 30) -> dict[str, Any]:
        """Verify the only advertised model ID exactly matches the pinned alias."""
        self._verified = False
        response = self._request_json("GET", "models", timeout=timeout)
        data = response.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise BackendError("Model service did not advertise exactly one verifiable model")
        model_id = data[0].get("id")
        if model_id != self.served_model_name:
            raise BackendError("Model service identity does not match the pinned checkpoint")
        self._verified = True
        return {"model": model_id, "verified": True}

    def _require_verified(self) -> None:
        if not self._verified:
            raise BackendError("Call verify() successfully before using the model service")

    def _tokenization_body(self, messages: Any) -> dict[str, Any]:
        safe_messages = _json_copy(messages, "messages")
        if not isinstance(safe_messages, list) or not safe_messages or any(not isinstance(m, dict) for m in safe_messages):
            raise BackendError("messages must be a non-empty list of objects")
        return {
            "model": self.served_model_name,
            "messages": safe_messages,
            "chat_template_kwargs": copy.deepcopy(self.chat_template_kwargs),
        }

    def count_tokens(self, messages: list, timeout: float = 30) -> int:
        """Count the exact server-side rendered chat input using vLLM /tokenize."""
        self._require_verified()
        response = self._request_json(
            "POST",
            "tokenize",
            timeout=timeout,
            payload=self._tokenization_body(messages),
            server_root=True,
        )
        count = response.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise BackendError("Model service returned an unverifiable token count")
        return count

    def _sampling_parameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key in _SAMPLING_KEYS:
            if key not in self.model:
                continue
            value = self.model[key]
            if key == "top_k":
                params[key] = _positive_int(value, key, allow_zero=True)
            else:
                number = _finite_number(
                    value,
                    key,
                    minimum=0 if key == "temperature" else None,
                )
                if key in ("top_p", "min_p") and not 0 <= number <= 1:
                    raise BackendError(f"{key} is outside the supported range")
                if key == "repetition_penalty" and number <= 0:
                    raise BackendError("repetition_penalty must be positive")
                params[key] = number
        return params

    @staticmethod
    def _usage(response: Mapping[str, Any]) -> tuple[int, int, int | None]:
        usage = response.get("usage")
        if not isinstance(usage, Mapping):
            raise BackendError("Model service returned invalid token usage")

        def required_count(names: tuple[str, ...]) -> int:
            present = [usage[name] for name in names if name in usage]
            if not present:
                raise BackendError("Model service returned invalid token usage")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in present):
                raise BackendError("Model service returned invalid token usage")
            if len(set(present)) != 1:
                raise BackendError("Model service returned conflicting token usage")
            return present[0]

        input_tokens = required_count(("prompt_tokens", "input_tokens"))
        output_tokens = required_count(("completion_tokens", "output_tokens"))
        details = usage.get("completion_tokens_details")
        if details is not None and not isinstance(details, Mapping):
            raise BackendError("Model service returned invalid reasoning token usage")
        reasoning_values = []
        if isinstance(details, Mapping) and "reasoning_tokens" in details:
            reasoning_values.append(details["reasoning_tokens"])
        if "reasoning_tokens" in usage:
            reasoning_values.append(usage["reasoning_tokens"])
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in reasoning_values):
            raise BackendError("Model service returned invalid reasoning token usage")
        if len(set(reasoning_values)) > 1:
            raise BackendError("Model service returned conflicting reasoning token usage")
        reasoning_tokens = reasoning_values[0] if reasoning_values else None
        if reasoning_tokens is not None and reasoning_tokens > output_tokens:
            raise BackendError("Model service returned invalid reasoning token usage")
        return input_tokens, output_tokens, reasoning_tokens

    def generate(
        self,
        messages: list,
        seed: int,
        max_new_tokens: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Generate one response, preserving native reasoning and final text separately."""
        self._require_verified()
        body = self._tokenization_body(messages)
        body.update(self._sampling_parameters())
        body["seed"] = _positive_int(seed, "seed", allow_zero=True)
        body["max_tokens"] = _positive_int(max_new_tokens, "max_new_tokens")
        if "reasoning_effort" in self.model:
            effort = self.model["reasoning_effort"]
            if not isinstance(effort, str) or not effort:
                raise BackendError("reasoning_effort must be a non-empty string")
            body["reasoning_effort"] = effort

        started = time.perf_counter()
        response = self._request_json(
            "POST", "chat/completions", timeout=timeout, payload=body
        )
        seconds = time.perf_counter() - started
        if response.get("model") != self.served_model_name:
            raise BackendError("Generation response identity does not match the pinned checkpoint")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise BackendError("Model service returned an unsupported completion response")
        item = choices[0]
        finish_reason = item.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise BackendError("Model service returned no valid finish reason")
        message = item.get("message")
        if not isinstance(message, dict):
            raise BackendError("Model service returned an unsupported completion message")
        tool_calls = message.get("tool_calls")
        if (tool_calls is not None and tool_calls != []) or finish_reason == "tool_calls" or message.get("function_call"):
            raise BackendError("Tool calls are unsupported by the generator adapter")
        content = message.get("content")
        if content is None and finish_reason == "length":
            # Preserve the length finish and usage so orchestration can record
            # a truncated generation without treating it as a transport error.
            content = ""
        if not isinstance(content, str) or (not content.strip() and finish_reason != "length"):
            raise BackendError("Model service returned no final content")
        if _THINK_RE.search(content):
            raise BackendError("Final content contains unparsed native thinking markers")

        reason_content = message.get("reasoning_content")
        reason_legacy = message.get("reasoning")
        if reason_content is not None and not isinstance(reason_content, str):
            raise BackendError("Model service returned invalid reasoning content")
        if reason_legacy is not None and not isinstance(reason_legacy, str):
            raise BackendError("Model service returned invalid reasoning content")
        if reason_content is not None and reason_legacy is not None and reason_content != reason_legacy:
            raise BackendError("Model service returned ambiguous reasoning fields")
        reasoning = reason_content if reason_content is not None else reason_legacy
        input_tokens, output_tokens, reasoning_tokens = self._usage(response)

        return {
            "content": content,
            "reasoning": reasoning,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "finish_reason": finish_reason,
            "raw_response": response,
            "seconds": seconds,
        }


class OllamaGenerator:
    """Loopback Ollama client for one named Qwen generator.

    Ollama has no separate tokenize route in the pinned 0.40 server. Prompt
    size is the server's ``prompt_eval_count`` from a one-token preflight that
    uses the same ``think`` setting as generation. Recorded usage still comes
    from the real generation response.
    """

    def __init__(self, model: dict, service: dict, session: Any = None):
        if not isinstance(model, Mapping) or not isinstance(service, Mapping):
            raise BackendError("Model and service config must be objects")
        self.model = copy.deepcopy(dict(model))
        self.service = copy.deepcopy(dict(service))
        name = self.model.get("ollama_model")
        if not isinstance(name, str) or not name.strip() or any(c.isspace() for c in name):
            raise BackendError("Ollama generator requires ollama_model")
        self.model_name = name
        template_kwargs = self.model.get("chat_template_kwargs")
        if self.model.get("thinking") is not True or not isinstance(template_kwargs, Mapping) or template_kwargs.get("enable_thinking") is not True:
            raise BackendError("Generator config must explicitly enable native thinking")
        base_url = self.service.get("base_url")
        if not isinstance(base_url, str):
            raise BackendError("Ollama base_url must be a loopback origin")
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in ("http", "https")
            or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise BackendError("Ollama base_url must be a loopback origin without a path")
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.num_ctx = _positive_int(self.service.get("num_ctx"), "num_ctx")
        self._session = session
        self._verified = False

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                requests = importlib.import_module("requests")
                self._session = requests.Session()
            except Exception:
                raise BackendError("The HTTP client dependency is unavailable") from None
        return self._session

    def _request_json(self, method: str, path: str, *, timeout: Any, payload: Any = None) -> dict[str, Any]:
        timeout_value = _validate_timeout(timeout)
        url = f"{self.origin}/{path.lstrip('/')}"
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        try:
            if method == "GET":
                response = self._get_session().get(url, headers=headers, timeout=timeout_value, allow_redirects=False)
            else:
                response = self._get_session().post(url, headers=headers, json=payload, timeout=timeout_value, allow_redirects=False)
        except BackendError:
            raise
        except Exception:
            raise BackendError("Model service request failed") from None
        status = getattr(response, "status_code", None)
        if not isinstance(status, int) or not 200 <= status < 300:
            label = f"HTTP {status}" if isinstance(status, int) else "an invalid HTTP status"
            raise BackendError(f"Model service returned {label}")
        try:
            body = response.json()
        except Exception:
            raise BackendError("Model service returned invalid JSON") from None
        if not isinstance(body, dict):
            raise BackendError("Model service returned an unsupported response")
        return body

    def _messages(self, messages: Any) -> list:
        safe_messages = _json_copy(messages, "messages")
        if not isinstance(safe_messages, list) or not safe_messages or any(not isinstance(item, dict) for item in safe_messages):
            raise BackendError("messages must be a non-empty list of objects")
        return safe_messages

    def _options(self, *, seed: int | None = None, max_new_tokens: int | None = None) -> dict[str, Any]:
        options: dict[str, Any] = {"num_ctx": self.num_ctx}
        for key in _SAMPLING_KEYS:
            if key not in self.model:
                continue
            value = self.model[key]
            target = "repeat_penalty" if key == "repetition_penalty" else key
            if key == "top_k":
                options[target] = _positive_int(value, key, allow_zero=True)
            else:
                number = _finite_number(value, key, minimum=0 if key == "temperature" else None)
                if key in ("top_p", "min_p") and not 0 <= number <= 1:
                    raise BackendError(f"{key} is outside the supported range")
                if key == "repetition_penalty" and number <= 0:
                    raise BackendError("repetition_penalty must be positive")
                options[target] = number
        if seed is not None:
            options["seed"] = _positive_int(seed, "seed", allow_zero=True)
        if max_new_tokens is not None:
            options["num_predict"] = _positive_int(max_new_tokens, "max_new_tokens")
        return options

    def verify(self, timeout: float = 30) -> dict[str, Any]:
        """Require the configured Ollama tag to be installed on this server."""
        self._verified = False
        response = self._request_json("GET", "api/tags", timeout=timeout)
        models = response.get("models")
        if not isinstance(models, list):
            raise BackendError("Ollama did not return an installed-model list")
        names = [item.get("name") for item in models if isinstance(item, dict)]
        if self.model_name not in names:
            raise BackendError("Ollama model identity does not match the configured tag")
        self._verified = True
        return {"model": self.model_name, "verified": True}

    def _require_verified(self) -> None:
        if not self._verified:
            raise BackendError("Call verify() successfully before using the model service")

    def _chat(self, messages: list, *, seed: int | None, max_new_tokens: int, timeout: Any) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": self._messages(messages),
            "stream": False,
            "think": True,
            "options": self._options(seed=seed, max_new_tokens=max_new_tokens),
        }
        return self._request_json("POST", "api/chat", timeout=timeout, payload=payload)

    @staticmethod
    def _count(response: Mapping[str, Any], key: str) -> int:
        value = response.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BackendError("Ollama returned an unverifiable token count")
        return value

    def count_tokens(self, messages: list, timeout: float = 30) -> int:
        """Count the rendered prompt with the same think setting as generation."""
        self._require_verified()
        response = self._chat(messages, seed=None, max_new_tokens=1, timeout=timeout)
        if response.get("model") != self.model_name:
            raise BackendError("Ollama token-count identity does not match the configured tag")
        return self._count(response, "prompt_eval_count")

    def _stream_chat(self, messages: list, *, seed: int | None, max_new_tokens: int, timeout: Any) -> dict[str, Any]:
        """Stream one chat completion and stop a repeated-thinking loop."""
        payload = {
            "model": self.model_name,
            "messages": self._messages(messages),
            "stream": True,
            "think": True,
            "options": self._options(seed=seed, max_new_tokens=max_new_tokens),
        }
        timeout_value = _validate_timeout(timeout)
        url = f"{self.origin}/api/chat"
        headers = {"Accept": "application/x-ndjson", "Content-Type": "application/json"}
        try:
            response = self._get_session().post(url, headers=headers, json=payload, timeout=timeout_value, stream=True, allow_redirects=False)
        except BackendError:
            raise
        except Exception:
            raise BackendError("Model service request failed") from None
        status = getattr(response, "status_code", None)
        if not isinstance(status, int) or not 200 <= status < 300:
            label = f"HTTP {status}" if isinstance(status, int) else "an invalid HTTP status"
            raise BackendError(f"Model service returned {label}")
        if not hasattr(response, "iter_lines"):
            try:
                body = response.json()
            except Exception:
                raise BackendError("Model service returned invalid JSON") from None
            if not isinstance(body, dict):
                raise BackendError("Model service returned an unsupported response")
            return body
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        model = None
        done_reason = None
        prompt_eval = None
        eval_count = None
        try:
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except Exception:
                    raise BackendError("Model service returned invalid JSON") from None
                if not isinstance(chunk, dict):
                    raise BackendError("Model service returned an unsupported response")
                if isinstance(chunk.get("model"), str):
                    model = chunk["model"]
                message = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
                piece = message.get("content")
                if isinstance(piece, str):
                    content_parts.append(piece)
                thought = message.get("thinking")
                if isinstance(thought, str):
                    thinking_parts.append(thought)
                if chunk.get("done") is True:
                    reason = chunk.get("done_reason")
                    done_reason = reason if isinstance(reason, str) and reason else "stop"
                    prompt_eval = chunk.get("prompt_eval_count")
                    eval_count = chunk.get("eval_count")
                    break
                if thinking_is_looping("".join(thinking_parts)):
                    done_reason = "repetition"
                    break
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        if done_reason is None:
            raise BackendError("Ollama returned no valid finish reason")
        content = "".join(content_parts)
        thinking = "".join(thinking_parts)
        if not isinstance(prompt_eval, int) or isinstance(prompt_eval, bool) or prompt_eval < 0:
            prompt_eval = 0
        if not isinstance(eval_count, int) or isinstance(eval_count, bool) or eval_count < 0:
            eval_count = max(1, len((thinking + content).split()))
        return {
            "model": model or self.model_name,
            "message": {"role": "assistant", "content": content, "thinking": thinking},
            "done": True,
            "done_reason": done_reason,
            "prompt_eval_count": prompt_eval,
            "eval_count": eval_count,
        }

    def generate(self, messages: list, seed: int, max_new_tokens: int, timeout: float) -> dict[str, Any]:
        """Generate one response and keep Ollama thinking separate from final text."""
        self._require_verified()
        started = time.perf_counter()
        response = self._stream_chat(messages, seed=seed, max_new_tokens=max_new_tokens, timeout=timeout)
        seconds = time.perf_counter() - started
        if response.get("model") != self.model_name:
            raise BackendError("Generation response identity does not match the configured tag")
        message = response.get("message")
        if not isinstance(message, dict):
            raise BackendError("Ollama returned an unsupported completion message")
        tool_calls = message.get("tool_calls")
        if tool_calls not in (None, []):
            raise BackendError("Tool calls are unsupported by the generator adapter")
        finish_reason = response.get("done_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            raise BackendError("Ollama returned no valid finish reason")
        content = message.get("content")
        if content is None and finish_reason in ("length", "repetition"):
            content = ""
        if not isinstance(content, str) or (not content.strip() and finish_reason not in ("length", "repetition")):
            raise BackendError("Ollama returned no final content")
        if _THINK_RE.search(content):
            raise BackendError("Final content contains unparsed native thinking markers")
        reasoning = message.get("thinking")
        if reasoning is None:
            reasoning = message.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise BackendError("Ollama returned invalid reasoning content")
        return {
            "content": content,
            "reasoning": reasoning,
            "input_tokens": self._count(response, "prompt_eval_count"),
            "output_tokens": self._count(response, "eval_count"),
            "reasoning_tokens": None,
            "finish_reason": finish_reason,
            "raw_response": response,
            "seconds": seconds,
        }


def thinking_is_looping(text: str, *, repeats: int = 6, minimum_chars: int = 24) -> bool:
    """True when recent thinking is the same sentence copied over and over."""
    if repeats < 2 or minimum_chars < 1 or not isinstance(text, str):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= repeats:
        tail = lines[-repeats:]
        if len(tail[0]) >= minimum_chars and len(set(tail)) == 1:
            return True
    compact = "".join(lines)
    unit_at = minimum_chars
    if len(compact) >= unit_at * repeats:
        unit = compact[-unit_at:]
        if compact[-(unit_at * repeats):] == unit * repeats:
            return True
    return False


def _linux_snapshot_path(snapshot_path: str) -> str:
    if not isinstance(snapshot_path, str) or not snapshot_path or "\\" in snapshot_path:
        raise BackendError("snapshot_path must be an absolute Linux path")
    lower = snapshot_path.lower()
    if re.match(r"^[a-z]:", lower) or "wsl$" in lower or "ollama" in lower:
        raise BackendError("snapshot_path must identify a Linux model snapshot")
    if re.match(r"^/mnt/[a-z](?:/|$)", lower):
        raise BackendError("snapshot_path must not use a Windows-mounted WSL path")
    path = PurePosixPath(snapshot_path)
    if not path.is_absolute() or ".." in path.parts:
        raise BackendError("snapshot_path must be an absolute Linux path without traversal")
    return snapshot_path


def build_vllm_command(model: dict, gpu: dict, snapshot_path: str) -> list[str]:
    """Build an explicit vLLM launch command for a pinned local model snapshot.

    The caller sets ``CUDA_VISIBLE_DEVICES`` before launching this command. The
    model path must already resolve to files downloaded at the pinned revision.
    """
    if not isinstance(model, Mapping) or not isinstance(gpu, Mapping):
        raise BackendError("Model and GPU config must be objects")
    _, _, alias = _model_identity(model)
    path = _linux_snapshot_path(snapshot_path)

    tp = _positive_int(gpu.get("tensor_parallel_size", gpu.get("tp", 1)), "tensor_parallel_size")
    memory = _finite_number(gpu.get("gpu_memory_utilization", 0.9), "gpu_memory_utilization", minimum=0.01, maximum=1.0)
    max_model_len = _positive_int(gpu.get("max_model_len", 32768), "max_model_len")
    max_num_seqs = _positive_int(gpu.get("max_num_seqs", 1), "max_num_seqs")

    return [
        "vllm",
        "serve",
        path,
        "--served-model-name",
        alias,
        "--reasoning-parser",
        "qwen3",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(tp),
        "--gpu-memory-utilization",
        str(memory),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
    ]


class NativeD3Selector:
    """Pinned native D3 `system_one` selector on one CUDA device.

    PyTorch, Transformers, and Hugging Face Hub are imported only by ``load``.
    The remote Python modeling code is explicitly revision-pinned and is only
    loaded when a target device calls this method.
    """

    def __init__(self, model: dict, gpu: dict, cache_dir: str | None = None):
        if not isinstance(model, Mapping) or not isinstance(gpu, Mapping):
            raise BackendError("Model and GPU config must be objects")
        self.model_config = copy.deepcopy(dict(model))
        self.gpu_config = copy.deepcopy(dict(gpu))
        self.repo, self.revision, self.model_id = _model_identity(self.model_config)
        self.cache_dir = cache_dir
        if cache_dir is not None and not isinstance(cache_dir, str):
            raise BackendError("cache_dir must be a path string or None")
        self.device = self.gpu_config.get("device", "cuda:0")
        if not isinstance(self.device, str) or not re.fullmatch(r"cuda:\d+", self.device):
            raise BackendError("D3 requires one explicit CUDA device such as cuda:0")
        batch_size = self.gpu_config.get("batch_size", 1)
        if _positive_int(batch_size, "batch_size") != 1:
            raise BackendError("The pinned D3 runtime is configured for batch_size=1")
        if self.gpu_config.get("dtype", "bfloat16") != "bfloat16":
            raise BackendError("The pinned D3 runtime requires bfloat16 backbone weights")

        visible = self.gpu_config.get("visible_devices")
        if visible is not None and not isinstance(visible, str):
            raise BackendError("visible_devices must be a string when configured")
        self.visible_devices = visible
        if "question_context_tokens" not in self.gpu_config:
            raise BackendError("GPU config must set question_context_tokens explicitly")
        self.requested_context_tokens = _positive_int(
            self.gpu_config["question_context_tokens"], "question_context_tokens"
        )
        self._torch: Any = None
        self._model: Any = None
        self._metadata: dict[str, Any] | None = None
        self._lock = threading.RLock()

    @staticmethod
    def _cuda_index(device: str) -> int:
        return int(device.split(":", 1)[1])

    @staticmethod
    def _object_value(obj: Any, name: str) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(name)
        return getattr(obj, name, None)

    @classmethod
    def _positive_optional_cap(cls, value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BackendError(f"Native D3 declared an invalid {label}")
        return value

    @classmethod
    def _native_context_limits(cls, loaded: Any, runtime: Any) -> tuple[int, int | None, int | None]:
        runtime_cap = cls._positive_optional_cap(
            getattr(runtime, "max_length", None), "runtime max_length"
        )
        config = getattr(loaded, "config", None)
        text_config = cls._object_value(config, "text_config")
        model_cap_raw = cls._object_value(text_config, "max_position_embeddings")
        if model_cap_raw is None:
            model_cap_raw = cls._object_value(config, "max_position_embeddings")
        model_cap = cls._positive_optional_cap(model_cap_raw, "max_position_embeddings")
        caps = [cap for cap in (runtime_cap, model_cap) if cap is not None]
        if not caps:
            raise BackendError("Could not determine the pinned D3 context limit")
        return min(caps), runtime_cap, model_cap

    def load(self) -> dict[str, Any]:
        """Verify CUDA/BF16, download the exact snapshot, then load native D3 code."""
        with self._lock:
            if self._model is not None:
                return copy.deepcopy(self._metadata or {})
            if self.visible_devices is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != self.visible_devices:
                raise BackendError("CUDA_VISIBLE_DEVICES does not match the configured GPU mapping")
            try:
                torch = importlib.import_module("torch")
            except Exception:
                raise BackendError("PyTorch is unavailable in the target GPU environment") from None

            try:
                if not torch.cuda.is_available():
                    raise BackendError("CUDA is unavailable; native D3 has no CPU fallback")
                index = self._cuda_index(self.device)
                if index >= torch.cuda.device_count():
                    raise BackendError("Configured CUDA device is not visible")
                if not torch.cuda.is_bf16_supported():
                    raise BackendError("Configured CUDA device does not support bfloat16")
                device_name = torch.cuda.get_device_name(index)
            except BackendError:
                raise
            except Exception:
                raise BackendError("Could not verify the configured CUDA device") from None

            try:
                hub = importlib.import_module("huggingface_hub")
                transformers = importlib.import_module("transformers")
            except Exception:
                raise BackendError("Hugging Face model dependencies are unavailable") from None

            download_kwargs: dict[str, Any] = {"repo_id": self.repo, "revision": self.revision}
            if self.cache_dir is not None:
                download_kwargs["cache_dir"] = self.cache_dir
            try:
                snapshot = hub.snapshot_download(**download_kwargs)
                snapshot_path = os.path.realpath(os.fspath(snapshot))
                if os.path.basename(snapshot_path) != self.revision:
                    raise BackendError("Downloaded snapshot directory does not match the pinned revision")
                loaded = transformers.AutoModel.from_pretrained(
                    snapshot_path,
                    trust_remote_code=True,
                    revision=self.revision,
                    code_revision=self.revision,
                    device=self.device,
                    batch_size=1,
                    verify="full",
                    dtype=torch.bfloat16,
                )
            except BackendError:
                raise
            except Exception:
                raise BackendError("Pinned native D3 checkpoint could not be loaded") from None

            runtime = getattr(loaded, "runtime", None)
            runtime_device = getattr(runtime, "device", None)
            if not callable(getattr(loaded, "system_one", None)) or runtime is None:
                raise BackendError("Loaded checkpoint does not expose the native D3 system_one interface")
            if runtime_device is None or str(runtime_device) != self.device:
                raise BackendError("Loaded D3 runtime device does not match the configured CUDA device")

            native_context_cap, runtime_max_length, model_max_positions = self._native_context_limits(
                loaded, runtime
            )
            if self.requested_context_tokens > native_context_cap:
                raise BackendError(
                    "Requested common question context exceeds the pinned D3 context limit",
                    details={
                        "requested_question_context_tokens": self.requested_context_tokens,
                        "native_context_cap_tokens": native_context_cap,
                        "runtime_max_length": runtime_max_length,
                        "model_max_position_embeddings": model_max_positions,
                    },
                )

            manifest = getattr(loaded, "manifest", None)
            manifest_repo = manifest.get("repo_id") if isinstance(manifest, Mapping) else None
            if not isinstance(manifest, Mapping) or manifest_repo != self.repo:
                raise BackendError("Native D3 loader did not verify the expected repository manifest")
            model_sha256 = None
            if isinstance(manifest, Mapping):
                identity = manifest.get("identity")
                if isinstance(identity, Mapping):
                    model_sha256 = identity.get("model_sha256")
            native_reported_model = getattr(loaded, "model_name", None)
            if not isinstance(native_reported_model, str) or not native_reported_model:
                raise BackendError("Native D3 loader returned no verifiable model identity")
            self._torch = torch
            self._model = loaded
            self._metadata = {
                "model": self.model_id,
                "revision": self.revision,
                "snapshot_path": snapshot_path,
                "snapshot_directory_sha": self.revision,
                "code_revision": self.revision,
                "manifest_verification": "full requested from native loader",
                "manifest_repo_id": manifest_repo,
                "model_sha256": model_sha256,
                "native_reported_model": native_reported_model,
                "device": self.device,
                "device_name": device_name,
                "visible_devices_config": self.visible_devices,
                "requested_question_context_tokens": self.requested_context_tokens,
                "native_context_cap_tokens": native_context_cap,
                "runtime_max_length": runtime_max_length,
                "model_max_position_embeddings": model_max_positions,
                "native_context_cap_source": "minimum of loaded runtime max_length and model config max_position_embeddings when declared",
                "dtype": "bfloat16 backbone; native FP32 readout",
                "batch_size": 1,
                "system_one": True,
            }
            return copy.deepcopy(self._metadata)

    @staticmethod
    def _validate_choice_answer(answer: Any, criteria: Mapping[str, Any]) -> tuple[str, dict[str, float], float]:
        if not isinstance(answer, Mapping) or "error" in answer:
            raise BackendError("D3 returned an unsupported or failed choice answer")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in criteria:
            raise BackendError("D3 returned a choice outside the configured options")
        raw_probabilities = answer.get("probabilities")
        if not isinstance(raw_probabilities, Mapping) or set(raw_probabilities) != set(criteria):
            raise BackendError("D3 did not return one probability for every configured option")
        probabilities: dict[str, float] = {}
        for key in criteria:
            value = raw_probabilities[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value > 1:
                raise BackendError("D3 returned invalid choice probabilities")
            probabilities[key] = float(value)
        total = sum(probabilities.values())
        if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise BackendError("D3 returned an unnormalized choice distribution")
        confidence = answer.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise BackendError("D3 returned invalid choice confidence")
        return choice, probabilities, float(confidence)

    @staticmethod
    def _validate_usage(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise BackendError("D3 returned invalid usage")
        safe = _json_copy(dict(raw), "D3 usage")
        for name in ("input_tokens", "output_tokens"):
            value = safe.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BackendError("D3 returned invalid usage")
        return safe

    @staticmethod
    def _choice_input(
        state: Any, criteria: Mapping[str, Any], instructions: str
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= 255:
            raise BackendError("criteria must map between 2 and 255 option labels")
        if any(not isinstance(key, str) or not key for key in criteria):
            raise BackendError("criteria labels must be non-empty strings")
        safe_criteria = _json_copy(dict(criteria), "criteria")
        if not isinstance(instructions, str) or not instructions.strip():
            raise BackendError("instructions must be non-empty text")
        safe_state = _json_copy(state, "state")
        question = {
            "choice": {
                "type": "choice",
                "instructions": instructions,
                "criteria": safe_criteria,
            }
        }
        return safe_state, safe_criteria, question

    def _count_prepared_choice(self, state: Any, question: dict[str, Any]) -> int:
        """Count D3's fully rendered Choice prompt without invoking its model forward pass."""
        runtime = getattr(self._model, "runtime", None)
        prepare = getattr(runtime, "prepare", None)
        if runtime is None or not callable(prepare):
            raise BackendError("Pinned D3 runtime has no supported tokenization preflight")
        try:
            prepared = prepare(state, question)
        except Exception:
            raise BackendError("D3 could not tokenize the question context") from None

        sequences = getattr(prepared, "sequences", None)
        if isinstance(sequences, Mapping) and "choice" in sequences:
            return len(sequences["choice"])

        errors = getattr(prepared, "errors", None)
        error = errors.get("choice") if isinstance(errors, Mapping) else None
        if isinstance(error, Mapping) and error.get("error") == "max_length_exceeded":
            # `prepare()` validates against its own limit and omits overlong IDs.
            # Re-render with the same native prompt function and count those IDs
            # so the caller can reject against the study cap without inference.
            questions = getattr(prepared, "questions", None)
            normalized = questions.get("choice") if isinstance(questions, Mapping) else None
            rendered = getattr(normalized, "rendered", None)
            tokenizer = getattr(runtime, "tokenizer", None)
            render = getattr(runtime, "text", None)
            if rendered is None or not callable(tokenizer) or not callable(render):
                raise BackendError("D3 could not expose an over-limit preflight token count")
            try:
                prompt = render(state, rendered)
                encoded = tokenizer([prompt], add_special_tokens=False)
                ids = encoded["input_ids"][0]
                return len(ids)
            except Exception:
                raise BackendError("D3 could not count its over-limit prompt") from None
        raise BackendError("D3 rejected the question before tokenization")

    def count_tokens(self, state: Any, criteria: dict, instructions: str) -> int:
        """Return the exact native token count for the rendered Choice prompt, without inference."""
        with self._lock:
            if self._model is None:
                raise BackendError("Call load() successfully before counting tokens")
            safe_state, _, question = self._choice_input(state, criteria, instructions)
            return self._count_prepared_choice(safe_state, question)

    def _metrics_before(self, torch: Any) -> None:
        try:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass

    def _metrics_after(self, torch: Any) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "device": self.device,
            "visible_devices_config": self.visible_devices,
        }
        try:
            torch.cuda.synchronize(self.device)
            metrics["memory_allocated_bytes"] = int(torch.cuda.memory_allocated(self.device))
            metrics["peak_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
        except Exception:
            metrics["memory_allocated_bytes"] = None
            metrics["peak_memory_allocated_bytes"] = None
        try:
            metrics["device_name"] = torch.cuda.get_device_name(self._cuda_index(self.device))
        except Exception:
            metrics["device_name"] = None
        return metrics

    def choose(self, state: Any, criteria: dict, instructions: str) -> dict[str, Any]:
        """Use native D3 Choice; overlong contexts are surfaced as errors, never trimmed."""
        with self._lock:
            if self._model is None or self._torch is None:
                raise BackendError("Call load() successfully before choosing")
            safe_state, safe_criteria, question = self._choice_input(
                state, criteria, instructions
            )
            preflight_input_tokens = self._count_prepared_choice(safe_state, question)
            if preflight_input_tokens > self.requested_context_tokens:
                raise BackendError(
                    "Question context exceeds the configured common context cap",
                    details={
                        "error": "question_context_limit",
                        "input_tokens": preflight_input_tokens,
                        "requested_question_context_tokens": self.requested_context_tokens,
                    },
                )

            self._metrics_before(self._torch)
            started = time.perf_counter()
            try:
                raw_response = self._model.system_one(state=safe_state, questions=question)
            except Exception:
                raise BackendError("Native D3 system_one request failed or refused its context") from None
            seconds = time.perf_counter() - started
            if not isinstance(raw_response, Mapping):
                raise BackendError("D3 returned an unsupported system_one response")
            if raw_response.get("model") != (self._metadata or {}).get("native_reported_model"):
                raise BackendError("D3 response identity does not match the pinned checkpoint")
            answers = raw_response.get("answers")
            if not isinstance(answers, Mapping) or "choice" not in answers:
                raise BackendError("D3 returned no choice answer")
            choice, probabilities, confidence = self._validate_choice_answer(answers["choice"], safe_criteria)
            usage = self._validate_usage(raw_response.get("usage"))
            if usage["input_tokens"] > self.requested_context_tokens:
                raise BackendError(
                    "D3 usage exceeded the configured common context cap",
                    details={
                        "error": "question_context_limit",
                        "input_tokens": usage["input_tokens"],
                        "preflight_input_tokens": preflight_input_tokens,
                        "requested_question_context_tokens": self.requested_context_tokens,
                        "usage": usage,
                        "raw_response": _json_copy(dict(raw_response), "D3 response"),
                    },
                )
            gpu_metrics = self._metrics_after(self._torch)
            return {
                "choice": choice,
                "probabilities": probabilities,
                "confidence": confidence,
                "usage": usage,
                "input_tokens": usage["input_tokens"],
                "preflight_input_tokens": preflight_input_tokens,
                "raw_response": _json_copy(dict(raw_response), "D3 response"),
                "seconds": seconds,
                "gpu_metrics": gpu_metrics,
            }

    def close(self) -> None:
        """Release references held by this adapter; PyTorch may retain its CUDA cache."""
        with self._lock:
            self._model = None
            self._torch = None
            self._metadata = None
