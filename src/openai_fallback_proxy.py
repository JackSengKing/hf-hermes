#!/usr/bin/env python3
"""OpenAI-compatible fallback proxy for Hermes on Hugging Face Spaces."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

import requests


LISTEN_HOST = os.environ.get("FALLBACK_PROXY_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("FALLBACK_PROXY_PORT", "8787"))
REQUEST_TIMEOUT = int(os.environ.get("FALLBACK_PROXY_TIMEOUT", "180"))

PRIMARY_BASE_URL = os.environ.get("PRIMARY_BASE_URL", "").rstrip("/")
PRIMARY_API_KEY = os.environ.get("PRIMARY_API_KEY", "")
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "")

FALLBACK_BASE_URL = os.environ.get("FALLBACK_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
FALLBACK_API_KEY = os.environ.get("FALLBACK_API_KEY", "")
FALLBACK_MODEL = os.environ.get("FALLBACK_MODEL", "openrouter/free")
FALLBACK_REFERER = os.environ.get("OPENROUTER_HTTP_REFERER", "https://huggingface.co")
FALLBACK_TITLE = os.environ.get("OPENROUTER_X_TITLE", "Hermes HF Fallback")
VERBOSE_LOGGING = os.environ.get("FALLBACK_PROXY_VERBOSE", "true").lower() in {"1", "true", "yes", "on"}
MAX_LOG_CHARS = int(os.environ.get("FALLBACK_PROXY_MAX_LOG_CHARS", "6000"))


def is_retryable(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def should_fallback(status_code: int) -> bool:
    return status_code == 400 or is_retryable(status_code)


def build_headers(api_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra:
        headers.update(extra)
    return headers


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    normalized_messages = []
    for message in messages:
        if not isinstance(message, dict):
            normalized_messages.append(message)
            continue

        role = (message.get("role") or "user").lower()
        content = normalize_message_content(message.get("content"))

        if role == "developer":
            role = "system"
        elif role in {"tool", "function"}:
            role = "user"
            prefix = "Tool result"
            tool_name = message.get("name") or message.get("tool_call_id")
            if tool_name:
                prefix = f"Tool result ({tool_name})"
            content = f"{prefix}:\n{content}" if content else prefix
        elif role not in {"system", "user", "assistant"}:
            role = "user"

        normalized: Dict[str, Any] = {
            "role": role,
            "content": content,
        }

        if role == "assistant" and message.get("tool_calls"):
            normalized["tool_calls"] = message.get("tool_calls")
            if not content:
                normalized["content"] = json.dumps(message.get("tool_calls"), ensure_ascii=False)

        normalized_messages.append(normalized)

    return normalized_messages


def clip_text(value: str, limit: int = MAX_LOG_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"... [truncated {len(value) - limit} chars]"


def dump_json(data: Any) -> str:
    try:
        return clip_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        return f"<json-dump-error: {exc}>"


def summarize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    summary = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            summary.append({"index": idx, "type": type(message).__name__, "value": str(message)[:200]})
            continue
        content = message.get("content")
        if isinstance(content, str):
            content_preview = clip_text(content, 400)
            content_type = "str"
        elif isinstance(content, list):
            content_preview = clip_text(json.dumps(content, ensure_ascii=False), 400)
            content_type = "list"
        elif isinstance(content, dict):
            content_preview = clip_text(json.dumps(content, ensure_ascii=False), 400)
            content_type = "dict"
        else:
            content_preview = clip_text(str(content), 400)
            content_type = type(content).__name__
        summary.append(
            {
                "index": idx,
                "role": message.get("role"),
                "content_type": content_type,
                "content_preview": content_preview,
                "has_tool_calls": bool(message.get("tool_calls")),
                "tool_call_id": message.get("tool_call_id"),
                "name": message.get("name"),
                "keys": sorted(message.keys()),
            }
        )
    return summary


def log_debug(title: str, data: Any) -> None:
    if not VERBOSE_LOGGING:
        return
    print(f"[fallback-proxy] {title}:\n{dump_json(data)}")


def create_upstream_response(
    upstream_base: str,
    payload: Dict[str, Any],
    api_key: str,
    model_override: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> requests.Response:
    request_payload = dict(payload)
    request_payload["messages"] = normalize_messages(request_payload.get("messages"))
    request_payload["model"] = model_override
    log_debug(
        "outbound_request",
        {
            "upstream_base": upstream_base,
            "model_override": model_override,
            "stream": bool(request_payload.get("stream")),
            "keys": sorted(request_payload.keys()),
            "message_summary": summarize_messages(request_payload.get("messages")),
            "payload": request_payload,
        },
    )
    return requests.post(
        f"{upstream_base}/chat/completions",
        headers=build_headers(api_key, extra_headers),
        json=request_payload,
        timeout=REQUEST_TIMEOUT,
        stream=bool(request_payload.get("stream")),
    )


class Handler(BaseHTTPRequestHandler):
    server_version = "HermesFallbackProxy/0.1"
    protocol_version = "HTTP/1.1"

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "Invalid JSON body"}})
            return None

    def _relay_response(self, response: requests.Response, stream: bool) -> None:
        content_type = response.headers.get("Content-Type", "application/json")
        self.send_response(response.status_code)
        self.send_header("Content-Type", content_type)
        if stream:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
        else:
            self.send_header("Content-Length", str(len(response.content)))
        self.end_headers()

        if stream:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            response.close()
        else:
            self.wfile.write(response.content)

    def _send_plain(self, status_code: int, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "status": "ok",
                    "primary_configured": bool(PRIMARY_BASE_URL and PRIMARY_MODEL),
                    "fallback_configured": bool(FALLBACK_API_KEY and FALLBACK_MODEL),
                },
            )
            return

        if self.path in {"/version", "/v1/props", "/props"}:
            self._send_json(
                200,
                {
                    "version": "fallback-proxy",
                    "primary_model": PRIMARY_MODEL,
                    "fallback_model": FALLBACK_MODEL,
                },
            )
            return

        if self.path in {"/api/tags"}:
            self._send_json(200, {"models": [{"name": PRIMARY_MODEL or FALLBACK_MODEL}]})
            return

        if self.path in {"/v1/models", "/api/v1/models"}:
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": PRIMARY_MODEL or FALLBACK_MODEL or "openai-fallback-proxy",
                            "object": "model",
                            "owned_by": "hermes-local-proxy",
                        }
                    ],
                },
            )
            return

        if self.path.startswith("/v1/models/"):
            model_id = self.path.split("/v1/models/", 1)[1]
            self._send_json(
                200,
                {
                    "id": model_id or PRIMARY_MODEL or FALLBACK_MODEL,
                    "object": "model",
                    "owned_by": "hermes-local-proxy",
                },
            )
            return

        self._send_json(404, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": {"message": "Not found"}})
            return

        payload = self._read_json()
        if payload is None:
            return

        log_debug(
            "incoming_request",
            {
                "path": self.path,
                "keys": sorted(payload.keys()),
                "stream": bool(payload.get("stream")),
                "message_summary": summarize_messages(payload.get("messages")),
                "payload": payload,
            },
        )

        if not PRIMARY_BASE_URL or not PRIMARY_MODEL:
            self._send_json(500, {"error": {"message": "Primary model not configured"}})
            return

        stream = bool(payload.get("stream"))
        primary_response = None
        try:
            primary_response = create_upstream_response(
                PRIMARY_BASE_URL,
                payload,
                PRIMARY_API_KEY,
                PRIMARY_MODEL,
            )
            log_debug(
                "primary_response",
                {
                    "status_code": primary_response.status_code,
                    "headers": dict(primary_response.headers),
                    "body_preview": clip_text(primary_response.text if not stream else "<stream-response>"),
                },
            )
            if primary_response.status_code < 400:
                self._relay_response(primary_response, stream)
                return
            if primary_response.status_code == 400:
                try:
                    body_preview = primary_response.text[:500]
                except Exception:
                    body_preview = "<unavailable>"
                try:
                    message_roles = [m.get("role") for m in (payload.get("messages") or []) if isinstance(m, dict)]
                except Exception:
                    message_roles = []
                print(
                    "[fallback-proxy] primary 400 -> fallback; "
                    f"roles={message_roles} keys={sorted(payload.keys())} body={body_preview}"
                )
            if not FALLBACK_API_KEY or not should_fallback(primary_response.status_code):
                self._relay_response(primary_response, False)
                return
        except requests.RequestException as error:
            if not FALLBACK_API_KEY:
                self._send_json(502, {"error": {"message": f"Primary upstream request failed: {error}"}})
                return
        finally:
            if primary_response is not None and not stream:
                primary_response.close()

        try:
            fallback_response = create_upstream_response(
                FALLBACK_BASE_URL,
                payload,
                FALLBACK_API_KEY,
                FALLBACK_MODEL,
                {
                    "HTTP-Referer": FALLBACK_REFERER,
                    "X-Title": FALLBACK_TITLE,
                },
            )
            log_debug(
                "fallback_response",
                {
                    "status_code": fallback_response.status_code,
                    "headers": dict(fallback_response.headers),
                    "body_preview": clip_text(fallback_response.text if not stream else "<stream-response>"),
                },
            )
            self._relay_response(fallback_response, stream)
        except requests.RequestException as error:
            self._send_json(502, {"error": {"message": f"Fallback upstream request failed: {error}"}})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[fallback-proxy] {self.address_string()} - {fmt % args}")


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"[fallback-proxy] listening on http://{LISTEN_HOST}:{LISTEN_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
