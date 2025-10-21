import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import litellm


_ORIGINAL_COMPLETION: Optional[Callable[..., Any]] = None
_FIRST_CALL_PRINTED = False


@dataclass
class RecorderConfig:
    outdir: Path
    include_tools: bool = False
    strip_think: bool = True
    debug: bool = False


class _SessionContext(threading.local):
    def __init__(self) -> None:
        super().__init__()
        self.session_id: Optional[str] = None
        self.domain: Optional[str] = None
        self.task_id: Optional[str] = None


_session_ctx = _SessionContext()
_turn_index_by_session: dict[str, int] = {}
_turn_lock = threading.Lock()
_first_system_by_session: dict[str, Optional[str]] = {}


def set_session_context(session_id: str, domain: Optional[str], task_id: Optional[str]) -> None:
    _session_ctx.session_id = session_id
    _session_ctx.domain = domain
    _session_ctx.task_id = task_id
    with _turn_lock:
        _turn_index_by_session.setdefault(session_id, 0)
        _first_system_by_session.setdefault(session_id, None)


def clear_session_context() -> None:
    sess = _session_ctx.session_id
    _session_ctx.session_id = None
    _session_ctx.domain = None
    _session_ctx.task_id = None
    # Intentionally keep _turn_index_by_session for aggregation


def get_first_system_for_session(session_id: str) -> Optional[str]:
    return _first_system_by_session.get(session_id)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _serialize_response(resp: Any) -> Any:
    # Best-effort serialization of LiteLLM response
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
    except Exception:
        pass
    try:
        if hasattr(resp, "to_dict"):
            return resp.to_dict()
    except Exception:
        pass
    try:
        return json.loads(json.dumps(resp, default=str))
    except Exception:
        return str(resp)


def _extract_assistant_text(resp: Any) -> Optional[str]:
    try:
        # Typical LiteLLM response: has .choices[0].message.content
        choice0 = resp.choices[0]
        message = getattr(choice0, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            return content
        # Fallbacks
        if hasattr(choice0, "text"):
            return choice0.text
    except Exception:
        return None
    return None


def _extract_system_content(messages: list[dict]) -> Optional[str]:
    if not messages:
        return None
    first = messages[0]
    if first.get("role") == "system":
        content = first.get("content")
        return content if isinstance(content, str) else None
    return None


def _redact_request(req: dict) -> dict:
    # Redact common secret fields
    redacted = dict(req)
    for k in ["api_key", "api_secret", "access_token", "bearer_token", "authorization"]:
        if k in redacted:
            redacted[k] = "[REDACTED]"
    return redacted


def _normalize_tools_for_compat(tools: Any) -> Any:
    """
    Ensure each tool dict has top-level 'name' and 'type' when possible by
    mirroring from nested function schema for OpenAI-compatible validators.
    Keeps original structure otherwise.
    
    NOTE: This normalization adds a top-level 'name' field that some providers
    (e.g., Fireworks) reject due to strict validation. Skip this function for
    such providers. Most providers (GPT, Claude, Gemini) are lenient and accept
    both normalized and non-normalized formats.
    """
    if not isinstance(tools, list):
        return tools
    normalized: list[Any] = []
    for t in tools:
        if not isinstance(t, dict):
            normalized.append(t)
            continue
        tt = dict(t)
        fn = tt.get("function")
        if isinstance(fn, dict):
            fn_name = fn.get("name")
            if fn_name and "name" not in tt:
                tt["name"] = fn_name
            if "type" not in tt:
                tt["type"] = "function"
        normalized.append(tt)
    return normalized


def install_litellm_recorder(config: Optional[RecorderConfig] = None) -> None:
    """
    Monkeypatch litellm.completion to log request/response jsonl lines.
    Safe to call multiple times; only first call installs the wrapper.
    """
    global _ORIGINAL_COMPLETION
    if _ORIGINAL_COMPLETION is not None:
        return

    # Resolve config from ENV defaults
    if config is None:
        outdir = Path(os.environ.get("RECORDER_OUTDIR", "./recordings"))
        include_tools = os.environ.get("RECORDER_INCLUDE_TOOLS", "0") == "1"
        strip_think = os.environ.get("RECORDER_STRIP_THINK", "1") == "1"
        debug = os.environ.get("RECORDER_DEBUG", "0") == "1"
        config = RecorderConfig(outdir=outdir, include_tools=include_tools, strip_think=strip_think, debug=debug)

    _ensure_dir(config.outdir)
    payloads_path = config.outdir / "tau2_payloads.jsonl"

    _ORIGINAL_COMPLETION = litellm.completion

    def _wrapper(*args: Any, **kwargs: Any):
        nonlocal payloads_path, config

        # Parse model + request fields
        model = kwargs.get("model") if "model" in kwargs else (args[0] if len(args) > 0 else None)
        messages = kwargs.get("messages") if "messages" in kwargs else (args[1] if len(args) > 1 else None)
        tools = kwargs.get("tools")
        temperature = kwargs.get("temperature")
        top_p = kwargs.get("top_p")
        tool_choice = kwargs.get("tool_choice")
        budget_or_max_tokens = kwargs.get("budget_or_max_tokens")

        if not isinstance(messages, list):
            messages = []

        # First-call sanity print
        global _FIRST_CALL_PRINTED
        if not _FIRST_CALL_PRINTED:
            _FIRST_CALL_PRINTED = True
            has_system = len(messages) > 0 and messages[0].get("role") == "system"
            print(f"[Recorder] Logging to: {payloads_path}")
            print(f"[Recorder] First call sanity: has_system={has_system}, model={model}, messages={len(messages)}")

        # Capture session context and turn index
        session_id = _session_ctx.session_id or "unknown"
        domain = _session_ctx.domain
        task_id = _session_ctx.task_id
        with _turn_lock:
            turn_index = _turn_index_by_session.get(session_id, 0)
            _turn_index_by_session[session_id] = turn_index + 1

        # Persist first system prompt for this session if present
        if _first_system_by_session.get(session_id) is None:
            sys_content = _extract_system_content(messages)
            if sys_content:
                _first_system_by_session[session_id] = sys_content
            else:
                # Warn once per session
                print(f"[Recorder][warn] session_id={session_id} missing system prompt in first call")

        # Build request record
        start_ts = _now_iso()
        start_time = time.perf_counter()
        req = {
            "model": model,
            "messages": messages,
            "tools": tools if config.include_tools else None,
            "temperature": temperature,
            "top_p": top_p,
            "budget_or_max_tokens": budget_or_max_tokens,
        }
        # Extra params: copy from kwargs excluding known fields
        exclude_keys = {"model", "messages", "tools", "tool_choice", "temperature", "top_p", "budget_or_max_tokens"}
        extra_params = {k: v for k, v in kwargs.items() if k not in exclude_keys}
        if extra_params:
            req["extra_params"] = extra_params

        # Manually run pre-API callbacks because monkeypatching bypasses LiteLLM's internal callback handling
        if litellm.pre_api_callback:
            for callback in litellm.pre_api_callback:
                try:
                    # The callback might modify kwargs in-place
                    callback(kwargs=kwargs, model_name=model)
                except Exception as e:
                    # Log callback errors but don't crash the main call
                    print(f"[Recorder][warn] Pre-API callback {getattr(callback, '__name__', 'unknown')} failed: {e}")

        # Call real client (normalize tools for stricter backends)
        if tools is not None:
            # Fireworks rejects extra 'name' field - skip normalization for them
            if model and "fireworks" in str(model).lower():
                kwargs["tools"] = tools  # Use as-is for Fireworks
            else:
                kwargs["tools"] = _normalize_tools_for_compat(tools)
        resp = _ORIGINAL_COMPLETION(*args, **kwargs)

        # Parse response summary
        assistant_text = _extract_assistant_text(resp)
        if assistant_text in (None, ""):
            print(f"[Recorder][warn] empty assistant_text session_id={session_id} turn={turn_index}")
        end_ts = _now_iso()
        _ = time.perf_counter() - start_time  # elapsed, not used in output but available

        # Write JSONL line
        record = {
            "session_id": session_id,
            "domain": domain,
            "task_id": task_id,
            "turn_index": turn_index,
            "request": _redact_request(req),
            "response": {
                "raw": _serialize_response(resp) if config.debug else None,
                "assistant_text": assistant_text,
            },
            "timestamps": {"start": start_ts, "end": end_ts},
        }
        with open(payloads_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return resp

    litellm.completion = _wrapper  # type: ignore[assignment]
    # Ensure modules that imported 'completion' by name also use the wrapper
    try:  # best-effort, avoid hard failures
        import tau2.utils.llm_utils as _llm_utils  # type: ignore
        if getattr(_llm_utils, "completion", None) is not _wrapper:
            _llm_utils.completion = _wrapper  # type: ignore[assignment]
    except Exception:
        pass


