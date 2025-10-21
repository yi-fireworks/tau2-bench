import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import requests

from tau2.utils.reasoning_effort import reasoning_effort_pre_api_callback


TEST_PROMPT = (
    "Explain the concept of quantum entanglement in simple terms and then "
    "discuss its philosophical implications."
)


def detect_provider_and_endpoint(model: str) -> Tuple[str, str]:
    """
    Infer provider and HTTP endpoint from the model identifier.
    We avoid LiteLLM and talk to providers directly where possible.
    """
    name = (model or "").lower()

    # OpenAI
    if ("gpt-" in name or name.startswith("openai/")) and "fireworks" not in name:
        return (
            "openai",
            "https://api.openai.com/v1/chat/completions",
        )

    # Anthropic
    if name.startswith("anthropic/") or "claude" in name:
        return (
            "anthropic",
            "https://api.anthropic.com/v1/messages",
        )

    # Google Gemini
    if name.startswith("gemini/") or name.startswith("google/") or "gemini" in name:
        # The Gemini endpoint is model-specific in the path; use bare model id
        normalized_model = model.split("/", 1)[1] if "/" in model else model
        return (
            "google",
            f"https://generativelanguage.googleapis.com/v1beta/models/{normalized_model}:generateContent",
        )

    # Fireworks
    if "fireworks" in name or name.startswith("fireworks_ai/"):
        return (
            "fireworks",
            "https://api.fireworks.ai/inference/v1/chat/completions",
        )

    # Default to OpenAI-compatible chat completions if unknown
    return (
        "openai",
        "https://api.openai.com/v1/chat/completions",
    )


def build_headers(provider: str) -> Dict[str, str]:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY", "")
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    if provider == "google":
        # Per docs, support x-goog-api-key header
        key = os.getenv("GOOGLE_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["x-goog-api-key"] = key
        return headers
    if provider == "fireworks":
        key = os.getenv("FIREWORKS_API_KEY", "")
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    return {"Content-Type": "application/json"}


def build_payload(provider: str, model: str, prompt: str, mapped_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create provider-specific request payloads using our mapped kwargs.
    This highlights exactly how `_apply_provider_reasoning_effort` influences requests.
    """
    if provider == "openai":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(mapped_kwargs)
        return payload

    if provider == "anthropic":
        payload = {
            "model": model.split("/", 1)[1] if "/" in model else model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        }
        payload.update(mapped_kwargs)
        return payload

    if provider == "google":
        # Gemini's schema uses contents/parts and supports generationConfig.thinkingConfig.thinkingBudget
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
        }
        # Move known Gemini fields into generationConfig if present in mapped kwargs
        thinking_cfg = None
        if "thinkingConfig" in mapped_kwargs:
            thinking_cfg = mapped_kwargs.pop("thinkingConfig")
        gen_cfg = mapped_kwargs.pop("generationConfig", {})
        if thinking_cfg is not None:
            gen_cfg["thinkingConfig"] = thinking_cfg
        if gen_cfg:
            payload["generationConfig"] = gen_cfg
        # Any other fields pass-through (unlikely for Google)
        payload.update(mapped_kwargs)
        return payload

    if provider == "fireworks":
        # Normalize model id: Fireworks may list models as accounts/<org>/models/<name>;
        # some sources include a "fireworks_ai/" prefix – strip it if present.
        norm_model = model[len("fireworks_ai/"):] if model.startswith("fireworks_ai/") else model
        payload = {
            "model": norm_model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(mapped_kwargs)
        return payload

    # Fallback (OpenAI-compatible)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(mapped_kwargs)
    return payload


def maybe_send_request(provider: str, url: str, headers: Dict[str, str], payload: Dict[str, Any], live: bool | None = None) -> Dict[str, Any]:
    # CLI override takes precedence; otherwise fall back to env var (default live off)
    if live is None:
        live = os.getenv("REASONING_LIVE", "0") == "1"
    if not live:
        return {
            "live": False,
            "url": url,
            "headers": {k: ("***" if "key" in k.lower() or k.lower() == "authorization" else v) for k, v in headers.items()},
            "payload": payload,
        }

    # Google: prefer x-goog-api-key header; fall back to query param if missing
    if provider == "google":
        if "x-goog-api-key" not in headers:
            api_key = os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                return {"live": True, "status": 0, "elapsed_ms": 0, "data": {"error": "GOOGLE_API_KEY not set"}}
            url = f"{url}?key={api_key}"

    start = time.time()
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
        elapsed = time.time() - start
        try:
            data = resp.json()
        except Exception:
            data = {"text": resp.text}
        return {
            "live": True,
            "status": resp.status_code,
            "elapsed_ms": int(elapsed * 1000),
            "data": data,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"live": True, "status": -1, "elapsed_ms": int(elapsed * 1000), "data": {"error": str(e)}}


def extract_normalized(provider: str, response_obj: Dict[str, Any]) -> Dict[str, Any]:
    data = response_obj.get("data", {})
    text_snippet = None
    full_text = None
    usage = None
    reasoning_text = None
    try:
        if provider in ("openai", "fireworks"):
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            full_text = message.get("content")
            usage = data.get("usage")
            # Attempt to extract reasoning content if present (e.g., deepseek) or think-tag sections
            # DeepSeek (some APIs): reasoning_content field
            rc = message.get("reasoning_content")
            if isinstance(rc, str) and rc:
                reasoning_text = rc
            # GLM/Qwen often embed reasoning as <think> ... </think> inside content
            if reasoning_text is None and isinstance(full_text, str) and "<think>" in full_text:
                try:
                    import re as _re
                    m = _re.search(r"<think>([\s\S]*?)(</think>|$)", full_text)
                    if m:
                        reasoning_text = m.group(1).strip()
                except Exception:
                    pass
        elif provider == "anthropic":
            # messages API returns content list
            content = data.get("content")
            if isinstance(content, list) and content:
                part = content[0]
                full_text = part.get("text") or part.get("content") or data.get("content")
            usage = data.get("usage") or {
                k: data.get(k)
                for k in ("input_tokens", "output_tokens", "total_tokens")
                if k in data
            }
        elif provider == "google":
            cands = data.get("candidates", [])
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts:
                    # Concatenate non-thought text
                    texts = []
                    thoughts = []
                    for p in parts:
                        t = p.get("text")
                        if t is None:
                            continue
                        if p.get("thought") is True:
                            thoughts.append(t)
                        else:
                            texts.append(t)
                    full_text = "".join(texts) if texts else (parts[0].get("text"))
                    if thoughts:
                        reasoning_text = "\n".join(thoughts)
            usage = data.get("usageMetadata")
    except Exception:
        pass
    text_snippet = (full_text or "")[:200]
    return {"text_snippet": text_snippet, "text": full_text, "usage": usage, "reasoning_text": reasoning_text}


def test_model_reasoning_direct(model: str, reasoning_effort: str, prompt: str = TEST_PROMPT, live: bool | None = None) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print(f"Testing Model: {model}")
    print(f"Reasoning Effort: {reasoning_effort}")
    print(f"{'='*60}")

    provider, url = detect_provider_and_endpoint(model)
    print(f"Provider: {provider}")

    # Apply our mapping to construct provider-specific kwargs
    # We simulate the LiteLLM pre-api callback hook
    kwargs = {"reasoning_effort": reasoning_effort}
    mapped_kwargs = reasoning_effort_pre_api_callback(kwargs=kwargs, model_name=model)

    # Build request
    payload = build_payload(provider, model, prompt, mapped_kwargs)
    headers = build_headers(provider)

    # Show the exact request we would send
    print("Request URL:", url if provider != "google" else url.replace(model, "{model}"))
    print("Request Headers (redacted):", {k: ("***" if k.lower() in ("authorization", "x-api-key", "x-goog-api-key") else v) for k, v in headers.items()})
    print("Request Payload:")
    print(json.dumps(payload, indent=2))

    # Optionally send and show response summary
    result = maybe_send_request(provider, url, headers, payload, live=live)
    status = result.get("status")
    elapsed = result.get("elapsed_ms")
    normalized = extract_normalized(provider, result)
    # Derive simple token stats per provider
    tokens = {}
    u = normalized.get("usage") or {}
    if provider in ("openai", "fireworks"):
        tokens = {
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "total_tokens": u.get("total_tokens"),
        }
    elif provider == "anthropic":
        tokens = {
            "input_tokens": u.get("input_tokens") if isinstance(u, dict) else None,
            "output_tokens": u.get("output_tokens") if isinstance(u, dict) else None,
        }
        if tokens.get("input_tokens") is not None and tokens.get("output_tokens") is not None:
            tokens["total_tokens"] = tokens["input_tokens"] + tokens["output_tokens"]
    elif provider == "google":
        tokens = {
            "promptTokenCount": u.get("promptTokenCount") if isinstance(u, dict) else None,
            "candidatesTokenCount": u.get("candidatesTokenCount") if isinstance(u, dict) else None,
            "thoughtsTokenCount": u.get("thoughtsTokenCount") if isinstance(u, dict) else None,
            "totalTokenCount": u.get("totalTokenCount") if isinstance(u, dict) else None,
        }
    if result.get("live"):
        preview = normalized.get("text_snippet")
        reason_preview = None
        if isinstance(normalized.get("reasoning_text"), str):
            reason_preview = normalized["reasoning_text"][:300]
        print(f"Response status={status}, elapsed={elapsed}ms, tokens={tokens}, preview={preview!r}")
        if reason_preview:
            print("Reasoning (first 300 chars):")
            print(reason_preview)
    else:
        if result.get("error"):
            print("[DRY RUN]", result["error"])
        else:
            print("[DRY RUN] request constructed; set REASONING_LIVE=1 to send")

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "mapped_kwargs": mapped_kwargs,
        "request_payload": payload,
        "response": {
            "live": result.get("live"),
            "status": status,
            "elapsed_ms": elapsed,
            "raw": result.get("data"),
            "normalized": normalized,
            "tokens": tokens,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct reasoning effort tester (no LiteLLM)")
    parser.add_argument("--live", action="store_true", help="Send live API requests (default: dry run)")
    args = parser.parse_args()
    # Example configurations mirroring the earlier script, but using our mapping only
    tests = [
        ## possibly working
        # {"model": "gpt-5", "efforts": ["low", "medium"]}, # working
        # {"model": "gemini/gemini-2.5-pro", "efforts": ["low", "medium"]},
         {"model": "anthropic/claude-sonnet-4-5-20250929", "efforts": ["low", "medium"]},
        
        
        #{"model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus", "efforts": ["low", "medium"]},
        #{"model": "fireworks_ai/accounts/fireworks/models/glm-4p5", "efforts": ["low", "medium"]},
        #{"model": "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905", "efforts": ["low", "medium"]},
        #{"model": "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b", "efforts": ["low", "medium"]},
        #{"model": "fireworks_ai/accounts/fireworks/models/qwen3-30b-a3b", "efforts": ["low", "medium"]},

        # not working
        # Fireworks model not yet available based on ~/get_fireworks_models_curl (or maybe even if it is there)
        # {"model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p2-exp", "efforts": ["low", "medium"]},
        # {"model": "fireworks_ai/accounts/fireworks/models/glm-4p5-air", "efforts": ["low", "medium"]},
        
        ### {"model": "anthropic/claude-haiku-4-5.20251001-v1.0", "efforts": ["low", "medium"]},
    ]

    results = []
    for t in tests:
        for eff in t["efforts"]:
            rec = test_model_reasoning_direct(t["model"], eff, TEST_PROMPT, live=args.live)
            results.append(rec)

    # Persist to JSON under project data/reasoning_runs
    project_root = Path(__file__).resolve().parents[1]  # tau2-bench/
    out_dir = Path(os.getenv("REASONING_OUTPUT_DIR", project_root / "data" / "reasoning_runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"reasoning_run_{ts}.json"
    latest_path = out_dir / "latest.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"tests": results}, f, ensure_ascii=False, indent=2)
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump({"tests": results}, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {out_path}")
    print(f"Also wrote {latest_path}")


if __name__ == "__main__":
    main()


