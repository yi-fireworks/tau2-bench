# llm_recorder_v1 original design doc

Minimal, transparent recorder for τ² runs using a LiteLLM boundary shim, plus a thin CLI that aggregates per-trial dialogs for fine-tuning export.

## Overview
- Recorder hooks at the LLM invocation boundary by monkeypatching `litellm.completion`.
- Every LLM call writes one line to `recordings/tau2_payloads.jsonl` capturing the full request (incl. system) and response summary.
- A simple runner executes τ² tasks sequentially, injects a stable `session_id`, and writes one line per trial to `recordings/tau2_dialogs.jsonl`.
- Design goals: minimal surface area, explicit behavior, fail fast with visible, low-volume stdout sanity prints.

## Where it hooks
- Hook point: `litellm.completion` (see `recorder/llm_recorder.py`).
- Session context is injected per task/trial via `set_session_context(session_id, domain, task_id)` in `recorder/run_and_record.py` before every `run_task` call.
- Turn indices are tracked per `session_id` inside the recorder.

## Files
- `recorder/llm_recorder.py` — installs recorder, writes `tau2_payloads.jsonl`.
- `recorder/run_and_record.py` — CLI runner, writes `tau2_dialogs.jsonl`.

## Output artifacts
- Directory: `./recordings/` (created on first run or overridden by `RECORDER_OUTDIR`).

### tau2_payloads.jsonl (one line per LLM call)
Example (shortened):
```json
{
  "session_id": "20250101-120000:mock:task_1:0",
  "domain": "mock",
  "task_id": "task_1",
  "turn_index": 0,
  "request": {
    "model": "gpt-4.1",
    "messages": [
      {"role":"system","content":"<system prompt>"},
      {"role":"user","content":"hi"}
    ],
    "tools": null,
    "temperature": 0.2,
    "top_p": 1.0,
    "extra_params": {"seed": 42}
  },
  "response": {
    "raw": null,
    "assistant_text": "Hello! How can I help you today?"
  },
  "timestamps": {"start": "...Z", "end": "...Z"}
}
```

### tau2_dialogs.jsonl (one line per trial)
Example (shortened):
```json
{
  "session_id": "20250101-120000:mock:task_1:0",
  "domain": "mock",
  "task_id": "task_1",
  "messages": [
    {"role":"system","content":"<system prompt>"},
    {"role":"user","content":"hi"},
    {"role":"assistant","content":"Hello! How can I help you today?"}
  ],
  "metrics": {"success": true, "score": 0.78}
}
```

Notes
- Roles in dialogs are strictly `{system,user,assistant}`; tool transcripts are omitted by default (see toggle below).
- `system` is pulled from the first recorded request per session.

## CLI usage
Run sequentially across domains/tasks/trials using the programmatic τ² API:
```bash
python -m recorder.run_and_record \
  --domains mock \
  --num-trials 1 \
  --model gpt-4.1 \
  --temperature 0.2 \
  --outdir ./recordings
```

Args
- `--domains`: comma-separated domains (e.g., `airline,mock`).
- `--num-trials`: number of trials per task.
- `--model`: LiteLLM model string.
- `--temperature`: sampling temperature.
- `--outdir`: output directory (default `./recordings`).
- `--seed`: optional seed.
- `--debug`: include raw provider JSON in payloads.

## ENV toggles
- `RECORDER_OUTDIR` (default `./recordings`) — output dir.
- `RECORDER_INCLUDE_TOOLS` (default `0`) — if `1`, inline compact `[tool_call] ...` notes into assistant messages in `dialogs.jsonl`.
- `RECORDER_STRIP_THINK` (default `1`) — reserved switch for stripping `<think>` if used by providers.
- `RECORDER_DEBUG` (default `0`) — if `1` (or `--debug`), include `response.raw` in payloads.

## Sanity prints & checks
- First LLM call: prints recorder path, `has_system`, model, and message count.
- Per session: warns once if first call has no system.
- Per turn: warns if `assistant_text` is empty (tool-only responses are allowed; remains recorded).
- End of run: prints counts for payload and dialog lines.

## Data hygiene
- Common secret fields in request are redacted (`api_key`, `authorization`, etc.).
- Consider disabling `response.raw` unless needed (`--debug` or `RECORDER_DEBUG=1`).

## Limitations & design choices
- Recorder is provider-agnostic by using LiteLLM. Works with Hugging Face endpoints (HF Inference/TGI), OpenAI-compatible servers, and others supported by LiteLLM.
- Dialog aggregation prefers recorded system from payloads; if none, dialog may start with user (a warning is printed).
- Runner executes sequentially to simplify session scoping and ordering.

## Troubleshooting
- No output files: ensure `install_litellm_recorder` is being called (done by the CLI) and that τ² is using LiteLLM via `tau2.utils.llm_utils.generate`.
- Missing system prompt: confirm your agent’s first message includes a `system` entry; or check if a custom agent bypasses the standard call path.
- Empty `assistant_text`: expected when the model returns only tool calls; payload still records the raw response (in debug) and the turn is counted.
- Tool transcripts in dialogs: set `RECORDER_INCLUDE_TOOLS=1` to inline compact, human-readable notes.

## Related code locations
- Hook: `recorder/llm_recorder.py`
- Runner: `recorder/run_and_record.py`
- τ² LLM boundary: `tau2/utils/llm_utils.py` (`litellm.completion` is called there)
