import argparse
import json
import os
import sys
import platform
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger

from recorder.llm_recorder import (
    RecorderConfig,
    clear_session_context,
    get_first_system_for_session,
    install_litellm_recorder,
    set_session_context,
)
from tau2.data_model.message import AssistantMessage, Message, SystemMessage, UserMessage
from tau2.run import EvaluationType, get_tasks, run_task
from tau2.utils.utils import get_commit_hash


def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_dialog_messages(
    session_id: str,
    messages: list[Message],
    include_tools: bool,
) -> list[dict]:
    normalized: list[dict] = []
    # Prepend system if we captured it in payloads
    sys_prompt = get_first_system_for_session(session_id)
    if sys_prompt:
        normalized.append({"role": "system", "content": sys_prompt})

    for msg in messages:
        if isinstance(msg, UserMessage):
            if msg.content is not None:
                normalized.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AssistantMessage):
            # Minimal default: omit tool transcripts from dialogs.jsonl
            content = msg.content
            if include_tools and msg.is_tool_call():
                # Inline a compact note
                try:
                    tool_notes = []
                    for tc in msg.tool_calls or []:
                        tool_notes.append(f"[tool_call] {tc.name}: {json.dumps(tc.arguments, ensure_ascii=False)}")
                    note = "\n".join(tool_notes)
                    content = (content or "").strip()
                    content = f"{content}\n{note}" if content else note
                except Exception:
                    pass
            if content is not None and content != "":
                normalized.append({"role": "assistant", "content": content})
        else:
            # ToolMessage or others are omitted from FT dialog
            continue
    return normalized


@dataclass
class Args:
    domains: list[str]
    num_trials: int
    model: str
    temperature: float
    outdir: Path
    seed: Optional[int]
    debug: bool
    run_id: Optional[str]
    llm_retries: Optional[int]


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Run τ² with payload recording and dialog aggregation")
    p.add_argument("--domains", type=str, required=True, help="Comma-separated domain names (e.g., airline,mock)")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--model", type=str, default="gpt-4.1")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--outdir", type=str, default=os.environ.get("RECORDER_OUTDIR", "./recordings"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--llm-retries", type=int, default=None, help="Override LiteLLM per-call retries")
    ns = p.parse_args()
    return Args(
        domains=[d.strip() for d in ns.domains.split(",") if d.strip()],
        num_trials=ns.num_trials,
        model=ns.model,
        temperature=ns.temperature,
        outdir=Path(ns.outdir),
        seed=ns.seed,
        debug=ns.debug,
        run_id=ns.run_id,
        llm_retries=ns.llm_retries,
    )


def _sha256(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return None


def _write_manifest(run_dir: Path, run_id: str, args: Args, domain: str, tasks: list) -> None:
    litellm_cfg = Path("litellm.yaml")
    manifest = {
        "run_id": run_id,
        "timestamp": _now_id(),
        "tau2_commit": get_commit_hash(),
        "domains": [domain],
        "task_ids": [t.id for t in tasks],
        "model": args.model,
        "temperature": args.temperature,
        "num_trials": args.num_trials,
        "seed_base": 0 if args.seed is None else int(args.seed),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "litellm_config_path": str(litellm_cfg) if litellm_cfg.exists() else None,
        "litellm_config_sha256": _sha256(litellm_cfg) if litellm_cfg.exists() else None,
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    if litellm_cfg.exists():
        try:
            shutil.copy(litellm_cfg, run_dir / "litellm.yaml")
        except Exception:
            pass


def main() -> None:
    args = parse_args()
    run_id = args.run_id or _now_id()
    run_dir = Path(args.outdir) / f"run_{run_id}"
    _ensure_dir(run_dir)
    payloads_path = run_dir / "tau2_payloads.jsonl"
    dialogs_path = run_dir / "tau2_dialogs.jsonl"

    # Install recorder (monkeypatch LiteLLM)
    install_litellm_recorder(
        RecorderConfig(
            outdir=args.outdir,
            include_tools=os.environ.get("RECORDER_INCLUDE_TOOLS", "0") == "1",
            strip_think=os.environ.get("RECORDER_STRIP_THINK", "1") == "1",
            debug=args.debug or (os.environ.get("RECORDER_DEBUG", "0") == "1"),
        )
    )

    print(
        f"[Recorder] run_id={run_id} domains={args.domains} model={args.model} num_trials={args.num_trials} outdir={run_dir}"
    )

    # Force sequential execution to simplify session context handling
    # We'll call run_task directly per task/trial with EvaluationType.ALL
    base_seed = 0 if args.seed is None else int(args.seed)
    for domain in args.domains:
        # Load all tasks for this domain
        tasks = get_tasks(task_set_name=domain)
        if not tasks:
            print(f"[Recorder][warn] No tasks for domain={domain}")
            continue
        # Write manifest & snapshot config once per domain
        _write_manifest(run_dir, run_id, args, domain, tasks)
        for trial_index in range(args.num_trials):
            trial_seed = base_seed + trial_index
            for task_index, task in enumerate(tasks):
                session_id = f"{run_id}:{domain}:{task.id}:{trial_index}"
                set_session_context(session_id=session_id, domain=domain, task_id=task.id)
                try:
                    simulation = run_task(
                        domain=domain,
                        task=task,
                        agent="llm_agent",
                        user="user_simulator",
                        llm_agent=args.model,
                        llm_args_agent={
                            "temperature": args.temperature,
                            **({"num_retries": args.llm_retries} if args.llm_retries is not None else {}),
                        },
                        llm_user=args.model,
                        llm_args_user={
                            "temperature": args.temperature,
                            **({"num_retries": args.llm_retries} if args.llm_retries is not None else {}),
                        },
                        max_steps=100,
                        max_errors=10,
                        evaluation_type=EvaluationType.ALL,
                        seed=trial_seed,
                    )
                finally:
                    clear_session_context()

                # Aggregate dialog for this trial instance
                include_tools = os.environ.get("RECORDER_INCLUDE_TOOLS", "0") == "1"
                normalized_messages = _normalize_dialog_messages(
                    session_id=session_id,
                    messages=simulation.messages,
                    include_tools=include_tools,
                )
                reward = simulation.reward_info.reward if simulation.reward_info else None
                metrics = None
                if reward is not None:
                    try:
                        metrics = {"success": bool(reward and reward > 0), "score": float(reward)}
                    except Exception:
                        metrics = {"success": None, "score": reward}

                dialog_record = {
                    "session_id": session_id,
                    "domain": domain,
                    "task_id": task.id,
                    "messages": normalized_messages,
                    "metrics": metrics,
                }
                with open(dialogs_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dialog_record, ensure_ascii=False) + "\n")

            # Trial-level sanity print
            has_system = get_first_system_for_session(session_id) is not None
            print(
                f"[Recorder] session_id={session_id} turns={len(simulation.messages)} has_system={has_system}"
            )

    # Final sanity print (counts)
    def _count_lines(p: Path) -> int:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return 0

    payload_lines = _count_lines(payloads_path)
    dialog_lines = _count_lines(dialogs_path)
    print(
        f"[Recorder] Done. payloads={payload_lines} lines, dialogs={dialog_lines} lines written to {args.outdir}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)


