import argparse
import json
import os
import sys
import time
import platform
import hashlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import litellm
from loguru import logger

from recorder.common_args import RecorderArgs as Args, add_model_args, add_execution_args
from recorder.llm_recorder import (
    RecorderConfig,
    clear_session_context,
    get_first_system_for_session,
    install_litellm_recorder,
    set_session_context,
)
from tau2.data_model.message import AssistantMessage, Message, SystemMessage, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.data_model.tasks import Task
from tau2.run import EvaluationType, get_tasks, run_task
from tau2.utils.llm_utils import to_litellm_messages
from tau2.utils.reasoning_effort import reasoning_effort_pre_api_callback
from tau2.utils.utils import get_commit_hash


def _now_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Run τ² with payload recording and dialog aggregation")
    p.add_argument("--domains", type=str, required=True, help="Comma-separated domain names (e.g., airline,mock)")
    p.add_argument("--num-trials", type=int, default=1)
    
    # Add shared model arguments
    add_model_args(p, allow_user_override=True)
    
    p.add_argument("--outdir", type=str, default=os.environ.get("RECORDER_OUTDIR", "./recordings"))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    
    # Add shared execution arguments
    add_execution_args(p)
    
    ns = p.parse_args()
    return Args(
        domains=[d.strip() for d in ns.domains.split(",") if d.strip()],
        num_trials=ns.num_trials,
        model=ns.model,
        temperature=ns.temperature,
        user_model=ns.user_model or "gemini-2.5-pro",
        user_temperature=ns.user_temperature if ns.user_temperature is not None else 0.0,
        outdir=Path(ns.outdir),
        seed=ns.seed,
        debug=ns.debug,
        run_id=ns.run_id,
        llm_retries=ns.llm_retries,
        reasoning_effort_agent=ns.reasoning_effort_agent,
        reasoning_effort_user=ns.reasoning_effort_user,
        budget_or_max_tokens_agent=ns.budget_agent,
        budget_or_max_tokens_user=ns.budget_user,
        max_steps=ns.max_steps,
        max_workers=ns.max_workers,
        infra_retries=ns.infra_retries,
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
        "user_model": args.user_model,
        "user_temperature": args.user_temperature,
        "reasoning_effort_agent": args.reasoning_effort_agent,
        "reasoning_effort_user": args.reasoning_effort_user,
        "budget_or_max_tokens_agent": args.budget_or_max_tokens_agent,
        "budget_or_max_tokens_user": args.budget_or_max_tokens_user,
        "num_trials": args.num_trials,
        "max_workers": args.max_workers,
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


def _run_single_trial(
    domain: str,
    task: Task,
    trial_index: int,
    trial_seed: int,
    run_id: str,
    args: Args,
    dialogs_path: Path,
    dialogs_write_lock: threading.Lock,
) -> Optional[SimulationRun]:
    """Run a single task/trial combination and record the dialog."""
    session_id = f"{run_id}:{domain}:{task.id}:{trial_index}"
    
    for attempt in range(args.infra_retries + 1):
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
                    # reasoning_effort is overridden by budget_or_max_tokens if provided
                    "reasoning_effort": args.reasoning_effort_agent,
                    "budget_or_max_tokens": args.budget_or_max_tokens_agent,
                    **({"num_retries": args.llm_retries} if args.llm_retries is not None else {}),
                },
                llm_user=args.user_model,
                llm_args_user={
                    "temperature": args.user_temperature,
                    # reasoning_effort is overridden by budget_or_max_tokens if provided
                    "reasoning_effort": args.reasoning_effort_user,
                    "budget_or_max_tokens": args.budget_or_max_tokens_user,
                    **({"num_retries": args.llm_retries} if args.llm_retries is not None else {}),
                },
                max_steps=args.max_steps,
                max_errors=10,
                evaluation_type=EvaluationType.ALL,
                seed=trial_seed,
            )
            infra_failure = False
            infra_failure_reason = None
            clear_session_context()
            break  # Success - exit retry loop
        except Exception as e:
            clear_session_context()
            if attempt < args.infra_retries:
                wait_time = 2 ** attempt
                logger.warning(f"Retry {attempt + 1}/{args.infra_retries} for {session_id} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to run {session_id} after {args.infra_retries + 1} attempts: {e}")
                # Mark as infrastructure failure - should be excluded from RFT
                infra_failure = True
                infra_failure_reason = str(e)
                simulation = None
    
    # If infrastructure failure, record a minimal entry
    if infra_failure:
        dialog_record = {
            "session_id": session_id,
            "domain": domain,
            "task_id": task.id,
            "messages": [],
            "metrics": {
                "success": None,
                "score": None,
                "infra_failure": True,
                "infra_failure_reason": infra_failure_reason,
            },
        }
        with dialogs_write_lock:
            with open(dialogs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dialog_record, ensure_ascii=False) + "\n")
        return None

    # Convert messages to OpenAI format for recording
    # Uses tau2's standard conversion (works with all providers)
    normalized_messages = to_litellm_messages(simulation.messages)
    
    # Enhanced metrics for SFT/RFT filtering and analysis
    metrics = None
    if simulation.reward_info:
        try:
            metrics = {
                # Basic success indicators
                "success": bool(simulation.reward_info.reward and simulation.reward_info.reward > 0),
                "score": float(simulation.reward_info.reward),
                
                # Component-level breakdown (critical for RFT filtering)
                "reward_breakdown": simulation.reward_info.reward_breakdown,
                "reward_basis": simulation.reward_info.reward_basis,
                
                # Trajectory characteristics
                "termination_reason": simulation.termination_reason,
                "num_steps": len(simulation.messages),
                
                # Detailed component checks
                "db_success": simulation.reward_info.db_check.db_match if simulation.reward_info.db_check else None,
                "num_action_checks": len(simulation.reward_info.action_checks) if simulation.reward_info.action_checks else 0,
                "action_success_rate": (
                    sum(1 for ac in simulation.reward_info.action_checks if ac.action_match) / len(simulation.reward_info.action_checks)
                    if simulation.reward_info.action_checks else None
                ),
                "num_communicate_checks": len(simulation.reward_info.communicate_checks) if simulation.reward_info.communicate_checks else 0,
                "communicate_success_rate": (
                    sum(1 for cc in simulation.reward_info.communicate_checks if cc.met) / len(simulation.reward_info.communicate_checks)
                    if simulation.reward_info.communicate_checks else None
                ),
                
                # Infrastructure failure indicator (False for successful runs)
                "infra_failure": False,
            }
        except Exception as e:
            # Fallback to basic metrics if detailed parsing fails
            logger.warning(f"Failed to extract detailed metrics for {session_id}: {e}")
            metrics = {
                "success": bool(simulation.reward_info.reward and simulation.reward_info.reward > 0),
                "score": float(simulation.reward_info.reward),
                "infra_failure": False,
            }
    else:
        metrics = {"success": None, "score": None, "infra_failure": False}

    dialog_record = {
        "session_id": session_id,
        "domain": domain,
        "task_id": task.id,
        "messages": normalized_messages,
        "metrics": metrics,
    }
    
    # Thread-safe write to dialogs file
    with dialogs_write_lock:
        with open(dialogs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dialog_record, ensure_ascii=False) + "\n")

    return simulation


def main() -> None:
    args = parse_args()
    run_id = args.run_id or _now_id()
    run_dir = Path(args.outdir) / f"run_{run_id}"
    _ensure_dir(run_dir)
    payloads_path = run_dir / "tau2_payloads.jsonl"
    dialogs_path = run_dir / "tau2_dialogs.jsonl"
    
    # Add file logging for errors and warnings
    error_log = run_dir / "errors.log"
    logger.add(
        error_log,
        level="WARNING",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=None,
        retention=None,
    )
    logger.info(f"Error logging enabled to: {error_log}")

    # Register LiteLLM callback to handle reasoning effort and budget
    litellm.pre_api_callback = [reasoning_effort_pre_api_callback]

    # Install recorder (monkeypatch LiteLLM)
    install_litellm_recorder(
        RecorderConfig(
            outdir=run_dir,
            include_tools=os.environ.get("RECORDER_INCLUDE_TOOLS", "0") == "1",
            strip_think=os.environ.get("RECORDER_STRIP_THINK", "1") == "1",
            debug=args.debug or (os.environ.get("RECORDER_DEBUG", "0") == "1"),
        )
    )

    print(
        f"[Recorder] run_id={run_id} domains={args.domains} model={args.model} num_trials={args.num_trials} max_workers={args.max_workers} outdir={run_dir}"
    )
    print(f"[Recorder] Error log: {error_log}")

    # Parallel execution with ThreadPoolExecutor
    base_seed = 0 if args.seed is None else int(args.seed)
    dialogs_write_lock = threading.Lock()
    
    for domain in args.domains:
        # Load all tasks for this domain
        tasks = get_tasks(task_set_name=domain)
        if not tasks:
            print(f"[Recorder][warn] No tasks for domain={domain}")
            continue
        
        # Write manifest & snapshot config once per domain
        _write_manifest(run_dir, run_id, args, domain, tasks)
        
        # Build work items for parallel execution
        work_items = []
        for trial_index in range(args.num_trials):
            trial_seed = base_seed + trial_index
            for task in tasks:
                work_items.append((
                    domain,
                    task,
                    trial_index,
                    trial_seed,
                    run_id,
                    args,
                    dialogs_path,
                    dialogs_write_lock,
                ))
        
        print(f"[Recorder] Starting {len(work_items)} simulations for domain={domain} with {args.max_workers} workers...")
        
        # Run in parallel using ThreadPoolExecutor with proper KeyboardInterrupt handling
        simulations = []
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Submit all tasks and get futures
                futures = [executor.submit(_run_single_trial, *item) for item in work_items]
                
                # Collect results as they complete (allows Ctrl-C to work)
                for future in as_completed(futures):
                    try:
                        simulations.append(future.result())
                    except Exception as e:
                        # Individual task failures are already logged, just skip
                        simulations.append(None)
        except KeyboardInterrupt:
            print("\n[Recorder] Interrupted by user (Ctrl-C). Cleaning up...")
            # Note: ThreadPoolExecutor.__exit__ will wait for running tasks to complete
            raise
        
        # Filter out failed simulations and report summary
        successful_sims = [s for s in simulations if s is not None]
        failed_count = len(simulations) - len(successful_sims)
        
        print(
            f"[Recorder] Completed domain={domain}: {len(successful_sims)}/{len(simulations)} successful "
            f"({failed_count} failed)"
        )
        
        # Sample sanity print from a successful simulation
        if successful_sims:
            last_sim = successful_sims[-1]
            last_session_id = f"{run_id}:{domain}:{last_sim.task_id}:{args.num_trials-1}"
            has_system = get_first_system_for_session(last_session_id) is not None
            print(
                f"[Recorder] Sample simulation: turns={len(last_sim.messages)} has_system={has_system}"
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


