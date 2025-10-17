#!/usr/bin/env python3
"""
Retry infrastructure failures from a previous run.

Reads tau2_dialogs.jsonl to identify failed simulations, then retries them
with the exact same configuration and seeds.

Usage:
    python -m recorder.retry_failures <run_dir> [--max-retries N] [--max-workers N]
    
Example:
    python -m recorder.retry_failures recordings/run_20251014-213614_airline_gemini-2.5-flash_temp1.0_tr4/
    python -m recorder.retry_failures recordings/run_20251014-213614_airline_gemini-2.5-flash_temp1.0_tr4/ --max-retries 3 --max-workers 2
    python -m recorder.retry_failures recordings/run_20251014-180208_airline_claude-sonnet-4-5-20250929/

    python -m recorder.analyze_dialogs recordings/run_20251014-213614_airline_gemini-2.5-flash_temp1.0_tr4/tau2_dialogs.jsonl
    python -m recorder.analyze_dialogs recordings/run_20251014-180208_airline_claude-sonnet-4-5-20250929/tau2_dialogs.jsonl

    python -m recorder.retry_failures recordings/run_20251014-234430_airline_gpt-5-mini_temp1.0_tr4/ --max-retries 3 --max-workers 2 --dry-run
    python -m recorder.retry_failures `ls -t recordings/*${RUN_ID}/tau2_dialogs.jsonl | head -n 1` --max-retries 3 --max-workers 2 --dry-run

"""

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger

from recorder.common_args import RecorderArgs as Args, add_model_args, merge_args_with_manifest
from recorder.llm_recorder import RecorderConfig, install_litellm_recorder
from recorder.run_and_record import _run_single_trial
from tau2.run import get_tasks


@dataclass
class FailedTrial:
    """Information about a failed trial that needs retry."""
    domain: str
    task_id: str
    trial_index: int
    trial_seed: int
    session_id: str
    failure_reason: str


def identify_missing_trials(dialogs_path: Path, manifest: dict) -> list[FailedTrial]:
    """Identify which task:trial combinations are missing legitimate conversations.
    
    A task:trial combination is considered missing if it has NO legitimate 
    (non-infra-failure) conversations, even if it has multiple infra failure entries.
    
    This ensures we only retry combinations that truly lack successful runs,
    not combinations that have both infra failures and successful retries.
    
    Args:
        dialogs_path: Path to tau2_dialogs.jsonl
        manifest: Run manifest dict with task_ids, num_trials, etc.
    
    Returns:
        List of FailedTrial objects for missing task:trial combinations
    """
    # Load all dialogs
    dialogs = []
    with open(dialogs_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                dialog = json.loads(line)
                dialogs.append(dialog)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                continue
    
    # Track which (task_id, trial_index) combinations have legitimate conversations
    legitimate_combinations = set()
    
    for dialog in dialogs:
        # Skip infrastructure failures - we only care about legitimate conversations
        if dialog.get("metrics", {}).get("infra_failure", False):
            continue
        
        # This is a legitimate conversation (success or failure, but not infra)
        session_id = dialog.get("session_id", "")
        parts = session_id.split(":")
        if len(parts) >= 4:
            task_id = parts[-2]
            trial_index = int(parts[-1])
            legitimate_combinations.add((task_id, trial_index))
    
    # Determine expected combinations from manifest
    task_ids = manifest["task_ids"]
    num_trials = manifest["num_trials"]
    domain = manifest["domains"][0]
    seed_base = manifest.get("seed_base", 0)
    run_id = manifest["run_id"]
    
    # Find missing combinations
    missing_trials = []
    for task_id in task_ids:
        for trial_index in range(num_trials):
            if (str(task_id), trial_index) not in legitimate_combinations:
                # This combination is missing a legitimate conversation - needs retry
                session_id = f"{run_id}:{domain}:{task_id}:{trial_index}"
                missing_trials.append(FailedTrial(
                    domain=domain,
                    task_id=str(task_id),
                    trial_index=trial_index,
                    trial_seed=seed_base + trial_index,
                    session_id=session_id,
                    failure_reason="missing_legitimate_conversation",
                ))
    
    return missing_trials


def load_manifest(run_dir: Path) -> dict:
    """Load run_manifest.json from the run directory."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Retry infrastructure failures from a previous run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("run_dir", type=str, help="Path to run directory (e.g., recordings/run_YYYYMMDD-HHMMSS)")
    
    # Add model override arguments (user can override user model for retries)
    add_model_args(parser, allow_user_override=True)
    
    # Add execution override arguments
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Override max workers (default: use from manifest)")
    parser.add_argument("--max-retries", type=int, default=None,
                       help="Override infra retry attempts (default: 3)")
    
    parser.add_argument("--dry-run", action="store_true", help="Show what would be retried without actually retrying")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output file (default: append to original dialogs file)")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    
    dialogs_path = run_dir / "tau2_dialogs.jsonl"
    if not dialogs_path.exists():
        print(f"Error: Dialogs file not found: {dialogs_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load manifest first (needed to identify missing trials)
    manifest = load_manifest(run_dir)
    
    # Identify missing task:trial combinations
    print(f"[Retry] Analyzing {dialogs_path}...")
    missing_trials = identify_missing_trials(dialogs_path, manifest)
    
    if not missing_trials:
        print("[Retry] All task:trial combinations have legitimate conversations. Nothing to retry.")
        return
    
    print(f"[Retry] Found {len(missing_trials)} missing task:trial combinations (out of {len(manifest['task_ids'])} tasks × {manifest['num_trials']} trials = {len(manifest['task_ids']) * manifest['num_trials']} expected):")
    
    # Group by task for display
    by_task = defaultdict(list)
    for f in missing_trials:
        by_task[f.task_id].append(f.trial_index)
    
    for task_id in sorted(by_task.keys()):
        trials = sorted(by_task[task_id])
        print(f"  Task {task_id}: trials {trials}")
    
    if args.dry_run:
        print("[Retry] Dry run mode - exiting without retrying")
        return
    
    # Reconstruct Args using merge function (allows CLI overrides)
    run_args = merge_args_with_manifest(
        manifest,
        user_model_override=args.user_model,
        user_temperature_override=args.user_temperature,
        max_workers=args.max_workers,
        infra_retries=args.max_retries or 3,  # Default to 3 if not specified
    )
    
    # Override outdir to point to parent directory
    run_args.outdir = run_dir.parent
    
    # Determine output path
    if args.output_suffix:
        output_dialogs_path = run_dir / f"tau2_dialogs{args.output_suffix}.jsonl"
        print(f"[Retry] Will write retries to new file: {output_dialogs_path}")
    else:
        output_dialogs_path = dialogs_path
        print(f"[Retry] Will append retries to existing file: {output_dialogs_path}")
    
    # Install recorder
    install_litellm_recorder(
        RecorderConfig(
            outdir=run_args.outdir,
            include_tools=os.environ.get("RECORDER_INCLUDE_TOOLS", "0") == "1",
            strip_think=os.environ.get("RECORDER_STRIP_THINK", "1") == "1",
            debug=os.environ.get("RECORDER_DEBUG", "0") == "1",
        )
    )
    
    # Load tasks for the domain
    domain = missing_trials[0].domain
    all_tasks = get_tasks(task_set_name=domain)
    tasks_by_id = {str(t.id): t for t in all_tasks}
    
    # Build work items
    work_items = []
    dialogs_write_lock = threading.Lock()
    
    for failure in missing_trials:
        task = tasks_by_id.get(failure.task_id)
        if task is None:
            logger.warning(f"Task {failure.task_id} not found in domain {domain}, skipping")
            continue
        
        work_items.append((
            failure.domain,
            task,
            failure.trial_index,
            failure.trial_seed,
            manifest["run_id"],
            run_args,
            output_dialogs_path,
            dialogs_write_lock,
        ))
    
    print(f"[Retry] Starting {len(work_items)} retry attempts with {args.max_workers} workers...")
    print(f"[Retry] Max retries per trial: {args.max_retries}")
    
    # Run retries in parallel
    retry_results = {"completed": 0, "infra_failed": 0}
    
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(_run_single_trial, *item) for item in work_items]
            
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    simulation = future.result()
                    if simulation is not None:
                        retry_results["completed"] += 1
                        print(f"[Retry] Progress: {i}/{len(work_items)} - ✓ Completed (simulation ran)")
                    else:
                        retry_results["infra_failed"] += 1
                        print(f"[Retry] Progress: {i}/{len(work_items)} - ✗ Infra failure")
                except Exception as e:
                    retry_results["infra_failed"] += 1
                    print(f"[Retry] Progress: {i}/{len(work_items)} - ✗ Infra failure: {e}")
    
    except KeyboardInterrupt:
        print("\n[Retry] Interrupted by user (Ctrl-C)")
        sys.exit(130)
    
    # Summary
    print("\n" + "=" * 70)
    print("[Retry] SUMMARY")
    print("=" * 70)
    print(f"Total retry attempts: {len(work_items)}")
    print(f"Completed (generated conversation): {retry_results['completed']}")
    print(f"Infrastructure failures (crashed/exception): {retry_results['infra_failed']}")
    print(f"Completion rate: {retry_results['completed'] / len(work_items) * 100:.1f}%")
    print(f"\nDialogs written to: {output_dialogs_path}")
    print("Note: Completed conversations may have task success or failure (check analyze_dialogs for details)")
    
    if retry_results['infra_failed'] > 0:
        print(f"\n⚠️  {retry_results['infra_failed']} retries had infrastructure failures after {args.max_retries or 3} attempts")
        print("Consider: --user-model gpt-4.1 --user-temperature 0.0, or --max-workers 1")


if __name__ == "__main__":
    main()

