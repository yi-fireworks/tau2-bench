#!/usr/bin/env python3
"""
Analyze tau2_dialogs.jsonl files to compute summary statistics.

Usage:
    python -m recorder.analyze_dialogs <dialogs_jsonl_path> [--output <csv_path>]
    
Example:
    python -m recorder.analyze_dialogs recordings/run_20251014-175631_airline_gemini-2.5-flash-lite/tau2_dialogs.jsonl
    python -m recorder.analyze_dialogs recordings/run_20251014-175631_airline_gemini-2.5-flash-lite/tau2_dialogs.jsonl --output analysis.csv
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def load_dialogs(path: Path) -> List[Dict[str, Any]]:
    """Load all dialog records from a JSONL file."""
    dialogs = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                dialogs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}", file=sys.stderr)
    return dialogs


def compute_summary_stats(dialogs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute comprehensive summary statistics from dialog records."""
    if not dialogs:
        return {"error": "No dialogs found"}
    
    # Basic counts
    total = len(dialogs)
    
    # Separate infrastructure failures from legitimate results
    infra_failures = [d for d in dialogs if d.get("metrics", {}).get("infra_failure", False)]
    legitimate_dialogs = [d for d in dialogs if not d.get("metrics", {}).get("infra_failure", False)]
    
    # Extract metrics
    successes = [d for d in legitimate_dialogs if d.get("metrics", {}).get("success")]
    failures = [d for d in legitimate_dialogs if d.get("metrics", {}).get("success") is False]
    
    # Score statistics
    scores = [d["metrics"]["score"] for d in legitimate_dialogs if d.get("metrics", {}).get("score") is not None]
    
    # Termination reasons
    termination_reasons = Counter(
        d["metrics"].get("termination_reason") 
        for d in legitimate_dialogs 
        if d.get("metrics", {}).get("termination_reason")
    )
    
    # Step counts (conversation length) - use num_steps if available, else count messages
    step_counts = []
    for d in legitimate_dialogs:
        if d.get("metrics", {}).get("num_steps"):
            step_counts.append(d["metrics"]["num_steps"])
        elif d.get("messages"):
            step_counts.append(len(d["messages"]))
    
    # Reward basis distribution
    reward_bases = defaultdict(int)
    for d in legitimate_dialogs:
        bases = d.get("metrics", {}).get("reward_basis")
        if bases:
            for basis in bases:
                reward_bases[basis] += 1
    
    # DB success rates
    db_checks = [d["metrics"]["db_success"] for d in legitimate_dialogs if d.get("metrics", {}).get("db_success") is not None]
    
    # Action check statistics
    action_rates = [
        d["metrics"]["action_success_rate"] 
        for d in legitimate_dialogs 
        if d.get("metrics", {}).get("action_success_rate") is not None
    ]
    
    # Communication check statistics
    comm_rates = [
        d["metrics"]["communicate_success_rate"] 
        for d in legitimate_dialogs 
        if d.get("metrics", {}).get("communicate_success_rate") is not None
    ]
    
    # Per-task breakdown
    task_stats = defaultdict(lambda: {"total": 0, "success": 0, "scores": []})
    for d in legitimate_dialogs:
        task_id = d.get("task_id")
        task_stats[task_id]["total"] += 1
        if d.get("metrics", {}).get("success"):
            task_stats[task_id]["success"] += 1
        score = d.get("metrics", {}).get("score")
        if score is not None:
            task_stats[task_id]["scores"].append(score)
    
    # Per-trial breakdown
    trial_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for d in legitimate_dialogs:
        session_id = d.get("session_id", "")
        # Extract trial index from session_id (format: run_id:domain:task_id:trial_index)
        parts = session_id.split(":")
        if len(parts) >= 4:
            trial_idx = parts[-1]
            trial_stats[trial_idx]["total"] += 1
            if d.get("metrics", {}).get("success"):
                trial_stats[trial_idx]["success"] += 1
    
    # Build summary
    summary = {
        "total_dialogs": total,
        "infra_failures": len(infra_failures),
        "legitimate_dialogs": len(legitimate_dialogs),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": len(successes) / len(legitimate_dialogs) if len(legitimate_dialogs) > 0 else 0.0,
        
        "score_mean": sum(scores) / len(scores) if scores else 0.0,
        "score_median": sorted(scores)[len(scores) // 2] if scores else 0.0,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        
        "steps_mean": sum(step_counts) / len(step_counts) if step_counts else 0.0,
        "steps_median": sorted(step_counts)[len(step_counts) // 2] if step_counts else 0,
        "steps_min": min(step_counts) if step_counts else None,
        "steps_max": max(step_counts) if step_counts else None,
        
        "termination_reasons": dict(termination_reasons),
        "reward_bases": dict(reward_bases),
        
        "db_success_rate": sum(db_checks) / len(db_checks) if db_checks else None,
        
        "action_success_rate_mean": sum(action_rates) / len(action_rates) if action_rates else None,
        "comm_success_rate_mean": sum(comm_rates) / len(comm_rates) if comm_rates else None,
        
        "num_unique_tasks": len(task_stats),
        "num_trials": len(trial_stats),
    }
    
    # Add per-task success rates (top 10 worst performing)
    task_success_rates = [
        (task_id, stats["success"] / stats["total"] if stats["total"] > 0 else 0.0)
        for task_id, stats in task_stats.items()
    ]
    task_success_rates.sort(key=lambda x: x[1])
    summary["worst_tasks"] = task_success_rates[:10]
    
    # Add per-trial success rates
    trial_success_rates = [
        (trial_id, stats["success"] / stats["total"] if stats["total"] > 0 else 0.0)
        for trial_id, stats in trial_stats.items()
    ]
    trial_success_rates.sort(key=lambda x: x[0])
    summary["trial_success_rates"] = trial_success_rates
    
    # Add histogram: for each task, count how many trials succeeded
    # Then group tasks by their success count
    task_success_histogram = defaultdict(int)  # {num_successes: num_tasks}
    for task_id, stats in task_stats.items():
        num_successes = stats["success"]
        task_success_histogram[num_successes] += 1
    
    # Convert to sorted list for display
    summary["task_success_histogram"] = sorted(task_success_histogram.items())
    summary["num_trials_per_task"] = len(trial_stats)  # Number of trials
    
    return summary


def format_summary(summary: Dict[str, Any]) -> str:
    """Format summary statistics as a readable string."""
    lines = []
    lines.append("=" * 70)
    lines.append("TAU2 DIALOG ANALYSIS SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append(f"Total dialogs: {summary['total_dialogs']}")
    
    if summary.get('infra_failures', 0) > 0:
        lines.append(f"Infrastructure failures: {summary['infra_failures']} (excluded from analysis)")
        lines.append(f"Legitimate dialogs: {summary['legitimate_dialogs']}")
    
    lines.append(f"Successes: {summary['successes']} ({summary['success_rate']:.1%})")
    lines.append(f"Failures: {summary['failures']}")
    lines.append("")
    
    lines.append("SCORE STATISTICS:")
    lines.append(f"  Mean:   {summary['score_mean']:.3f}")
    lines.append(f"  Median: {summary['score_median']:.3f}")
    lines.append(f"  Range:  [{summary['score_min']}, {summary['score_max']}]")
    lines.append("")
    
    lines.append("CONVERSATION LENGTH (message count):")
    lines.append(f"  Mean:   {summary['steps_mean']:.1f}")
    lines.append(f"  Median: {summary['steps_median']}")
    lines.append(f"  Range:  [{summary['steps_min']}, {summary['steps_max']}]")
    lines.append("")
    
    if summary['termination_reasons']:
        lines.append("TERMINATION REASONS:")
        for reason, count in sorted(summary['termination_reasons'].items(), key=lambda x: -x[1]):
            pct = count / summary['total_dialogs'] * 100
            lines.append(f"  {reason:20s}: {count:4d} ({pct:5.1f}%)")
        lines.append("")
    
    if summary['reward_bases']:
        lines.append("REWARD BASIS DISTRIBUTION:")
        for basis, count in sorted(summary['reward_bases'].items(), key=lambda x: -x[1]):
            pct = count / summary['total_dialogs'] * 100
            lines.append(f"  {basis:20s}: {count:4d} ({pct:5.1f}%)")
        lines.append("")
    
    if summary.get('db_success_rate') is not None:
        lines.append(f"DB Check Success Rate: {summary['db_success_rate']:.1%}")
    if summary.get('action_success_rate_mean') is not None:
        lines.append(f"Action Success Rate (mean): {summary['action_success_rate_mean']:.1%}")
    if summary.get('comm_success_rate_mean') is not None:
        lines.append(f"Communication Success Rate (mean): {summary['comm_success_rate_mean']:.1%}")
    lines.append("")
    
    lines.append(f"Unique Tasks: {summary['num_unique_tasks']}")
    lines.append(f"Trials: {summary['num_trials']}")
    lines.append("")
    
    if summary.get('worst_tasks'):
        lines.append("WORST PERFORMING TASKS (lowest success rate):")
        for task_id, rate in summary['worst_tasks']:
            lines.append(f"  {task_id:20s}: {rate:.1%}")
        lines.append("")
    
    if summary.get('trial_success_rates'):
        lines.append("PER-TRIAL SUCCESS RATES:")
        for trial_id, rate in summary['trial_success_rates']:
            lines.append(f"  Trial {trial_id}: {rate:.1%}")
        lines.append("")
    
    if summary.get('task_success_histogram'):
        lines.append("TASK SUCCESS LEVEL HISTOGRAM:")
        lines.append(f"(How many tasks achieved exactly N successful trials out of {summary.get('num_trials_per_task', '?')})")
        lines.append("")
        total_tasks = sum(count for _, count in summary['task_success_histogram'])
        for num_successes, num_tasks in summary['task_success_histogram']:
            pct = 100.0 * num_tasks / total_tasks if total_tasks > 0 else 0.0
            bar = "█" * int(pct / 2)  # Scale bar to ~50 chars max
            lines.append(f"  {num_successes} successes: {num_tasks:3d} tasks ({pct:5.1f}%) {bar}")
        lines.append("")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def export_to_csv(dialogs: List[Dict[str, Any]], output_path: Path):
    """Export dialog metrics to CSV for further analysis."""
    rows = []
    for d in dialogs:
        m = d.get("metrics", {})
        row = {
            "session_id": d.get("session_id"),
            "domain": d.get("domain"),
            "task_id": d.get("task_id"),
            "success": m.get("success"),
            "score": m.get("score"),
            "termination_reason": m.get("termination_reason"),
            "num_steps": m.get("num_steps"),
            "db_success": m.get("db_success"),
            "num_action_checks": m.get("num_action_checks"),
            "action_success_rate": m.get("action_success_rate"),
            "num_communicate_checks": m.get("num_communicate_checks"),
            "communicate_success_rate": m.get("communicate_success_rate"),
            "reward_basis": ",".join(m.get("reward_basis", [])) if m.get("reward_basis") else None,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(rows)} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tau2_dialogs.jsonl files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("dialogs_path", type=str, help="Path to tau2_dialogs.jsonl file")
    parser.add_argument("--output", "-o", type=str, help="Optional: Export metrics to CSV")
    parser.add_argument("--json", action="store_true", help="Output summary as JSON")
    
    args = parser.parse_args()
    
    dialogs_path = Path(args.dialogs_path)
    if not dialogs_path.exists():
        print(f"Error: File not found: {dialogs_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading dialogs from {dialogs_path}...", file=sys.stderr)
    dialogs = load_dialogs(dialogs_path)
    print(f"Loaded {len(dialogs)} dialog records", file=sys.stderr)
    print("", file=sys.stderr)
    
    summary = compute_summary_stats(dialogs)
    
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_summary(summary))
    
    if args.output:
        output_path = Path(args.output)
        export_to_csv(dialogs, output_path)


if __name__ == "__main__":
    main()

