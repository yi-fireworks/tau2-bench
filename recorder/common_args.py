"""Shared argument parsing and configuration for tau2 recording tools."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RecorderArgs:
    """Shared arguments for recording and retry operations."""
    domains: list[str]
    num_trials: int
    model: str
    temperature: float
    user_model: str
    user_temperature: float
    outdir: Path
    seed: Optional[int]
    debug: bool
    run_id: Optional[str]
    llm_retries: Optional[int]
    reasoning_effort_agent: Optional[str] = None  # Deprecated; use budget_or_max_tokens_agent
    reasoning_effort_user: Optional[str] = None  # Deprecated; use budget_or_max_tokens_user
    budget_or_max_tokens_agent: Optional[int] = None
    budget_or_max_tokens_user: Optional[int] = None
    max_steps: int = 60
    max_workers: int = 1
    infra_retries: int = 3


def add_model_args(parser: argparse.ArgumentParser, allow_user_override: bool = True):
    """Add model configuration arguments to parser.
    
    Args:
        parser: ArgumentParser to add arguments to
        allow_user_override: If True, adds --user-model and --user-temperature args
    """
    parser.add_argument("--model", type=str, default="gpt-4.1", 
                       help="Model for the agent")
    parser.add_argument("--temperature", type=float, default=0.2, 
                       help="Temperature for the agent")
    
    if allow_user_override:
        parser.add_argument("--user-model", type=str, default=None,
                           help="Model for the user simulator (default: gpt-4.1, or from manifest for retries)")
        parser.add_argument("--user-temperature", type=float, default=None,
                           help="Temperature for the user simulator (default: 0.0, or from manifest for retries)")


def add_execution_args(parser: argparse.ArgumentParser):
    """Add execution configuration arguments to parser."""
    g = parser.add_argument_group("Execution arguments")
    g.add_argument("--reasoning-effort-agent", type=str, default=None, help="Reasoning effort for closed-source models (e.g., gpt-5, claude-4.5). Ignored for open models.")
    g.add_argument("--reasoning-effort-user", type=str, default=None, help="Reasoning effort for closed-source user simulator models. Ignored for open models.")
    g.add_argument("--llm-retries", type=int, default=None, help="Max retries for LLM calls")
    g.add_argument("--max-steps", type=int, default=60, help="Max steps per simulation")
    g.add_argument("--max-workers", type=int, default=1, help="Maximum number of parallel workers")
    g.add_argument("--budget-agent", type=int, default=None, help="Max tokens for the final answer (closed models) or total output (open models). If not set, provider defaults are used.")
    g.add_argument("--budget-user", type=int, default=None, help="Max tokens for user simulator. If not set, provider defaults are used.")
    g.add_argument("--infra-retries", type=int, default=3, help="Max retries for infrastructure failures (e.g., API errors, network issues)")


def merge_args_with_manifest(
    manifest: dict,
    user_model_override: Optional[str] = None,
    user_temperature_override: Optional[float] = None,
    max_workers: Optional[int] = None,
    infra_retries: Optional[int] = None,
) -> RecorderArgs:
    """Create RecorderArgs by merging manifest data with CLI overrides.
    
    Used by retry_failures to reconstruct configuration from manifest while
    allowing selective overrides (especially for user model).
    
    Args:
        manifest: Run manifest dict loaded from run_manifest.json
        user_model_override: Override user model (if None, uses manifest value)
        user_temperature_override: Override user temperature (if None, uses manifest value)
        max_workers: Override max_workers (if None, uses manifest value)
        infra_retries: Override infra_retries (if None, uses manifest value)
    
    Returns:
        RecorderArgs with merged configuration
    """
    return RecorderArgs(
        domains=manifest["domains"],
        num_trials=manifest["num_trials"],
        model=manifest["model"],
        temperature=manifest["temperature"],
        user_model=user_model_override or manifest.get("user_model", "gpt-4.1"),
        user_temperature=(
            user_temperature_override 
            if user_temperature_override is not None 
            else manifest.get("user_temperature", 0.0)
        ),
        outdir=Path("./recordings"),  # Will be overridden by caller
        seed=manifest.get("seed_base"),
        debug=False,
        run_id=manifest["run_id"],
        llm_retries=None,
        reasoning_effort_agent=manifest.get("reasoning_effort_agent", "low"),
        reasoning_effort_user=manifest.get("reasoning_effort_user", "low"),
        max_steps=manifest.get("max_steps", 60),
        max_workers=max_workers or manifest.get("max_workers", 6),
        infra_retries=infra_retries or manifest.get("infra_retries", 2),
    )

