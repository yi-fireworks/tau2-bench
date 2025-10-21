import os
from pathlib import Path

# --- Configuration ---
MODELS = [
    "gpt-5",
    "gemini-2.5-pro",
    "claude-sonnet-4-5-20250929",
    "gpt-5-mini",
    "fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b",
    "fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
    "fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus",
    "fireworks_ai/accounts/fireworks/models/glm-4p5",
]

OUTPUT_DIR = Path("recorder/run_scripts")
RECORDER_SCRIPT = "./recorder/run_and_record.py"
BASE_OUTDIR = "recordings/airline-gemini-2.5-pro-user"
DOMAIN = "airline"
TEMPERATURE = 1.0
NUM_TRIALS = 4
USER_MODEL = "gemini-2.5-pro"
MAX_WORKERS = 3
BUDGET = 16384

# --- Script Template ---
SCRIPT_TEMPLATE = """#!/bin/bash
# To run this script in a new tmux session, use the following command:
# tmux new-session -d -s {tmux_session_name} 'bash {script_path}'

# --- Variables ---
export MODEL="{model}"
export DOMAIN="{domain}"
export NOW=$(date -u +%Y%m%d-%H%M%S)
export TEMPERATURE={temperature} # Agent temperature
export NUM_TRIALS={num_trials}
export BASE_MODEL=$(basename "$MODEL")
export RUN_ID="${{NOW}}_${{DOMAIN}}_${{BASE_MODEL}}_temp${{TEMPERATURE}}_tr${{NUM_TRIALS}}"
export OUTDIR="{base_outdir}"
export LOG_DIR="$OUTDIR/logs"

# --- Execution ---
mkdir -p "$LOG_DIR"
echo "Activating virtual environment..."
source ~/venv/bin/activate

echo "Starting run for model: $MODEL"
echo "RUN_ID: $RUN_ID"
echo "Output Directory: $OUTDIR"
echo "Log file: $LOG_DIR/$RUN_ID.log"

python {recorder_script} \\
    --domains "$DOMAIN" \\
    --num-trials "$NUM_TRIALS" \\
    --model "$MODEL" \\
    --temperature "$TEMPERATURE" \\
    --user-model "{user_model}" \\
    --user-temperature 0.0 \\
{reasoning_effort_arg}    --budget-agent {budget} \\
    --budget-user {budget} \\
    --outdir "$OUTDIR" \\
    --run-id "$RUN_ID" \\
    --max-workers {max_workers} > "$LOG_DIR/$RUN_ID.log" 2>&1

echo "Run for model $MODEL completed."
"""

def generate_scripts():
    """Generates a shell script for each model."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    tmux_commands = []
    
    for model in MODELS:
        # Sanitize model name for filenames and tmux sessions
        if "/" in model:
            base_model_name = model.split("/")[-1]
        else:
            base_model_name = model
        
        script_name = f"run_{base_model_name}.sh"
        script_path = OUTPUT_DIR / script_name
        
        # Conditionally add reasoning effort for closed models
        reasoning_effort_arg = ""
        closed_model_keywords = ["gpt", "gemini", "claude"]
        is_closed_model = any(keyword in model.lower() for keyword in closed_model_keywords)
        if is_closed_model:
            reasoning_effort_arg = "    --reasoning-effort-agent medium \\\n"
        
        tmux_session_name = f"tau2-gen-{base_model_name}"
        script_content = SCRIPT_TEMPLATE.format(
            tmux_session_name=tmux_session_name,
            script_path=script_path,
            model=model,
            domain=DOMAIN,
            temperature=TEMPERATURE,
            num_trials=NUM_TRIALS,
            base_outdir=BASE_OUTDIR,
            recorder_script=RECORDER_SCRIPT,
            user_model=USER_MODEL,
            max_workers=MAX_WORKERS,
            budget=BUDGET,
            reasoning_effort_arg=reasoning_effort_arg,
        )
        
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        print(f"Generated script: {script_path}")
        
        tmux_command = f"tmux new-session -d -s {tmux_session_name} 'bash {script_path}'"
        tmux_commands.append(tmux_command)

    all_tmux_script_path = OUTPUT_DIR / "run_all_tmux.sh"
    with open(all_tmux_script_path, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# This script starts a tmux session for each model run.\n\n")
        f.write("\n".join(tmux_commands))
        f.write("\n")
    
    os.chmod(all_tmux_script_path, 0o755)
    print(f"\nGenerated script to run all tmux sessions: {all_tmux_script_path}")

if __name__ == "__main__":
    # Change directory to the project root (tau2-bench)
    # This ensures that the script can be run from any directory
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory set to: {os.getcwd()}")
    generate_scripts()
