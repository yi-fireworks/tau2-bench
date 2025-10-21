# Tau2 Recording and Training Workflow

This directory contains tools for recording, analyzing, and preparing conversation traces from τ²-bench simulations for SFT/RFT fine-tuning.

## Overview: 5-Step Workflow

```
1. Generate       → 2. Record        → 3. Analyze       → 4. Export        → 5. Train
   execution          traces via          results         to RLOR           with RLOR
   scripts           tmux             with dialogs        format
   (parallel)        (parallel)       & logs
```

---

## Step 1: Generate Execution Scripts

Use `generate_run_scripts.py` to create tmux-compatible shell scripts for multiple models running in parallel.

### Usage

```bash
python recorder/generate_run_scripts.py
```

### Configuration

Edit the script to specify:
- **MODELS**: List of models to test (e.g., gpt-5, gemini-2.5-pro, claude-sonnet)
- **DOMAIN**: Domain to evaluate (e.g., airline)
- **NUM_TRIALS**: Trials per task (default: 4)
- **TEMPERATURE**: Agent temperature (default: 1.0)
- **USER_MODEL**: User simulator model (default: gemini-2.5-pro)
- **MAX_WORKERS**: Parallel task workers per model (default: 3)
- **BUDGET**: Token budget per call (default: 16384)

### Output

Generates shell scripts in `recorder/run_scripts/`:
- `run_<model_name>.sh` for each model
- `run_all_tmux.sh` to launch all in parallel

### Example

```bash
# View what will be generated
cat recorder/generate_run_scripts.py | head -30

# Generate scripts
python recorder/generate_run_scripts.py

# Launch all models in parallel tmux sessions
bash recorder/run_scripts/run_all_tmux.sh

# Or launch a single model
bash recorder/run_scripts/run_gpt-5.sh

# Monitor progress
tmux list-sessions
tmux attach-session -t tau2-gen-gpt-5
```

---

## Step 2: Record Traces

The `run_and_record.py` script records full conversation traces and LLM payloads during simulations.

### Key Features

- **Full transparency**: Captures every LLM call (agent + user simulator) in `tau2_payloads.jsonl`
- **Structured dialogs**: Normalized conversations in OpenAI format with evaluation metrics
- **Parallel execution**: ThreadPoolExecutor for concurrent task evaluation
- **Resilience**: Individual failures don't crash the run; errors are logged separately
- **Separate user model**: Reliable user simulator (e.g., gpt-4.1 @ T=0.0) ensures protocol compliance
- **Rich metrics**: Captures DB success, communication rates, reward breakdowns, termination reasons

### Output Files (per model/run)

```
recordings/run_<timestamp>_<domain>_<model>_<config>/
├── tau2_dialogs.jsonl        # Primary output: dialogs + metrics (one JSON per line)
├── tau2_payloads.jsonl       # Raw LLM API calls (for debugging/transparency)
├── run_manifest.json         # Metadata: model, temperature, seeds, commit hash
├── errors.log                # Warnings and infrastructure failures
└── litellm.yaml              # LiteLLM configuration snapshot
```

### Example Dialog Record (tau2_dialogs.jsonl)

```json
{
  "session_id": "20251021-002721_airline_gpt-5_temp1.0_tr4:airline:0:0",
  "domain": "airline",
  "task_id": "0",
  "messages": [
    {"role": "assistant", "content": "Hi! How can I help you today?", "tool_calls": null},
    {"role": "user", "content": "I want to cancel my reservation...", ...},
    ...
  ],
  "metrics": {
    "success": true,
    "score": 1.0,
    "db_success": true,
    "num_steps": 17,
    "termination_reason": "user_stop",
    "reward_breakdown": {"DB": 1.0, "COMMUNICATE": 1.0},
    "infra_failure": false
  }
}
```

### Configuration Parameters

- `--domains`: Comma-separated domain names (required)
- `--num-trials`: Trials per task (default: 1)
- `--model`: Agent LLM (default: gpt-4.1)
- `--temperature`: Agent temperature (default: 0.2)
- `--user-model`: User simulator LLM (default: gpt-4.1)
- `--user-temperature`: User simulator temperature (default: 0.0)
- `--reasoning-effort-agent`: Reasoning effort (low/medium/high)
- `--budget-agent`: Token budget for agent (default: None)
- `--max-workers`: Parallel workers (default: 6)
- `--llm-retries`: Retries per LLM call (default: 0)
- `--infra-retries`: Retries for infrastructure failures (default: 2)
- `--seed`: Random seed for reproducibility
- `--outdir`: Output directory (default: ./recordings)
- `--run-id`: Custom run identifier
- `--debug`: Enable debug mode

---

## Step 3: Analyze Results

### 3a. Quick Summary with analyze_dialogs.py

Get high-level statistics on your runs:

```bash
# Single run summary
python -m recorder.analyze_dialogs \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/tau2_dialogs.jsonl

# Multiple runs (e.g., all models)
for dialogs in recordings/run_*/tau2_dialogs.jsonl; do
  echo "=== $dialogs ==="
  python -m recorder.analyze_dialogs "$dialogs" | head -30
done

# Export to CSV for Excel/R analysis
python -m recorder.analyze_dialogs \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/tau2_dialogs.jsonl \
  --output metrics.csv

# Machine-readable JSON output
python -m recorder.analyze_dialogs \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/tau2_dialogs.jsonl \
  --json > metrics.json
```

**Statistics Provided:**
- Overall success rate, score distribution
- Per-task and per-trial performance
- Termination reasons (user_stop, transfer, max_steps, etc.)
- DB success rates, communication rates
- Worst-performing tasks for focused analysis
- Task success histograms (e.g., "20 tasks had 4/4 successes")

### 3b. Examine Logs for Issues

```bash
# Check for infrastructure/API errors
grep "Failed to run" recordings/run_*/errors.log | wc -l

# View specific warnings
grep "WARNING" recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/errors.log | head -20

# Check for rate limits or specific errors
grep "list index out of range\|rate_limit\|timeout" recordings/run_*/errors.log

# Count unique error types
grep "Failed to extract\|Error" recordings/run_*/errors.log | cut -d: -f3- | sort | uniq -c
```

### 3c. Deep Dive with Python

See `EVALUATION_SYSTEM.md` for understanding:
- DB evaluation (strict function call matching)
- Communication scoring (substring matching in agent messages)
- NL assertions (LLM judge evaluations)
- Failure modes and what causes them

---

## Step 4: Export for Training

Use `prepare_tau2_data.py` to combine multiple runs and create RLOR-compatible training/test splits.

### Usage

```bash
python recorder/prepare_tau2_data.py \
  --input-dirs \
    recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4 \
    recordings/run_20251021-002740_airline_gemini-2.5-pro_temp1.0_tr4 \
    recordings/run_20251021-002750_airline_gpt-5-mini_temp1.0_tr4 \
    recordings/run_20251021-004406_airline_claude-sonnet-4-5-20250929_temp1.0_tr4 \
  --output-dir /mnt/datasets/tau2-bench/recordings/airlines/airline-user-gemini2.5pro-til20pct \
  --run-name-map \
    run_20251021-002721_airline_gpt-5_temp1.0_tr4:gpt-5 \
    run_20251021-002740_airline_gemini-2.5-pro_temp1.0_tr4:gemini-2.5-pro \
    run_20251021-002750_airline_gpt-5-mini_temp1.0_tr4:gpt-5-mini \
    run_20251021-004406_airline_claude-sonnet-4-5-20250929_temp1.0_tr4:claude-sonnet \
  --test-fraction 0.2
```

### What It Does

1. **Combines multiple runs**: Merges dialogs from all input directories
2. **Cleans data**: Removes infrastructure failures and scoreless records
3. **Creates train/test split**: Splits by **task IDs** (not individual trials)
   - All trials of a task go together (train or test)
   - Ensures no data leakage between splits
4. **Remaps run names**: Maps long directory names to clean identifiers
5. **Generates split manifest**: `split_manifest.json` defines the split (reusable for future runs)

### Output

```
/mnt/datasets/tau2-bench/recordings/airlines/airline-user-gemini2.5pro-til20pct/
├── train.jsonl              # 80% of tasks (all trials, all models)
├── test.jsonl               # 20% of tasks (all trials, all models)
└── split_manifest.json      # Defines train/test task split
```

### Statistics (Example)

```
Total records read: 800 (4 models × 50 tasks × 4 trials)
Records processed: 800 (100% - no failures)
Train split: 640 records (40 tasks × 4 trials × 4 models)
Test split:  160 records (10 tasks × 4 trials × 4 models)
```

### Reusing Splits

To apply the same split to new model runs:

```bash
python recorder/prepare_tau2_data.py \
  --input-dirs recordings/run_new_model_* \
  --output-dir /mnt/datasets/tau2-bench/recordings/airlines/airline-user-newmodel-til20pct \
  --split-manifest /mnt/datasets/tau2-bench/recordings/airlines/airline-user-gemini2.5pro-til20pct/split_manifest.json
```

---

## Step 5: Train Models (Next Steps)

Once you have prepared data, use the RLOR training workflow to fine-tune models.

### Reference

- **Main entry point**: `cookbook-internal/recipes/workflow/rlor/main.py`
- **Example config**: `cookbook-internal/recipes/workflow/rlor/conf/tau2_airline_multiturn.yaml`

### Typical Workflow

```yaml
# In tau2_airline_multiturn.yaml, reference your prepared data:
load_train_data:
  kwargs:
    input_files:
      - /mnt/datasets/tau2-bench/recordings/airlines/airline-user-gemini2.5pro-til20pct/train.jsonl

load_test_data:
  kwargs:
    input_files:
      - /mnt/datasets/tau2-bench/recordings/airlines/airline-user-gemini2.5pro-til20pct/test.jsonl
```

Then run:
```bash
python recipes/workflow/rlor/main.py --config-name tau2_airline_multiturn
```

This will:
1. Load your prepared training data
2. Filter successful examples for SFT (supervised fine-tuning warm-start)
3. Prepare preference pairs for RFT (rejection fine-tuning / RLOR)
4. Fine-tune a base model through both phases
5. Evaluate at each checkpoint (base → SFT → RFT)

---

## Troubleshooting

### High Failure Rate During Recording

**Symptoms:** Many tasks fail with `list index out of range` or rate limit errors

**Solutions:**
1. **Reduce parallelism** to avoid rate limiting:
   ```bash
   --max-workers 2  # or even 1 for very strict limits
   ```

2. **Add retries** for transient failures:
   ```bash
   --llm-retries 3 --infra-retries 5
   ```

3. **Increase token budgets** if hitting limits:
   ```bash
   --budget-agent 32000 --budget-user 32000
   ```

### Analyzing Partial Failures

You can retry infrastructure failures separately without re-running successes:

```bash
python -m recorder.retry_failures \
  recordings/run_20251021-002721_airline_gpt-5_temp1.0_tr4/ \
  --max-retries 3 \
  --max-workers 1
```

### Data Quality Issues

**Inspect problematic records:**
```python
import json

with open("train.jsonl") as f:
    for line in f:
        record = json.loads(line)
        if record["metrics"]["score"] == 0.0:
            print(f"Task {record['task_id']}: Failed")
            print(f"  Reason: {record['metrics'].get('reward_basis')}")
```

**Filter by quality for RFT:**
```python
# Exclude near-misses to focus on clear failures
rft_data = [
    d for d in all_records
    if not d["metrics"].get("infra_failure", False)
    and d["metrics"]["score"] < 0.5  # Only clear failures
]
```

---

## Key Concepts

### Infrastructure Failures vs. Model Failures

Each record includes `metrics.infra_failure`:
- **false**: Task completed (success or legitimate failure)
- **true**: Infrastructure error (API failure, rate limit, timeout)

**Always exclude infrastructure failures from training data** — they don't represent model behavior.

### Reward Structure

See `EVALUATION_SYSTEM.md` for details, but briefly:

- **DB Score**: Deterministic check of database state after agent actions
- **Communication Score**: Semantic check if agent communicated required info
- **NL Assertions**: LLM judge evaluation of complex policies
- **Final Score**: Product of applicable components (both must pass to get 1.0)

### SFT vs. RFT

- **SFT (Supervised Fine-Tuning)**: Train on successful examples only
  ```python
  sft_data = [d for d in dialogs if d["metrics"]["success"] == true]
  ```

- **RFT (Rejection Fine-Tuning)**: Learn from failures alongside successes
  ```python
  rft_data = [d for d in dialogs if not d["metrics"].get("infra_failure")]
  ```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `run_and_record.py` | Record traces via tau2 simulations |
| `generate_run_scripts.py` | Create tmux execution scripts for multiple models |
| `prepare_tau2_data.py` | Prepare data for RLOR training (train/test split) |
| `analyze_dialogs.py` | Compute summary statistics from recorded traces |
| `EVALUATION_SYSTEM.md` | Deep dive into τ²-bench evaluation mechanics |
| `llm_recorder.py` | LiteLLM monkeypatching for payload capture |
| `common_args.py` | Shared argument definitions |

---

**Updated**: October 21, 2025  
**Workflow Version**: 5-step (Generate → Record → Analyze → Export → Train)

