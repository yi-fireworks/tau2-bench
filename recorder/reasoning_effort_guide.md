# LLM Reasoning Control Guide

A guide to controlling reasoning effort when using reasoning-capable models through LiteLLM in the Tau2 framework.

## Overview

Our reasoning effort system provides a unified interface for controlling how deeply models "think" before responding. The implementation is in `tau2/utils/reasoning_effort.py` with a test harness in `recorder/view_reasoning_effort_direct.py`.

**Key principle**: Closed models (GPT, Gemini, Claude) separate reasoning from output budgets, while open models use a single combined budget.

## Architecture

The system uses a LiteLLM pre-API callback (`reasoning_effort_pre_api_callback`) that intercepts completion calls and translates our parameters into provider-specific API formats:

```python
# In tau2/utils/llm_utils.py
litellm.pre_api_callback = [reasoning_effort_pre_api_callback]
litellm.drop_params = True  # Silently drop unknown params
```

## Parameter Reference

### `reasoning_effort`
Controls thinking depth for reasoning-capable models.
- **Values**: `"low"`, `"medium"`, `"high"`
- **Default**: `"medium"`
- **Behavior**: Provider-specific (see below)

### `budget_or_max_tokens`
Overrides `max_tokens` for all models. Use this to control output length.
- **Value**: Integer token count
- **Applied to**: All providers

## Provider-Specific Behavior

### GPT Models (OpenAI)
**Models**: `gpt-5`, `gpt-o*`, `gpt-4*` (when not via Fireworks)

**Implementation**:
```python
kwargs["reasoning_effort"] = reasoning_effort  # Passed directly to OpenAI API
```

**Behavior**:
- `reasoning_effort` is passed directly as a parameter
- Reasoning tokens are tracked separately from completion tokens
- `max_tokens` controls final answer length only

**Status**: ✅ Working

---

### Gemini Models (Google)
**Models**: Models containing `"gemini"` in name

**Implementation**:
```python
budget_map = {
    "low": 2048,
    "medium": 8192,
    "high": -1,  # Dynamic budget
}
kwargs["thinkingConfig"] = {"thinkingBudget": budget}
kwargs["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}
```

**Behavior**:
- Maps effort levels to `thinkingBudget` parameter
- `-1` for "high" enables dynamic budget
- `max_tokens` controls final answer length only
- Both top-level and nested `thinkingConfig` set for API version compatibility

**Status**: ✅ Working

---

### Anthropic Models (Claude)
**Models**: Models containing `"sonnet"`, `"opus"`, or `"haiku"` in name

**Implementation**:
```python
# Currently disabled - see reasoning_effort.py lines 34-36
logger.debug(f"Anthropic model {model_name}: thinking is disabled.")
```

**Behavior**:
- **Currently disabled** to avoid multi-turn conversation issues
- The Anthropic API requires passing `thinking` tokens from previous turns in multi-turn conversations
- This feature is not yet implemented in our conversation state management

**Technical Note**: To enable Anthropic thinking in the future, we would need to:
1. Track thinking tokens from each conversation turn
2. Pass them in subsequent API calls using the extended-thinking feature
3. Ensure `max_tokens > budget_tokens` (validation exists at lines 68-79)

**Status**: ⚠️ Disabled

---

### Open Models (Fireworks)
**Models**: DeepSeek, Kimi, Qwen3, GLM, and other open reasoning models

**Models tested/supported**:
- `fireworks_ai/accounts/fireworks/models/deepseek-v3p1-terminus`
- `fireworks_ai/accounts/fireworks/models/glm-4p5`
- `fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905`
- `fireworks_ai/accounts/fireworks/models/qwen3-235b-a22b`
- `fireworks_ai/accounts/fireworks/models/qwen3-30b-a3b`

**Implementation**:
```python
# reasoning_effort is ignored (not a parameter these models support)
if reasoning_effort != "medium":
    logger.warning(f"Open model {model_name}: 'reasoning_effort' is set to '{reasoning_effort}' but is ignored.")
```

**Behavior**:
- No separate reasoning effort parameter
- `max_tokens` controls **TOTAL** output (reasoning + answer combined)
- Models generate reasoning in `<think>` tags followed by the answer
- If `max_tokens` is exhausted, response truncates immediately (mid-reasoning or mid-answer)

**Critical difference**: Unlike closed models, the reasoning and answer share the same token budget.

**Recommendation**: Set `max_tokens` to 32,000-64,000 for reasoning tasks to avoid truncation.

**Status**: ✅ Working (reasoning_effort ignored as expected)

---

## The Critical Distinction

### Closed Models (GPT, Gemini, Claude)
- **Separate budgets**: Reasoning depth vs. output length
- **`reasoning_effort`**: Controls thinking (separate from output)
- **`max_tokens`**: Controls final answer length ONLY
- **Risk**: Provider defaults for `max_tokens` may be too low; always set explicitly

### Open Models (Fireworks)
- **Single budget**: `max_tokens` covers reasoning + answer together
- **No `reasoning_effort`**: Parameter is ignored
- **Risk**: Low `max_tokens` will truncate output mid-stream
- **Solution**: Set `max_tokens` much higher (3-4x your expected output)

## Usage in Tau2

When calling `generate()` in `tau2/utils/llm_utils.py`:

```python
from tau2.utils.llm_utils import generate

# Closed model with reasoning control
message = generate(
    model="gpt-5",
    messages=conversation_history,
    reasoning_effort="high",        # Controls thinking depth
    budget_or_max_tokens=8192,      # Controls answer length
    **other_kwargs
)

# Open model (reasoning_effort ignored)
message = generate(
    model="fireworks_ai/accounts/fireworks/models/kimi-k2-instruct-0905",
    messages=conversation_history,
    reasoning_effort="medium",      # Ignored, but won't cause errors
    budget_or_max_tokens=32000,     # Total budget for reasoning + answer
    **other_kwargs
)
```

## Testing

Use `recorder/view_reasoning_effort_direct.py` to test reasoning configurations:

```bash
# Dry run (shows request payload without sending)
python recorder/view_reasoning_effort_direct.py

# Live API test
python recorder/view_reasoning_effort_direct.py --live
# or
REASONING_LIVE=1 python recorder/view_reasoning_effort_direct.py
```

Edit the `tests` array in the script to configure which models and effort levels to test.

## Budget Validation (Anthropic-specific)

When Anthropic thinking is enabled, the callback validates that `max_tokens > thinking_budget`:

```python
# From reasoning_effort.py lines 68-79
if kwargs.get("max_tokens", 0) <= thinking_budget:
    new_max_tokens = thinking_budget + 4096
    logger.warning(f"Anthropic model: Setting max_tokens to {new_max_tokens}")
    kwargs["max_tokens"] = new_max_tokens
```

This ensures the model has room for both thinking and output.

## Constants

```python
# Anthropic thinking budget mapping (when enabled)
ANTHROPIC_BUDGET_LOW = 2048
ANTHROPIC_BUDGET_MEDIUM = 8192
ANTHROPIC_BUDGET_HIGH = 32768

# Gemini thinking budget mapping
GEMINI_BUDGET_LOW = 2048
GEMINI_BUDGET_MEDIUM = 8192
GEMINI_BUDGET_HIGH = -1  # Dynamic budget
```

## Best Practices

1. **Always set `budget_or_max_tokens` explicitly** - Don't rely on provider defaults
2. **For open models**: Set `max_tokens` to 32K+ for complex reasoning tasks
3. **For closed models**: Set both `reasoning_effort` AND `budget_or_max_tokens`
4. **Monitor usage**: Track token consumption to optimize your settings
5. **Use the test script**: Verify behavior before running production workloads

## Cost Implications

- **Closed models**: Reasoning and output tokens billed separately (check provider pricing)
- **Open models**: All tokens (reasoning + output) billed together
- Higher reasoning effort = higher cost but potentially better answers
- Open models with high `max_tokens` may consume budget even if unused tokens are not billed

## Troubleshooting

**Response truncated mid-answer (closed models)**:
- Increase `budget_or_max_tokens` - you hit the output limit

**Response truncated mid-reasoning (open models)**:
- Increase `max_tokens` significantly (try doubling it)

**Anthropic multi-turn issues**:
- Thinking is currently disabled for Anthropic models
- For multi-turn support, use models without extended thinking or wait for implementation

**Parameter not working**:
- Check model detection logic in `reasoning_effort.py` lines 29-31
- Verify model name matches expected patterns
- Review logs for warnings about ignored parameters
