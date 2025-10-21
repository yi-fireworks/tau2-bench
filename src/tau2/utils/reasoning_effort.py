from loguru import logger

# Constants for Anthropic thinking budget mapping.
# These are required for direct API calls, as Anthropic's API mandates
# a `budget_tokens` value when `thinking` is enabled.
ANTHROPIC_BUDGET_LOW = 2048
ANTHROPIC_BUDGET_MEDIUM = 8192
ANTHROPIC_BUDGET_HIGH = 32768

# Constants for Gemini thinking budget mapping.
# Corresponds to thinkingBudget parameter. We align low/medium with
# Anthropic's values for consistency, but high remains -1 for dynamic budget.
GEMINI_BUDGET_LOW = ANTHROPIC_BUDGET_LOW
GEMINI_BUDGET_MEDIUM = ANTHROPIC_BUDGET_MEDIUM
GEMINI_BUDGET_HIGH = -1


def reasoning_effort_pre_api_callback(kwargs, model_name, **extra_info):
    """
    LiteLLM pre-API callback to apply reasoning effort and budget parameters.
    This function modifies kwargs in-place.
    """
    budget_override = kwargs.pop("budget_or_max_tokens", None)
    reasoning_effort = kwargs.pop("reasoning_effort", None)
    name = (model_name or "").lower()

    # Step 1: Apply reasoning effort
    if reasoning_effort is not None:
        is_anthropic_model = "sonnet" in name or "opus" in name or "haiku" in name
        is_gemini_model = "gemini" in name
        is_other_closed_model = "gpt-5" in name or "gpt-o" in name or "gpt-4" in name

        if is_anthropic_model:
            # For Anthropic, reasoning_effort is temporarily disabled to avoid multi-turn conversation issues.
            logger.debug(f"Anthropic model {model_name}: thinking is disabled.")
        elif is_gemini_model:
            # For Gemini, we map effort to a thinkingBudget.
            budget_map = {
                "low": GEMINI_BUDGET_LOW,
                "medium": GEMINI_BUDGET_MEDIUM,
                "high": GEMINI_BUDGET_HIGH,
            }
            budget = budget_map.get(reasoning_effort, GEMINI_BUDGET_MEDIUM) # Default medium
            # Both thinkingConfig and generationConfig.thinkingConfig are supported.
            # We set both for maximum compatibility across different API versions.
            kwargs["thinkingConfig"] = {"thinkingBudget": budget}
            if "generationConfig" not in kwargs:
                kwargs["generationConfig"] = {}
            kwargs["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}
            logger.debug(f"Gemini model {model_name}: set thinkingBudget={budget} for effort='{reasoning_effort}'")
        elif is_other_closed_model:
            # For other closed models (GPT), it's a direct parameter.
            kwargs["reasoning_effort"] = reasoning_effort
            logger.debug(f"Closed model {model_name}: set reasoning_effort='{reasoning_effort}'")
        else:
            # For open models, it's ignored.
            if reasoning_effort != "medium":
                logger.warning(f"Open model {model_name}: 'reasoning_effort' is set to '{reasoning_effort}' but is ignored.")
            else:
                logger.debug(f"Open model {model_name}: 'reasoning_effort' is ignored as expected.")

    # Step 2: Apply budget override to max_tokens for all models
    if budget_override is not None:
        kwargs["max_tokens"] = budget_override
        logger.debug(f"Model {model_name}: set max_tokens={budget_override} from budget_or_max_tokens")

    # Step 3: Validate and fix Anthropic params to ensure max_tokens > thinking_budget
    if "sonnet" in name or "opus" in name or "haiku" in name:
        if "thinking" in kwargs and "budget_tokens" in kwargs["thinking"]:
            thinking_budget = kwargs["thinking"]["budget_tokens"]
            
            # If max_tokens is not set, or is <= thinking_budget, we must fix it.
            if kwargs.get("max_tokens", 0) <= thinking_budget:
                new_max_tokens = thinking_budget + 4096  # Add a healthy margin
                logger.warning(
                    f"Anthropic model {model_name}: max_tokens ({kwargs.get('max_tokens', 'Not set')}) is not greater than "
                    f"thinking budget ({thinking_budget}). Setting max_tokens to {new_max_tokens}."
                )
                kwargs["max_tokens"] = new_max_tokens
    
    return kwargs
