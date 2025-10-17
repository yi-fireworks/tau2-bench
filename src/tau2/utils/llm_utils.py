import json
import re
from typing import Any, Optional

import litellm
from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")


def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def _apply_provider_reasoning_effort(
    model: str,
    kwargs: dict,
    reasoning_effort: str,
) -> dict:
    """
    Map "low" | "medium" | "high" to provider-specific reasoning parameters
    for current (Oct 2025) model families.
    """
    name = (model or "").lower()

    budget_map = {
        "low": 1000,
        "medium": 5000,
        "high": 10000,
    }

    if reasoning_effort not in budget_map:
        logger.warning(f"Invalid reasoning_effort '{reasoning_effort}'. Defaulting to 'low'.")
        reasoning_effort = "low"

    # Never re-enable thinking if explicitly disabled upstream
    if isinstance(kwargs.get("thinking"), dict) and kwargs["thinking"].get("type") == "disabled":
        logger.debug(f"{model}: thinking explicitly disabled upstream; skipping reasoning mapping")
        return kwargs

    # OpenAI GPT-5 series (current generation). We explicitly do NOT handle o-series anymore.
    if "gpt-5" in name or "gpt-o" in name or "gpt-4" in name:
        kwargs["reasoning_effort"] = reasoning_effort
        logger.debug(f"OpenAI {model}: set reasoning_effort={reasoning_effort}")
    # Anthropic Claude 4.5 only (we do not handle 4.0 and earlier here)
    elif ("sonnet" in name) or "opus" in name or "haiku" in name:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget_map[reasoning_effort],
        }
        logger.debug(f"Anthropic {model}: set thinking with budget_tokens={budget_map[reasoning_effort]}")
    # Google Gemini 2.5 series (thinkingBudget)
    elif "gemini" in name:
        # Map efforts to Gemini thinkingBudget per docs:
        # low -> 512, medium -> 2048, high -> -1 (dynamic)
        gemini_budget_map = {
            "low": 128,
            "medium": 512,
            "high": -1,
        }
        budget = gemini_budget_map.get(reasoning_effort, 0)
        # Include both to maximize compatibility across SDKs/REST
        kwargs["thinkingConfig"] = {"thinkingBudget": budget}
        if "generationConfig" not in kwargs:
            kwargs["generationConfig"] = {}
        kwargs["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}
        logger.debug(f"Google {model}: set thinkingBudget={budget}")
    # DeepSeek models (Reasoner and V3 families): follow GLM token caps; no explicit thinking field required
    elif "deepseek" in name:
        ds_tokens_map = {
            "low": 128,
            "medium": 512,
            "high": 2048,
        }
        kwargs["max_tokens"] = ds_tokens_map.get(reasoning_effort, 128)
        logger.debug(
            f"DeepSeek {model}: set max_tokens={kwargs['max_tokens']} for effort={reasoning_effort}"
        )

    # Qwen series (Qwen3, Qwen-3, Qwen-Plus, etc.)
    elif ("qwen3" in name) or ("qwen-3" in name) or ("qwen-plus" in name) or name.startswith("qwen"):
        # Always enable thinking; map effort to thinking_budget like GLM
        qwen_budget_map = {
            "low": 128,
            "medium": 512,
            "high": 2048,
        }
        # kwargs["thinking"] = {"type": "enabled"}  #  no thinking arg. not dict, not float, not bool, not int.
        kwargs["max_tokens"] = qwen_budget_map.get(reasoning_effort, 128)
        logger.debug(
            f"Qwen {model}: enabled thinking, set max_tokens={kwargs['max_tokens']} for effort={reasoning_effort}"
        )

    # GLM-4.5/4.6 series (includes 'glm-4p6' variants). Map effort to max_tokens and enable thinking.
    elif ("glm-4.6" in name) or ("glm-4.5" in name) or ("glm-4p6" in name) or ("glm-4p5" in name):
        glm_tokens_map = {
            "low": 128,
            "medium": 512,
            "high": 2048,
        }

        kwargs["max_tokens"] = glm_tokens_map.get(reasoning_effort, 128)
        logger.debug(
            f"GLM {model}: enabled thinking, set max_tokens={kwargs['max_tokens']} for effort={reasoning_effort}"
        )

    else:
        # Includes deepseek v3p1 and v3p2 series
        logger.warning(f"Unknown model provider: {model}. Passing reasoning_effort as-is.")
        kwargs["reasoning_effort"] = reasoning_effort

    return kwargs


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    
    # TODO: TEST - Apply provider-specific reasoning_effort handling
    # Extract reasoning_effort if present and convert to provider-specific parameters
    if "reasoning_effort" in kwargs:
        reasoning_effort = kwargs.pop("reasoning_effort")
        kwargs = _apply_provider_reasoning_effort(model, kwargs, reasoning_effort)
    
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"
    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]
    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return message


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage
