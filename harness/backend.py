import logging
import os
import asyncio


ALLOWED_MODELS = (
    "qwen3.8-27b", "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason",
)
DEFAULT_MODEL = "qwen3.8-27b-noreason"


def normalize_usage(usage, reasoning_accounting="included"):
    if usage is None:
        raise ValueError("missing_usage")
    data = usage.model_dump() if hasattr(usage, "model_dump") else usage
    fields = ("input_tokens", "output_tokens", "reasoning_tokens")
    result = {}
    for name in fields:
        value = data.get(name, 0 if name == "reasoning_tokens" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid_usage")
        result[name] = value
    if reasoning_accounting == "separate":
        result["output_tokens"] += result["reasoning_tokens"]
    elif reasoning_accounting != "included":
        raise ValueError("invalid_reasoning_accounting")
    elif result["reasoning_tokens"] > result["output_tokens"]:
        raise ValueError("inconsistent_reasoning_usage")
    result["reported_output_tokens"] = data["output_tokens"]
    result["cache_tokens"] = data.get("cache_tokens", 0)
    result["reasoning_accounting"] = reasoning_accounting
    return result


class JiuwenBackend:
    def __init__(self, model_name=DEFAULT_MODEL, reasoning_accounting="included", timeout=120,
                 call_delay_seconds=0.0):
        if model_name not in ALLOWED_MODELS:
            raise ValueError("model_not_allowed")
        self.model_name = model_name
        self.reasoning_accounting = reasoning_accounting
        self.call_delay_seconds = call_delay_seconds
        self.request_timeout_seconds = timeout
        # Configure before importing foundation/runner: their defaults create ./logs.
        # Own JSONL journal is the only persistent logger and contains no config.
        from openjiuwen.core.common.logging.log_config import configure_log_config
        configure_log_config({"backend": "default", "level": "CRITICAL",
                              "output": [], "interface_output": [],
                              "performance_output": [], "propagate": False,
                              "log_path": "/tmp/jiuwen-logs"})
        logging.disable(logging.CRITICAL)
        from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
        self.model = Model(
            ModelClientConfig(client_provider="OpenAI", api_key=os.environ["LLM_API_KEY"],
                              api_base=os.environ["LLM_BASE_URL"].rstrip("/"),
                              timeout=timeout, max_retries=0,
                              use_shared_llm_http_client=False),
            ModelRequestConfig(model=model_name, temperature=0),
        )

    async def invoke(self, messages, max_tokens):
        if self.call_delay_seconds:
            await asyncio.sleep(self.call_delay_seconds)
        response = await self.model.invoke(messages, max_tokens=max_tokens)
        return {"content": response.content,
                "usage": normalize_usage(response.usage_metadata, self.reasoning_accounting)}
