"""Estimate weighted usage when the proxy supplies no token counts."""

import asyncio
from pathlib import Path

from harness.backend import JiuwenBackend


TOKEN_BUDGET_REFERENCE = 100_000
LOCAL_USAGE_SOURCE = "local_qwen_estimate"
TOKENIZERS = {
    "qwen3.8-27b": "Qwen3.8-27B",
    "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B",
}


class QwenTokenCounter:
    def __init__(self, model_name):
        from transformers import PreTrainedTokenizerFast

        model = TOKENIZERS[model_name.removesuffix("-noreason")]
        directory = Path(__file__).resolve().parents[1] / "competition" / "tokenizers" / model
        self.thinking_enabled = not model_name.endswith("-noreason")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(directory / "tokenizer.json"),
            chat_template=(directory / "chat_template.jinja").read_text(encoding="utf-8"),
        )

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def input_tokens(self, messages):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return len(self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=self.thinking_enabled,
            tokenize=True, return_dict=False,
        ))

    def completion_usage(self, response, max_tokens):
        reasoning = response.reasoning_content
        completion = response.content
        if reasoning:
            completion = reasoning + "\n</think>\n\n" + completion
        if response.finish_reason == "stop" and not completion.endswith("<|im_end|>"):
            completion += "<|im_end|>"
        estimated = self.count(completion)
        reasoning_unavailable = self.thinking_enabled and not reasoning
        # Hidden reasoning and truncated generations can cost more than the returned text.
        charged = estimated
        if reasoning_unavailable or response.finish_reason != "stop":
            charged = max(estimated, max_tokens)
        return {
            "output_tokens": charged,
            "reasoning_tokens": self.count(reasoning) if reasoning else 0,
            "estimated_output_tokens": estimated,
            "reasoning_unavailable": reasoning_unavailable,
        }


class LocallyCountedBackend(JiuwenBackend):
    usage_source = LOCAL_USAGE_SOURCE

    def __init__(self, model_name, reasoning_accounting, *, call_delay_seconds):
        if reasoning_accounting != "included":
            raise ValueError("local_count_includes_reasoning")
        super().__init__(model_name, reasoning_accounting, call_delay_seconds=call_delay_seconds)
        self.token_counter = QwenTokenCounter(self.model_name)
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0
        self.calls_without_reasoning_text = 0
        self.output_reserve_tokens = 0

    async def invoke(self, messages, max_tokens):
        if self.call_delay_seconds:
            await asyncio.sleep(self.call_delay_seconds)
        input_tokens = self.token_counter.input_tokens(messages)
        # Count input at dispatch even if the request fails or is cancelled.
        self.estimated_input_tokens += input_tokens
        response = await self.model.invoke(messages, max_tokens=max_tokens)
        usage = self.token_counter.completion_usage(response, max_tokens)
        self.estimated_output_tokens += usage["estimated_output_tokens"]
        self.calls_without_reasoning_text += int(usage["reasoning_unavailable"])
        self.output_reserve_tokens += usage["output_tokens"] - usage["estimated_output_tokens"]
        return {"content": response.content, "usage": {
            **usage, "input_tokens": input_tokens, "source": LOCAL_USAGE_SOURCE,
        }}
