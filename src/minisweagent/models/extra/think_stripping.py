"""Model wrapper for locally-served, OpenAI-compatible reasoning endpoints.

Used when generation runs against a self-hosted SGLang/vLLM server instead of a
hosted provider.
Point litellm at it with an ``openai/<served-model-name>`` model id plus
``OPENAI_API_BASE`` / ``OPENAI_API_KEY`` in the environment; the wiring itself
needs no code. This class exists only for the reasoning-trace problem below.

**Why strip the thinking trace.** DeepSeek-V4-Flash always reasons, and its chat
template pre-fills the opening ``<think>`` tag, so the model's own output starts
*inside* the trace and only ever emits the closing ``</think>``. SGLang's
``--reasoning-parser deepseek-v4`` looks for a matching pair, finds no opening
tag, and hands the whole trace back in ``message.content`` with a dangling
``</think>`` -- ``reasoning_content`` stays empty. mini appends that content to
the conversation and re-sends it every step, so an unstripped trace accumulates
in-context for the entire episode (~35 steps median, and the trace is often
larger than the answer). Stripping restores the behaviour hosted providers give
us for free, and matches DeepSeek's own guidance that prior-turn reasoning must
not be fed back.

The action itself is unaffected either way -- mini reads actions from
``tool_calls``, not from the content -- so this only controls context hygiene.
Models that emit no ``</think>`` are passed through untouched, which makes the
class safe as the config-wide default (OpenRouter runs included).
"""

from __future__ import annotations

import logging

from minisweagent.models.litellm_model import LitellmModel

logger = logging.getLogger("local_model")

_THINK_END = "</think>"


def strip_reasoning(content: str | None) -> str | None:
    """Drop everything up to and including the last ``</think>``.

    Handles both the tagless-open form this server produces (``trace</think>answer``)
    and a well-formed ``<think>trace</think>answer``. Returns the input unchanged
    when there is no closing tag.
    """
    if not isinstance(content, str) or _THINK_END not in content:
        return content
    return content.rsplit(_THINK_END, 1)[1].lstrip()


class ThinkStrippingLitellmModel(LitellmModel):
    """LitellmModel that keeps the reasoning trace out of the message history."""

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        message = super().query(messages, **kwargs)
        original = message.get("content")
        stripped = strip_reasoning(original)
        if stripped is not original:
            message["content"] = stripped
            # Keep the trace for post-hoc inspection: `extra` is saved to the
            # trajectory but stripped before every API call (_prepare_messages_for_api).
            message.setdefault("extra", {})["reasoning_content"] = original
            logger.debug("stripped %d chars of reasoning", len(original) - len(stripped or ""))
        return message
