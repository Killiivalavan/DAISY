"""LLM client protocol and shared dataclasses.

All providers implement LLMClient. The rest of the system only depends on
this protocol — never on a specific provider SDK.
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol


@dataclass
class ToolCall:
    """Normalised tool call returned by any provider's complete().

    Stored in OpenAI-canonical format: arguments is a JSON string.
    Providers that return parsed dicts (Anthropic, Gemini) are serialized
    by their respective clients.
    """

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMResponse:
    """Normalised response from a non-streaming complete() call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class LLMClient(Protocol):
    """Interface every LLM provider must satisfy.

    The canonical message format is OpenAI Chat Completions:
        {"role": "system"|"user"|"assistant"|"tool", "content": ...}
    with tool schemas in the OpenAI ``tools`` array format.

    Non-OpenAI providers translate to/from this format internally.
    """

    async def complete(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        """Send messages and optionally tools, returning a single response.

        Used for the tool-calling loop where we need to inspect tool_calls
        before continuing.
        """
        ...

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream content-only text tokens from the model.

        Used for the final spoken response, announcements, summarization,
        and sub-agent work.  No tool-calling support at this level.
        """
        ...
