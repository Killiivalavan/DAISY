"""Google Gemini client — translates between OpenAI-canonical format
and the Gemini API.
"""

import json
import os
import uuid
from typing import AsyncIterator

from daisy.llm.client import LLMClient, LLMResponse, ToolCall


def _extract_system_instruction(messages: list[dict]) -> str | None:
    parts = [m["content"] for m in messages if m["role"] == "system" and m.get("content")]
    return "\n".join(parts) if parts else None


def _to_gemini_contents_and_tools(
    messages: list[dict],
) -> tuple[list[dict], dict[str, str] | None]:
    """Convert OpenAI messages to Gemini contents.

    Returns (contents, function_name_to_id_map).
    The map is needed because Gemini has no tool-call ID system — it
    matches function responses by name. We generate synthetic IDs so
    the rest of the code sees normal OpenAI-format IDs.
    """
    try:
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError(
            "The 'google-genai' package is required to use Gemini. "
            "Install it with: pip install google-genai"
        )

    contents = []
    name_to_id: dict[str, str] = {}

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "system":
            continue  # handled separately

        if role == "user":
            parts = [genai_types.Part.from_text(text=content or "")]
            contents.append(genai_types.Content(role="user", parts=parts))

        elif role == "assistant":
            parts = []
            if content:
                parts.append(genai_types.Part.from_text(text=content))
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    gid = tc["id"]
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    name_to_id[name] = gid
                    parts.append(
                        genai_types.Part.from_function_call(
                            name=name, args=args
                        )
                    )
            contents.append(genai_types.Content(role="model", parts=parts))

        elif role == "tool":
            fun_name = _find_tool_function_name(messages, msg.get("tool_call_id", ""))
            part = genai_types.Part.from_function_response(
                name=fun_name or "unknown",
                response={"result": content or ""},
            )
            contents.append(genai_types.Content(role="user", parts=[part]))

    return contents, name_to_id


def _find_tool_function_name(messages: list[dict], tool_call_id: str) -> str | None:
    """Walk backwards through messages to find the function name for a tool_call_id."""
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc["id"] == tool_call_id:
                    return tc["function"]["name"]
    return None


def _to_gemini_tools(tools: list[dict]) -> list:
    """Convert OpenAI-format tools to Gemini FunctionDeclarations."""
    try:
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError(
            "The 'google-genai' package is required to use Gemini. "
            "Install it with: pip install google-genai"
        )

    declarations = []
    for t in tools:
        func = t["function"]
        declarations.append(
            genai_types.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=func.get("parameters", {"type": "object", "properties": {}}),
            )
        )
    return [genai_types.Tool(function_declarations=declarations)]


class GeminiClient:
    """LLM client for Google's Gemini API."""

    def __init__(
        self,
        *,
        api_key_env: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required to use Gemini. "
                "Install it with: pip install google-genai"
            )
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _make_config(self, system_instruction: str | None, tools: list | None):
        try:
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "The 'google-genai' package is required to use Gemini. "
                "Install it with: pip install google-genai"
            )

        return genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

    async def complete(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> LLMResponse:
        from google import genai as _  # validate import at call time

        system = _extract_system_instruction(messages)
        gemini_tools = _to_gemini_tools(tools) if tools else None
        config = self._make_config(system, gemini_tools)

        contents, _name_map = _to_gemini_contents_and_tools(messages)

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        return self._parse_response(response)

    def _parse_response(self, response) -> LLMResponse:
        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            return LLMResponse(content=None, finish_reason="stop")
        if candidate.content.parts is None:
            return LLMResponse(content=None, finish_reason="stop")

        content = None
        tool_calls = []
        for part in candidate.content.parts:
            if part.text:
                content = (content or "") + part.text
            if part.function_call:
                gid = f"call_{uuid.uuid4().hex[:12]}"
                tool_calls.append(
                    ToolCall(
                        id=gid,
                        name=part.function_call.name,
                        arguments=json.dumps(part.function_call.args),
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
        )

    async def stream_tokens(self, messages: list[dict]) -> AsyncIterator[str]:
        from google import genai as _

        system = _extract_system_instruction(messages)
        config = self._make_config(system, None)

        contents, _name_map = _to_gemini_contents_and_tools(messages)

        async for chunk in await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            if chunk.candidates and chunk.candidates[0].content:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        yield part.text
