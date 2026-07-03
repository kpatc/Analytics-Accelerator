"""Executive Analytics Copilot — supports Anthropic Claude API and OpenRouter.

The copilot answers business questions by:
1. Sending the question to the LLM with analytics tool definitions
2. Executing whichever tools the model requests against real NovaMart data
3. Feeding the grounded results back for final synthesis
4. Repeating until the model produces a final text answer

Backend priority: OpenRouter (OPENROUTER_KEY) → Anthropic (ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from loguru import logger

from bcgx.copilot.prompts import SYSTEM_PROMPT
from bcgx.copilot.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS


# ── Response schema ────────────────────────────────────────────────────────────

@dataclass
class CopilotResponse:
    answer: str
    tools_called: list[str] = field(default_factory=list)
    thinking_steps: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


# ── Tool definition converter (Anthropic → OpenAI format) ─────────────────────

def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic ToolParam dicts to OpenAI function-calling format."""
    result = []
    for t in anthropic_tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


# ── Copilot class ──────────────────────────────────────────────────────────────

class ExecutiveCopilot:
    """AI-powered analytics copilot backed by Claude (Anthropic or OpenRouter).

    Backend selection:
    - If OPENROUTER_KEY is set → uses OpenRouter (OpenAI-compatible API).
    - Otherwise falls back to ANTHROPIC_API_KEY with the native Anthropic SDK.
    """

    _MAX_ITERATIONS: int = 5

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from bcgx.config.settings import get_settings

        settings = get_settings()

        # Prefer OpenRouter when its key is configured
        or_key = settings.openrouter.api_key
        if or_key:
            self._backend = "openrouter"
            self._api_key = or_key
            self._model = model or settings.openrouter.model
        else:
            self._backend = "anthropic"
            self._api_key = api_key or settings.anthropic.api_key or None
            self._model = model or settings.anthropic.model or "claude-sonnet-4-6"

        self._client = None  # Lazy-initialised

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def ask(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> CopilotResponse:
        if not self.is_configured():
            return CopilotResponse(
                answer=(
                    "**AI Copilot Not Configured**\n\n"
                    "Set `OPENROUTER_KEY` or `ANTHROPIC_API_KEY` in your `.env` file and restart."
                ),
                tools_called=[],
                thinking_steps=["API key not configured — returning setup instructions."],
                sources=[],
            )

        if self._backend == "openrouter":
            return self._ask_openrouter(question, conversation_history)
        return self._ask_anthropic(question, conversation_history)

    # ── OpenRouter backend (OpenAI-compatible) ─────────────────────────────────

    def _ask_openrouter(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> CopilotResponse:
        client = self._get_client()
        tools = _to_openai_tools(TOOL_DEFINITIONS)  # type: ignore[arg-type]

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") in ("user", "assistant") and isinstance(
                    msg.get("content"), str
                ):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        tools_called: list[str] = []
        thinking_steps: list[str] = []

        for iteration in range(1, self._MAX_ITERATIONS + 1):
            logger.debug(f"OpenRouter iteration {iteration}, messages={len(messages)}")
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    max_tokens=4096,
                    tools=tools,
                    messages=messages,
                )
            except Exception as exc:
                logger.error(f"OpenRouter API call failed: {exc}")
                return CopilotResponse(
                    answer=f"**API Error**\n\n{exc}\n\nCheck your OpenRouter key and try again.",
                    tools_called=tools_called,
                    thinking_steps=thinking_steps + [f"API error: {exc}"],
                    sources=[],
                )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message

            if finish_reason == "tool_calls":
                # Append assistant message with tool_calls
                messages.append(message)

                for tc in message.tool_calls or []:
                    tool_name = tc.function.name
                    try:
                        tool_input = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        tool_input = {}

                    thinking_steps.append(
                        f"Calling tool: {tool_name}"
                        + (f" with {tool_input}" if tool_input else "")
                    )
                    logger.debug(f"Executing tool {tool_name}(input={tool_input})")

                    result_str = self._execute_tool(tool_name, tool_input)
                    tools_called.append(tool_name)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_str,
                    })

            else:
                # stop / end_turn → final answer
                answer = (message.content or "").strip()
                if not answer:
                    answer = "The copilot did not produce a text response. Please try rephrasing."

                sources = list({
                    t.replace("get_", "").replace("_", " ").title()
                    for t in tools_called
                })
                thinking_steps.append(f"Generated final answer after {iteration} iteration(s).")
                logger.info(f"OpenRouter answered in {iteration} iteration(s), tools={tools_called}")

                return CopilotResponse(
                    answer=answer,
                    tools_called=tools_called,
                    thinking_steps=thinking_steps,
                    sources=sources,
                )

        logger.warning(f"Copilot hit max iterations ({self._MAX_ITERATIONS})")
        return CopilotResponse(
            answer="The copilot reached its maximum analysis depth. Try a more specific question.",
            tools_called=tools_called,
            thinking_steps=thinking_steps + ["Max iterations reached."],
            sources=[],
        )

    # ── Anthropic backend ──────────────────────────────────────────────────────

    def _ask_anthropic(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> CopilotResponse:
        client = self._get_client()

        messages: list[dict] = []
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") in ("user", "assistant") and isinstance(
                    msg.get("content"), str
                ):
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        tools_called: list[str] = []
        thinking_steps: list[str] = []

        for iteration in range(1, self._MAX_ITERATIONS + 1):
            logger.debug(f"Anthropic iteration {iteration}, messages={len(messages)}")
            try:
                response = client.messages.create(
                    model=self._model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOL_DEFINITIONS,  # type: ignore[arg-type]
                    messages=messages,
                )
            except Exception as exc:
                logger.error(f"Claude API call failed: {exc}")
                return CopilotResponse(
                    answer=f"**API Error**\n\n{exc}\n\nCheck your API key and try again.",
                    tools_called=tools_called,
                    thinking_steps=thinking_steps + [f"API error: {exc}"],
                    sources=[],
                )

            stop_reason = response.stop_reason

            if stop_reason == "end_turn":
                answer_parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text") and block.type == "text"
                ]
                answer = "\n\n".join(answer_parts).strip()
                if not answer:
                    answer = "The copilot did not produce a text response. Please try rephrasing."

                sources = list({
                    t.replace("get_", "").replace("_", " ").title()
                    for t in tools_called
                })
                thinking_steps.append(f"Generated final answer after {iteration} iteration(s).")
                logger.info(f"Copilot answered in {iteration} iteration(s), tools={tools_called}")

                return CopilotResponse(
                    answer=answer,
                    tools_called=tools_called,
                    thinking_steps=thinking_steps,
                    sources=sources,
                )

            elif stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_name: str = block.name
                    tool_input: dict = block.input or {}
                    tool_use_id: str = block.id

                    thinking_steps.append(
                        f"Calling tool: {tool_name}"
                        + (f" with {tool_input}" if tool_input else "")
                    )
                    logger.debug(f"Executing tool {tool_name}(input={tool_input})")

                    result_str = self._execute_tool(tool_name, tool_input)
                    tools_called.append(tool_name)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_str,
                    })

                messages.append({"role": "user", "content": tool_results})

            else:
                logger.warning(f"Unexpected stop_reason: {stop_reason}")
                return CopilotResponse(
                    answer=f"The copilot stopped unexpectedly (reason: {stop_reason}). Please try again.",
                    tools_called=tools_called,
                    thinking_steps=thinking_steps,
                    sources=[],
                )

        logger.warning(f"Copilot hit max iterations ({self._MAX_ITERATIONS})")
        return CopilotResponse(
            answer="The copilot reached its maximum analysis depth. Try a more specific question.",
            tools_called=tools_called,
            thinking_steps=thinking_steps + ["Max iterations reached."],
            sources=[],
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            if self._backend == "openrouter":
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url="https://openrouter.ai/api/v1",
                )
            else:
                from anthropic import Anthropic

                self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        fn = TOOL_FUNCTIONS.get(tool_name)
        if fn is None:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                result = fn(**tool_input) if tool_input else fn()
            except Exception as exc:
                logger.error(f"Tool {tool_name} raised: {exc}")
                result = {"error": str(exc), "tool": tool_name}

        try:
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.error(f"Failed to serialise tool result: {exc}")
            return json.dumps({"error": "Result serialisation failed", "detail": str(exc)})
