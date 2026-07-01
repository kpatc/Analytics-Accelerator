"""Executive Analytics Copilot — Anthropic Claude API with tool use.

The copilot answers business questions by:
1. Sending the question to Claude with analytics tool definitions
2. Executing whichever tools Claude requests against real NovaMart data
3. Feeding the grounded results back to Claude for final synthesis
4. Repeating until Claude produces a final text answer (stop_reason="end_turn")

This ensures every answer is grounded in actual project data — no hallucination.
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
    """Structured response from the Executive Copilot.

    Attributes:
        answer: Markdown-formatted answer ready for display.
        tools_called: Ordered list of tool names invoked during the response.
        thinking_steps: Human-readable log of what the copilot did.
        sources: Data source labels cited in the answer.
    """

    answer: str
    tools_called: list[str] = field(default_factory=list)
    thinking_steps: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


# ── Copilot class ──────────────────────────────────────────────────────────────

class ExecutiveCopilot:
    """AI-powered analytics copilot backed by Claude with tool use.

    Args:
        api_key: Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var
                 (via settings) if not provided.
        model: Claude model ID.  Falls back to settings value if not provided.
    """

    _MAX_ITERATIONS: int = 5  # Guard against infinite tool-use loops

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        from bcgx.config.settings import get_settings

        settings = get_settings()
        self._api_key = api_key or settings.anthropic.api_key or None
        self._model = model or settings.anthropic.model or "claude-sonnet-4-6"
        self._client = None  # Lazy-initialise so import succeeds without a key

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return True if an Anthropic API key is available."""
        return bool(self._api_key)

    def ask(
        self,
        question: str,
        conversation_history: list[dict] | None = None,
    ) -> CopilotResponse:
        """Send a business question to the copilot and get a grounded answer.

        Implements the full Claude tool-use agentic loop:
        - Claude requests tools → we execute them against real data → Claude synthesises.

        Args:
            question: Natural-language business question.
            conversation_history: Prior messages for multi-turn conversation.
                                   Each dict has "role" and "content" keys.

        Returns:
            CopilotResponse with the answer and metadata.
        """
        if not self.is_configured():
            return CopilotResponse(
                answer=(
                    "**AI Copilot Not Configured**\n\n"
                    "The Executive Analytics Copilot requires an Anthropic API key.\n\n"
                    "To enable it:\n"
                    "1. Get an API key from [console.anthropic.com](https://console.anthropic.com)\n"
                    "2. Add `ANTHROPIC_API_KEY=sk-ant-...` to your `.env` file\n"
                    "3. Restart the dashboard\n\n"
                    "The copilot will then answer questions with data grounded in NovaMart's "
                    "actual 36-month transaction history."
                ),
                tools_called=[],
                thinking_steps=["API key not configured — returning setup instructions."],
                sources=[],
            )

        client = self._get_client()

        # Build initial message list
        messages: list[dict] = []
        if conversation_history:
            # Only carry forward user/assistant text messages — not raw tool results
            for msg in conversation_history:
                if msg.get("role") in ("user", "assistant") and isinstance(
                    msg.get("content"), str
                ):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": question})

        tools_called: list[str] = []
        thinking_steps: list[str] = []
        iteration = 0

        while iteration < self._MAX_ITERATIONS:
            iteration += 1
            logger.debug(f"Copilot iteration {iteration}, messages={len(messages)}")

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
                    answer=(
                        f"**API Error**\n\n"
                        f"The copilot encountered an error communicating with Claude: {exc}\n\n"
                        "Please check your API key and try again."
                    ),
                    tools_called=tools_called,
                    thinking_steps=thinking_steps + [f"API error: {exc}"],
                    sources=[],
                )

            stop_reason = response.stop_reason

            if stop_reason == "end_turn":
                # Extract text from the final response
                answer_parts = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text") and block.type == "text"
                ]
                answer = "\n\n".join(answer_parts).strip()
                if not answer:
                    answer = "The copilot did not produce a text response. Please try rephrasing your question."

                sources = list({
                    t.replace("get_", "").replace("_", " ").title()
                    for t in tools_called
                })
                thinking_steps.append(f"Generated final answer after {iteration} iteration(s).")

                logger.info(
                    f"Copilot answered in {iteration} iteration(s), "
                    f"tools={tools_called}"
                )
                return CopilotResponse(
                    answer=answer,
                    tools_called=tools_called,
                    thinking_steps=thinking_steps,
                    sources=sources,
                )

            elif stop_reason == "tool_use":
                # Append the full assistant message (including tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                # Execute every tool_use block and collect results
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

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result_str,
                        }
                    )

                # Feed all results back in a single user message
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason — bail out
                logger.warning(f"Unexpected stop_reason: {stop_reason}")
                return CopilotResponse(
                    answer=(
                        f"The copilot stopped unexpectedly (reason: {stop_reason}). "
                        "Please try again."
                    ),
                    tools_called=tools_called,
                    thinking_steps=thinking_steps,
                    sources=[],
                )

        # Max iterations reached
        logger.warning(f"Copilot hit max iterations ({self._MAX_ITERATIONS})")
        return CopilotResponse(
            answer=(
                "The copilot reached its maximum analysis depth without producing a final answer. "
                "Please try a more specific question."
            ),
            tools_called=tools_called,
            thinking_steps=thinking_steps + ["Max iterations reached."],
            sources=[],
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_client(self):  # type: ignore[return]
        """Lazily initialise the Anthropic client."""
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a named tool and return the JSON-encoded result string.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dict of keyword arguments for the tool.

        Returns:
            JSON string of the tool output (always valid JSON, never raises).
        """
        fn = TOOL_FUNCTIONS.get(tool_name)
        if fn is None:
            result = {"error": f"Unknown tool: {tool_name}"}
        else:
            try:
                if tool_input:
                    result = fn(**tool_input)
                else:
                    result = fn()
            except Exception as exc:
                logger.error(f"Tool {tool_name} raised: {exc}")
                result = {"error": str(exc), "tool": tool_name}

        try:
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.error(f"Failed to serialise tool result: {exc}")
            return json.dumps({"error": "Result serialisation failed", "detail": str(exc)})
