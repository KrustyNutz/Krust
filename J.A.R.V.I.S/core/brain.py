from __future__ import annotations

import anthropic
from typing import Generator

from config.settings import MODEL, MAX_TOKENS, EFFORT
from core.memory import Memory
from core.personality import build_system_prompt
from tools.registry import ToolRegistry

# Event types yielded by Brain.chat()
# ("text",      str)                   - streaming text chunk
# ("tool_call", {"name": str, "input": dict}) - tool being invoked
# ("tool_done", {"name": str, "result": str}) - tool result
# ("thinking",  str)                   - thinking block (if display=summarized)
# ("done",      None)                  - stream finished


class Brain:
    def __init__(self, memory: Memory, tools: ToolRegistry):
        self.client = anthropic.Anthropic()
        self.memory = memory
        self.tools = tools
        self.history: list[dict] = []
        self._session_id: str = ""

    def set_session(self, session_id: str):
        self._session_id = session_id

    def chat(self, user_input: str) -> Generator[tuple, None, None]:
        memories = self.memory.recall(user_input, limit=5)
        system_text = build_system_prompt(memories)
        system = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]

        self.history.append({"role": "user", "content": user_input})
        self._trim_history()

        accumulated_text = ""

        while True:
            with self.client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT},
                system=system,
                tools=self.tools.get_definitions(),
                messages=self.history,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            accumulated_text += event.delta.text
                            yield ("text", event.delta.text)
                        elif event.delta.type == "thinking_delta":
                            if event.delta.thinking:
                                yield ("thinking", event.delta.thinking)

                response = stream.get_final_message()

            # Append the full assistant response (all content blocks) to history
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                # Store conversation in memory
                if accumulated_text.strip():
                    self.memory.store(user_input, accumulated_text, self._session_id)
                yield ("done", None)
                break

            # Execute tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    yield ("tool_call", {"name": block.name, "input": block.input})
                    result = self.tools.execute(block.name, block.input)
                    yield ("tool_done", {"name": block.name, "result": result})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            self.history.append({"role": "user", "content": tool_results})

    def reset(self):
        self.history = []
        accumulated_text = ""

    def _trim_history(self):
        from config.settings import MAX_HISTORY_TURNS
        # Keep pairs of (user, assistant) messages — each pair is 2 entries
        max_entries = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_entries:
            # Always keep the latest user message (just appended), trim the middle
            self.history = self.history[-max_entries:]
