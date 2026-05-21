"""Multi-step research agent. Uses JARVIS brain to do deep research loops."""
from __future__ import annotations

from typing import Callable


def deep_research(brain, topic: str, depth: int = 3, on_update: Callable | None = None) -> str:
    """
    Multi-step research loop:
    1. Break topic into sub-questions
    2. Search + gather sources for each
    3. Synthesize into a coherent report
    """
    plan_prompt = (
        f"I need to research: {topic}\n\n"
        f"Break this into {depth} specific sub-questions to investigate, then use web_search "
        f"to gather information on each, and finally synthesize everything into a thorough report. "
        f"Use your tools autonomously — search, read results, and compile. "
        f"Structure the final output as a well-organized report."
    )

    report = ""
    for event_type, payload in brain.chat(plan_prompt):
        if event_type == "text":
            report += payload
            if on_update:
                on_update(payload)
        elif event_type == "tool_call" and on_update:
            on_update(f"\n[Searching: {payload.get('input', {}).get('query', '')}]\n")

    return report


def register_research_tools(registry, brain):
    def research(topic: str, depth: int = 3) -> str:
        parts = []
        for event_type, payload in brain.chat(
            f"Research this topic thoroughly using web_search: {topic}. "
            f"Search multiple angles, synthesize findings into a structured report."
        ):
            if event_type == "text":
                parts.append(payload)
        return "".join(parts)

    registry.register(
        "deep_research",
        "Perform deep multi-step research on a topic using web search and synthesis.",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Research topic or question"},
                "depth": {"type": "integer", "description": "Research depth (1-5)", "default": 3},
            },
            "required": ["topic"],
        },
        research,
    )
