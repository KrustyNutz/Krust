import json
import traceback
from typing import Callable, Any


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str, description: str, input_schema: dict, handler: Callable):
        self._tools[name] = {
            "name": name,
            "description": description,
            "input_schema": input_schema,
        }
        self._handlers[name] = handler

    def get_definitions(self) -> list[dict]:
        return list(self._tools.values())

    def execute(self, name: str, inputs: dict) -> str:
        if name not in self._handlers:
            return f"Error: unknown tool '{name}'"
        try:
            result = self._handlers[name](**inputs)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as e:
            return f"Tool error: {e}\n{traceback.format_exc()}"

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


def build_registry(memory=None) -> ToolRegistry:
    from tools.web_search import search_web
    from tools.file_ops import read_file, write_file, list_directory, get_file_info
    from tools.code_runner import run_python
    from tools.notifications import send_notification
    import datetime, os, platform

    registry = ToolRegistry()

    # ── Web search ──────────────────────────────────────────────────
    registry.register(
        "web_search",
        "Search the web for current information. Returns titles, snippets, and URLs.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "num_results": {"type": "integer", "description": "Number of results (1-10)", "default": 5},
            },
            "required": ["query"],
        },
        search_web,
    )

    # ── File operations ──────────────────────────────────────────────
    registry.register(
        "read_file",
        "Read the contents of a file.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "max_chars": {"type": "integer", "description": "Max characters to read", "default": 20000},
            },
            "required": ["path"],
        },
        read_file,
    )

    registry.register(
        "write_file",
        "Write content to a file. Creates the file and parent directories if needed.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"},
                "append": {"type": "boolean", "description": "Append instead of overwrite", "default": False},
            },
            "required": ["path", "content"],
        },
        write_file,
    )

    registry.register(
        "list_directory",
        "List files and subdirectories in a directory.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: current dir)", "default": "."},
                "show_hidden": {"type": "boolean", "description": "Include hidden files", "default": False},
            },
        },
        list_directory,
    )

    # ── Code execution ───────────────────────────────────────────────
    registry.register(
        "run_python",
        "Execute Python code and return stdout/stderr. Runs in a sandboxed namespace.",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
            },
            "required": ["code"],
        },
        run_python,
    )

    # ── System info ──────────────────────────────────────────────────
    def get_system_info() -> dict:
        return {
            "datetime": datetime.datetime.now().isoformat(),
            "os": platform.system(),
            "hostname": platform.node(),
            "cwd": os.getcwd(),
        }

    registry.register(
        "get_system_info",
        "Get current date/time, OS, hostname, and working directory.",
        {"type": "object", "properties": {}},
        get_system_info,
    )

    # ── Notifications ────────────────────────────────────────────────
    registry.register(
        "notify",
        "Send a desktop notification to get the user's attention.",
        {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Notification title"},
                "message": {"type": "string", "description": "Notification body"},
                "urgency": {"type": "string", "enum": ["low", "normal", "critical"], "default": "normal"},
            },
            "required": ["title", "message"],
        },
        send_notification,
    )

    # ── Memory tools ─────────────────────────────────────────────────
    if memory:
        def remember(content: str, category: str = "general",
                     importance: int = 5, tags: list | None = None) -> str:
            mid = memory.remember(content, category, importance, tags)
            return f"Stored in memory (id={mid}): {content}"

        def recall(query: str, limit: int = 5) -> list:
            results = memory.recall(query, limit)
            return results if results else ["No relevant memories found."]

        def list_memories(category: str = "") -> list:
            return memory.list_memories(category)

        registry.register(
            "remember",
            "Store something important in persistent memory for future recall.",
            {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "What to remember"},
                    "category": {"type": "string", "description": "Category (project, goal, preference, fact, etc.)", "default": "general"},
                    "importance": {"type": "integer", "description": "Importance 1-10", "default": 5},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for search"},
                },
                "required": ["content"],
            },
            remember,
        )

        registry.register(
            "recall",
            "Search memory for relevant context. Use before answering questions about past conversations or stored info.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "limit": {"type": "integer", "description": "Max results", "default": 5},
                },
                "required": ["query"],
            },
            recall,
        )

        registry.register(
            "list_memories",
            "List all stored memories, optionally filtered by category.",
            {
                "type": "object",
                "properties": {
                    "category": {"type": "string", "description": "Filter by category (leave empty for all)", "default": ""},
                },
            },
            list_memories,
        )

    return registry
