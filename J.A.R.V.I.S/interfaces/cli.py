"""Rich terminal interface for J.A.R.V.I.S."""
from __future__ import annotations

import sys
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich.live import Live
from rich.spinner import Spinner
from rich.columns import Columns
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

from config.settings import USER_NAME, DATA_DIR

JARVIS_THEME = Theme({
    "jarvis": "bold cyan",
    "user": "bold white",
    "tool": "dim yellow",
    "memory": "dim green",
    "error": "bold red",
    "dim_text": "dim white",
    "status": "bold blue",
    "thinking": "dim magenta italic",
})

BANNER = """
[bold cyan] ╔═══════════════════════════════════════════════════════╗
 ║   J . A . R . V . I . S                             ║
 ║   Just A Rather Very Intelligent System              ║
 ║   ──────────────────────────────────────────────     ║
 ║   Type your message. Commands: /help /reset /exit    ║
 ╚═══════════════════════════════════════════════════════╝[/bold cyan]
"""

COMMANDS = {
    "/help":    "Show this help",
    "/reset":   "Clear conversation history",
    "/memory":  "Show stored memories",
    "/stats":   "Memory & session stats",
    "/voice":   "Toggle voice mode (if configured)",
    "/screen":  "Analyze current screen",
    "/nexus":   "Check NEXUS trading status",
    "/exit":    "Quit J.A.R.V.I.S",
}


class CLI:
    def __init__(self, brain, memory, voice_pipeline=None):
        self.brain = brain
        self.memory = memory
        self.voice = voice_pipeline
        self.console = Console(theme=JARVIS_THEME)
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.brain.set_session(self._session_start)

        history_file = DATA_DIR / ".jarvis_history"
        self._prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            style=Style.from_dict({"prompt": "bold cyan"}),
        )

    def run(self):
        self.console.print(BANNER)
        stats = self.memory.stats()
        self.console.print(
            f"[dim_text]  {stats['conversations']} conversations · "
            f"{stats['explicit_memories']} memories · "
            f"Session {self._session_start}[/dim_text]\n"
        )

        while True:
            try:
                user_input = self._prompt_session.prompt(
                    f"\n[{USER_NAME}] → ",
                ).strip()
            except (KeyboardInterrupt, EOFError):
                self._shutdown()
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                if not self._handle_command(user_input):
                    return
                continue

            self._respond(user_input)

    def _respond(self, user_input: str):
        self.console.print()
        response_parts: list[str] = []
        tool_calls: list[str] = []

        # Stream the response
        with self.console.status("[status]thinking…[/status]", spinner="dots") as status:
            started_text = False
            for event_type, payload in self.brain.chat(user_input):

                if event_type == "text":
                    if not started_text:
                        status.stop()
                        self.console.print("[jarvis]J.A.R.V.I.S[/jarvis]", end=" ")
                        started_text = True
                    self.console.print(payload, end="", markup=False)
                    response_parts.append(payload)

                elif event_type == "thinking":
                    pass  # Omit thinking display by default

                elif event_type == "tool_call":
                    name = payload.get("name", "?")
                    inp = payload.get("input", {})
                    summary = _summarize_tool_input(name, inp)
                    status.update(f"[tool]⚙  {name}: {summary}[/tool]")
                    tool_calls.append(name)

                elif event_type == "tool_done":
                    status.update("[status]thinking…[/status]")

                elif event_type == "done":
                    if not started_text:
                        status.stop()

        self.console.print()

        if tool_calls:
            self.console.print(
                f"[tool]  tools used: {', '.join(tool_calls)}[/tool]"
            )

        # Speak response if voice is active
        if self.voice and self.voice.is_ready():
            text = "".join(response_parts)
            if text.strip():
                self.voice.speak(text, block=False)

    def _handle_command(self, cmd: str) -> bool:
        cmd = cmd.lower().strip()

        if cmd == "/exit" or cmd == "/quit":
            self._shutdown()
            return False

        elif cmd == "/help":
            for c, desc in COMMANDS.items():
                self.console.print(f"  [jarvis]{c:<12}[/jarvis] {desc}")

        elif cmd == "/reset":
            self.brain.reset()
            self.console.print("[dim_text]Conversation history cleared.[/dim_text]")

        elif cmd == "/memory":
            mems = self.memory.list_memories()
            if not mems:
                self.console.print("[dim_text]No explicit memories stored.[/dim_text]")
            else:
                for m in mems[:20]:
                    self.console.print(
                        f"  [memory]#{m['id']}[/memory] [{m['category']}] "
                        f"[dim_text]{m['content'][:100]}[/dim_text]"
                    )

        elif cmd == "/stats":
            stats = self.memory.stats()
            self.console.print(
                f"  Conversations: [jarvis]{stats['conversations']}[/jarvis]  "
                f"Explicit memories: [jarvis]{stats['explicit_memories']}[/jarvis]  "
                f"History turns: [jarvis]{len(self.brain.history) // 2}[/jarvis]"
            )

        elif cmd == "/voice":
            if not self.voice or not self.voice.is_ready():
                self.console.print("[error]Voice not available. Install faster-whisper and edge-tts.[/error]")
            else:
                self.console.print("[jarvis]Starting voice mode. Ctrl+C to stop.[/jarvis]")
                self.voice.start_voice_mode()

        elif cmd == "/screen":
            self.console.print("[dim_text]Capturing screen…[/dim_text]")
            self._respond("Analyze my screen and tell me what you observe and anything notable.")

        elif cmd == "/nexus":
            self._respond("Check NEXUS trading engine status and give me a brief summary.")

        else:
            self.console.print(f"[error]Unknown command: {cmd}. Type /help for options.[/error]")

        return True

    def _shutdown(self):
        self.console.print("\n[jarvis]Standing by, sir.[/jarvis]\n")


def _summarize_tool_input(name: str, inp: dict) -> str:
    if name == "web_search":
        return inp.get("query", "")[:60]
    if name == "read_file":
        return inp.get("path", "")
    if name == "write_file":
        return inp.get("path", "")
    if name == "run_python":
        code = inp.get("code", "")
        return code.split("\n")[0][:60]
    if name == "remember":
        return inp.get("content", "")[:60]
    if name == "recall":
        return inp.get("query", "")[:60]
    return str(inp)[:60]
