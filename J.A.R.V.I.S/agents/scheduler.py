"""Background task scheduler for proactive JARVIS behaviors."""
from __future__ import annotations

import threading
import time
from typing import Callable


class Scheduler:
    def __init__(self):
        self._jobs: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def every(self, interval_seconds: int, fn: Callable, name: str = ""):
        self._jobs.append({
            "interval": interval_seconds,
            "fn": fn,
            "name": name or fn.__name__,
            "last_run": 0.0,
        })

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            now = time.time()
            for job in self._jobs:
                if now - job["last_run"] >= job["interval"]:
                    try:
                        job["fn"]()
                    except Exception as e:
                        pass
                    job["last_run"] = now
            time.sleep(1)


def build_scheduler(brain, memory) -> Scheduler:
    scheduler = Scheduler()

    def morning_briefing():
        """Auto-run at startup if it's morning."""
        import datetime
        h = datetime.datetime.now().hour
        if 6 <= h <= 9:
            parts = []
            for t, p in brain.chat(
                "Give me a brief morning status: what should I focus on today based on memory, "
                "any pending tasks, and current market conditions if available."
            ):
                if t == "text":
                    parts.append(p)

    # Check for proactive tasks every 30 minutes
    scheduler.every(1800, morning_briefing, "morning_briefing")

    return scheduler
