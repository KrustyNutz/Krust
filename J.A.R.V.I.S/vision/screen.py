"""Screen capture and visual analysis. Requires: pip install mss pillow"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional


def capture_screen(monitor: int = 0, save_path: Optional[str] = None) -> dict:
    """Capture the screen and return base64 PNG + optional save path."""
    try:
        import mss
        from PIL import Image

        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor >= len(monitors):
                monitor = 0
            shot = sct.grab(monitors[monitor])
            img = Image.frombytes("RGB", shot.size, shot.rgb)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

            if save_path:
                img.save(save_path)

            return {
                "success": True,
                "width": img.width,
                "height": img.height,
                "monitor": monitor,
                "base64_png": b64,
                "saved_to": save_path,
            }
    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {e}. Install mss and pillow."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def build_vision_message(user_text: str, base64_png: str) -> list[dict]:
    """Build a Claude content block list with an image + text."""
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_png,
            },
        },
        {"type": "text", "text": user_text},
    ]


def analyze_screen(brain, question: str = "What do you see on my screen?") -> str:
    """Capture screen and ask JARVIS to analyze it."""
    result = capture_screen()
    if not result["success"]:
        return f"Screen capture failed: {result.get('error')}"

    content = build_vision_message(question, result["base64_png"])
    # Inject as a user message directly into brain history
    brain.history.append({"role": "user", "content": content})

    response_text = ""
    for event_type, payload in brain.chat.__wrapped__(brain, "") if hasattr(brain.chat, '__wrapped__') else []:
        if event_type == "text":
            response_text += payload

    return response_text or "Vision analysis complete."


def register_vision_tools(registry, brain):
    def screenshot_and_describe(question: str = "What do you see on my screen?") -> dict:
        result = capture_screen()
        if not result["success"]:
            return result
        return {
            "captured": True,
            "width": result["width"],
            "height": result["height"],
            "instruction": f"Image captured. Analyze it for: {question}",
            "base64_png": result["base64_png"][:50] + "...",
        }

    registry.register(
        "capture_screen",
        "Capture a screenshot and describe what's on the screen.",
        {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to look for or analyze in the screenshot",
                    "default": "What do you see on my screen?",
                },
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0 = primary)",
                    "default": 0,
                },
            },
        },
        lambda question="What do you see on my screen?", monitor=0: capture_screen(monitor),
    )
