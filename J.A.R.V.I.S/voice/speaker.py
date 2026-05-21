"""Text-to-speech using edge-tts. Install: pip install edge-tts"""
from __future__ import annotations

import asyncio
import threading
import tempfile
import os


VOICE = "en-US-GuyNeural"  # Deep, professional male voice


class Speaker:
    def __init__(self):
        self._available = self._check_available()

    def _check_available(self) -> bool:
        try:
            import edge_tts  # noqa
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str, block: bool = True):
        if not self._available or not text.strip():
            return
        if block:
            asyncio.run(self._speak_async(text))
        else:
            t = threading.Thread(target=asyncio.run, args=(self._speak_async(text),), daemon=True)
            t.start()

    async def _speak_async(self, text: str):
        try:
            import edge_tts
            import subprocess, shutil

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp = f.name

            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(tmp)

            # Play audio
            if shutil.which("mpg123"):
                subprocess.run(["mpg123", "-q", tmp], check=False)
            elif shutil.which("ffplay"):
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp], check=False)
            elif shutil.which("aplay"):
                # Convert mp3 to wav first with ffmpeg
                wav = tmp.replace(".mp3", ".wav")
                subprocess.run(["ffmpeg", "-i", tmp, wav, "-y", "-loglevel", "quiet"], check=False)
                subprocess.run(["aplay", "-q", wav], check=False)
                os.unlink(wav)

            os.unlink(tmp)
        except Exception:
            pass
