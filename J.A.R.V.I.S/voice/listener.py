"""Speech-to-text using faster-whisper. Install: pip install faster-whisper sounddevice soundfile numpy"""
from __future__ import annotations

import threading
import queue
import time
from config.settings import VOICE_MODEL, WAKE_WORD


class Listener:
    def __init__(self, on_speech=None):
        self.on_speech = on_speech
        self._running = False
        self._q: queue.Queue = queue.Queue()
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(VOICE_MODEL, device="cpu", compute_type="int8")
        except ImportError:
            self._model = None

    def is_available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_path: str) -> str:
        if not self._model:
            return ""
        segments, _ = self._model.transcribe(audio_path, beam_size=5)
        return " ".join(s.text for s in segments).strip()

    def listen_once(self, duration: float = 5.0, sample_rate: int = 16000) -> str:
        """Record audio and return transcription."""
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            import tempfile, os

            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            sf.write(tmp, audio, sample_rate)
            text = self.transcribe(tmp)
            os.unlink(tmp)
            return text
        except Exception as e:
            return f"Listen error: {e}"

    def start_wake_word_loop(self):
        """Continuously listen for wake word in a background thread."""
        self._running = True
        t = threading.Thread(target=self._wake_loop, daemon=True)
        t.start()

    def _wake_loop(self):
        while self._running:
            text = self.listen_once(duration=3.0)
            if WAKE_WORD.lower() in text.lower():
                full_cmd = self.listen_once(duration=7.0)
                if self.on_speech and full_cmd:
                    self.on_speech(full_cmd)
            time.sleep(0.1)

    def stop(self):
        self._running = False
