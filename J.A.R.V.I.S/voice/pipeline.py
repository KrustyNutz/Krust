"""Full voice conversation pipeline."""
from __future__ import annotations

from voice.listener import Listener
from voice.speaker import Speaker


class VoicePipeline:
    def __init__(self, brain):
        self.brain = brain
        self.listener = Listener()
        self.speaker = Speaker()
        self._active = False

    def is_ready(self) -> bool:
        return self.listener.is_available() and self.speaker.is_available()

    def speak(self, text: str, block: bool = False):
        self.speaker.speak(text, block=block)

    def listen_and_respond(self, duration: float = 7.0) -> str:
        text = self.listener.listen_once(duration=duration)
        if not text:
            return ""
        full_response = ""
        for event_type, payload in self.brain.chat(text):
            if event_type == "text":
                full_response += payload
        self.speaker.speak(full_response, block=False)
        return full_response

    def start_voice_mode(self):
        """Blocking voice loop — listen for wake word, respond, repeat."""
        if not self.is_ready():
            print("Voice not available. Install faster-whisper and edge-tts.")
            return

        def on_speech(text: str):
            full_response = ""
            for event_type, payload in self.brain.chat(text):
                if event_type == "text":
                    full_response += payload
            self.speaker.speak(full_response, block=True)

        self.listener.on_speech = on_speech
        self.listener.start_wake_word_loop()
        print(f"Voice mode active. Say '{__import__('config.settings', fromlist=['WAKE_WORD']).WAKE_WORD}' to activate.")
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.listener.stop()
