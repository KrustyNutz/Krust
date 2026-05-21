import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
MODEL: str = os.getenv("JARVIS_MODEL", "claude-opus-4-7")
MAX_TOKENS: int = int(os.getenv("JARVIS_MAX_TOKENS", "16000"))
EFFORT: str = os.getenv("JARVIS_EFFORT", "high")
USER_NAME: str = os.getenv("JARVIS_USER_NAME", "Sir")

VOICE_ENABLED: bool = os.getenv("JARVIS_VOICE", "false").lower() == "true"
WAKE_WORD: str = os.getenv("JARVIS_WAKE_WORD", "jarvis")
VOICE_MODEL: str = os.getenv("JARVIS_VOICE_MODEL", "whisper-base")

MEMORY_DB: Path = DATA_DIR / "memory.db"
MAX_HISTORY_TURNS: int = 30

HOME_ASSISTANT_URL: str = os.getenv("HOME_ASSISTANT_URL", "")
HOME_ASSISTANT_TOKEN: str = os.getenv("HOME_ASSISTANT_TOKEN", "")

BRAVE_SEARCH_API_KEY: str = os.getenv("BRAVE_SEARCH_API_KEY", "")
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
