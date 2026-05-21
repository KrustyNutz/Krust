"""Bridge between J.A.R.V.I.S and NEXUS trading engine."""
from __future__ import annotations

import sys
import json
from pathlib import Path

# Add parent directory so nexus.py can be imported
REPO_ROOT = Path(__file__).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_nexus_status() -> dict:
    """Get NEXUS trading engine status."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("nexus", REPO_ROOT / "nexus.py")
        if spec is None:
            return {"error": "nexus.py not found"}
        # NEXUS status is exposed via its REST or shared state
        # This is a lightweight status check — not a full import of the live engine
        return {
            "status": "nexus.py found",
            "path": str(REPO_ROOT / "nexus.py"),
            "note": "Start NEXUS separately with `python nexus.py`. JARVIS monitors its output.",
        }
    except Exception as e:
        return {"error": str(e)}


def get_market_summary(registry=None) -> dict:
    """Quick market summary using Alpaca if configured."""
    from config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
    if not ALPACA_API_KEY:
        return {"error": "Alpaca API key not configured. Set ALPACA_API_KEY in .env"}
    try:
        import requests
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        # Account info
        acct = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=headers, timeout=10)
        acct.raise_for_status()
        data = acct.json()
        return {
            "equity": data.get("equity"),
            "cash": data.get("cash"),
            "buying_power": data.get("buying_power"),
            "day_trade_count": data.get("daytrade_count"),
            "portfolio_value": data.get("portfolio_value"),
        }
    except Exception as e:
        return {"error": str(e)}


def register_trading_tools(registry):
    registry.register(
        "nexus_status",
        "Check NEXUS trading engine status and configuration.",
        {"type": "object", "properties": {}},
        get_nexus_status,
    )

    registry.register(
        "market_summary",
        "Get current Alpaca account summary: equity, cash, buying power.",
        {"type": "object", "properties": {}},
        get_market_summary,
    )
