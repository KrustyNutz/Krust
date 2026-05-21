#!/usr/bin/env python3
"""
J.A.R.V.I.S — Just A Rather Very Intelligent System
Personal AI assistant, persistent companion, technical partner.

Usage:
    python main.py              # Interactive terminal (default)
    python main.py --voice      # Voice mode
    python main.py --api        # Start local REST API (port 7777)
    python main.py --api --cli  # Run both simultaneously
"""
from __future__ import annotations

import argparse
import sys
import os

# Ensure J.A.R.V.I.S directory is in path
sys.path.insert(0, os.path.dirname(__file__))


def check_api_key():
    from config.settings import ANTHROPIC_API_KEY
    if not ANTHROPIC_API_KEY:
        print("\n[JARVIS] Error: ANTHROPIC_API_KEY not set.")
        print("  Copy .env.example to .env and fill in your API key.\n")
        sys.exit(1)


def build_system():
    from core.memory import Memory
    from core.brain import Brain
    from tools.registry import build_registry
    from config.settings import HOME_ASSISTANT_URL

    memory = Memory()
    registry = build_registry(memory=memory)

    # Optional: Home Automation tools
    if HOME_ASSISTANT_URL:
        from tools.home_automation import register_ha_tools
        register_ha_tools(registry)

    # Optional: Trading tools
    from agents.trading import register_trading_tools
    register_trading_tools(registry)

    brain = Brain(memory=memory, tools=registry)
    return brain, memory


def main():
    parser = argparse.ArgumentParser(description="J.A.R.V.I.S Personal AI System")
    parser.add_argument("--voice", action="store_true", help="Start in voice mode")
    parser.add_argument("--api", action="store_true", help="Start local REST API on port 7777")
    parser.add_argument("--cli", action="store_true", help="Run CLI alongside API")
    parser.add_argument("--api-host", default="127.0.0.1", help="API host (default: 127.0.0.1)")
    parser.add_argument("--api-port", type=int, default=7777, help="API port (default: 7777)")
    parser.add_argument("--query", "-q", type=str, help="Single query mode — run one query and exit")
    args = parser.parse_args()

    check_api_key()
    brain, memory = build_system()

    # Single query mode
    if args.query:
        import sys
        for event_type, payload in brain.chat(args.query):
            if event_type == "text":
                print(payload, end="", flush=True)
        print()
        return

    # Voice pipeline (optional)
    voice_pipeline = None
    if args.voice:
        try:
            from voice.pipeline import VoicePipeline
            voice_pipeline = VoicePipeline(brain)
        except ImportError:
            print("[JARVIS] Voice dependencies not installed. Run without --voice or install voice deps.")
            sys.exit(1)

    # API mode
    if args.api:
        import threading
        from interfaces.api import start_api
        api_thread = threading.Thread(
            target=start_api,
            args=(brain, memory, args.api_host, args.api_port),
            daemon=True,
        )
        api_thread.start()
        print(f"[JARVIS] API running at http://{args.api_host}:{args.api_port}")

        if not args.cli and not args.voice:
            print("[JARVIS] API mode. Ctrl+C to stop.")
            try:
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            return

    # Voice-only mode
    if args.voice and not args.cli:
        if voice_pipeline:
            voice_pipeline.start_voice_mode()
        return

    # CLI mode (default)
    from interfaces.cli import CLI
    cli = CLI(brain=brain, memory=memory, voice_pipeline=voice_pipeline)
    cli.run()


if __name__ == "__main__":
    main()
