"""Run the Sentinel agent: python -m agents.sentinel"""

import asyncio
import sys
from pathlib import Path

from agents.sentinel.agent import SentinelAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = SentinelAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nSentinel shutting down...")


if __name__ == "__main__":
    main()
