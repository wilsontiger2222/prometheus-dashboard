"""Run the Watchdog agent: python -m agents.watchdog"""

import asyncio
import sys
from pathlib import Path

from agents.watchdog.agent import WatchdogAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = WatchdogAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nWatchdog shutting down...")


if __name__ == "__main__":
    main()
