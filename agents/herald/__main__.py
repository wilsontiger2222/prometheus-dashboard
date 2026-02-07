"""Run the Herald agent: python -m agents.herald"""

import asyncio
import sys
from pathlib import Path

from agents.herald.agent import HeraldAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = HeraldAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nHerald shutting down...")


if __name__ == "__main__":
    main()
