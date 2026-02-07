"""Run the Scout agent: python -m agents.scout"""

import asyncio
import sys
from pathlib import Path

from agents.scout.agent import ScoutAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = ScoutAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nScout shutting down...")


if __name__ == "__main__":
    main()
