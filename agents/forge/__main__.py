"""Run the Forge agent: python -m agents.forge"""

import asyncio
import sys
from pathlib import Path

from agents.forge.agent import ForgeAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = ForgeAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nForge shutting down...")


if __name__ == "__main__":
    main()
