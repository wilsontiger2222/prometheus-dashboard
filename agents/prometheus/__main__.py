"""Run the Prometheus orchestrator: python -m agents.prometheus"""

import asyncio
import sys
from pathlib import Path

from agents.prometheus.agent import PrometheusAgent

DEFAULT_CONFIG = str(Path(__file__).parent / "config.json")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    agent = PrometheusAgent(config_path=config_path)

    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        print("\nPrometheus shutting down...")


if __name__ == "__main__":
    main()
