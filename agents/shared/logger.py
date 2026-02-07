"""Structured JSON logging for Hivemind agents."""

import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record):
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": record.name.replace("hivemind.", ""),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "agent_data"):
            entry["data"] = record.agent_data
        return json.dumps(entry)


def get_agent_logger(
    agent_name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Return a named logger that emits structured JSON.

    Args:
        agent_name: Short name for the agent (e.g. "sentinel").
        log_file: Optional path; writes JSON lines to this file instead of stderr.
        level: Logging level, defaults to INFO.

    Returns:
        A ``logging.Logger`` instance named ``hivemind.<agent_name>``.
    """
    logger = logging.getLogger(f"hivemind.{agent_name}")
    logger.setLevel(level)

    if not logger.handlers:
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger
