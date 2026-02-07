"""Tests for the structured JSON logger."""

import json
import logging
import uuid

import pytest

from agents.shared.logger import get_agent_logger


def _unique_name(base: str) -> str:
    """Return a unique logger name to avoid cross-test pollution."""
    return f"{base}_{uuid.uuid4().hex[:8]}"


def test_logger_returns_named_logger():
    name = _unique_name("sentinel")
    logger = get_agent_logger(name)
    assert logger.name == f"hivemind.{name}"


def test_logger_formats_json(tmp_path):
    log_file = tmp_path / "test.log"
    name = _unique_name("watchdog")
    logger = get_agent_logger(name, log_file=str(log_file))
    logger.info("health check passed", extra={"agent_data": {"cpu": 45}})

    content = log_file.read_text()
    record = json.loads(content.strip().split("\n")[-1])
    assert record["agent"] == name
    assert record["message"] == "health check passed"
    assert record["data"]["cpu"] == 45


def test_logger_default_level():
    name = _unique_name("scout")
    logger = get_agent_logger(name)
    assert logger.level == logging.INFO
