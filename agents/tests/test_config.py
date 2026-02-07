import json
import pytest
from agents.shared.config import load_agent_config


def test_load_config_from_file(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "name": "sentinel",
        "watchlist": ["BTC", "ETH"],
        "mode": "paper"
    }))
    config = load_agent_config(str(config_file))
    assert config["name"] == "sentinel"
    assert config["watchlist"] == ["BTC", "ETH"]


def test_load_config_with_defaults(tmp_path):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"name": "watchdog"}))
    defaults = {"check_interval": 60, "name": "default"}
    config = load_agent_config(str(config_file), defaults=defaults)
    assert config["name"] == "watchdog"
    assert config["check_interval"] == 60


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_agent_config("/nonexistent/config.json")
