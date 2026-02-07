import json
import pytest
from unittest.mock import AsyncMock, patch
from agents.shared.ai_client import AIClient


@pytest.fixture
def client(tmp_path):
    return AIClient(
        gateway_url="ws://127.0.0.1:18789",
        token="test-token",
        token_log=str(tmp_path / "tokens.json"),
    )


def test_client_init(client):
    assert client._gateway_url == "ws://127.0.0.1:18789"
    assert client._total_tokens == 0


def test_token_tracking(client):
    client._track_tokens("sentinel", prompt_tokens=100, completion_tokens=50)
    assert client._total_tokens == 150
    assert client._agent_tokens["sentinel"] == 150


def test_token_log_written(client, tmp_path):
    client._track_tokens("scout", prompt_tokens=200, completion_tokens=100)
    client._save_token_log()
    log_file = tmp_path / "tokens.json"
    data = json.loads(log_file.read_text())
    assert data["total"] == 300
    assert data["by_agent"]["scout"] == 300


def test_get_usage_report(client):
    client._track_tokens("sentinel", 100, 50)
    client._track_tokens("scout", 200, 100)
    report = client.get_usage_report()
    assert report["total"] == 450
    assert report["by_agent"]["sentinel"] == 150
    assert report["by_agent"]["scout"] == 300
