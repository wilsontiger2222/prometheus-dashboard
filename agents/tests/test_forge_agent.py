"""Tests for the ForgeAgent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from agents.forge.agent import ForgeAgent
from agents.forge.git_ops import GitResult


@pytest.fixture
def forge():
    with patch("agents.shared.base_agent.load_agent_config") as mock_config:
        mock_config.return_value = {
            "name": "forge",
            "workspace": "/tmp/forge_test",
            "github_user": "wilsontiger2222",
            "default_branch": "main",
            "auto_test": True,
            "deploy_on_push": False,
            "test_command": "python -m pytest",
            "test_timeout_seconds": 120,
        }
        agent = ForgeAgent(config_path="dummy.json")
        agent.bus = AsyncMock()
        agent._running = True
        return agent


class TestForgeAgent:
    def test_forge_name(self, forge):
        assert forge.name == "forge"

    def test_loads_config(self, forge):
        assert forge._workspace is not None
        assert forge._auto_test is True

    @pytest.mark.asyncio
    async def test_on_dispatch_status(self, forge):
        """Dispatch with task=status returns git status."""
        forge._git.status = AsyncMock(return_value=GitResult(
            success=True, output="On branch main", command="git status"
        ))
        message = {"payload": {"task": "status"}}
        await forge.on_dispatch(message)
        forge.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_deploy(self, forge):
        """Dispatch with task=deploy runs the deploy pipeline."""
        forge._deployer.deploy = AsyncMock(return_value=MagicMock(
            success=True,
            to_dict=lambda: {"success": True, "repo": "test", "steps": ["pull"]},
        ))
        message = {"payload": {"task": "deploy", "args": "prometheus-dashboard"}}
        await forge.on_dispatch(message)
        forge._deployer.deploy.assert_called_once()
        forge.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_on_dispatch_git_log(self, forge):
        """Dispatch with task=log returns git log."""
        forge._git.log = AsyncMock(return_value=GitResult(
            success=True, output="abc123 commit msg", command="git log"
        ))
        message = {"payload": {"task": "log"}}
        await forge.on_dispatch(message)
        forge.bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_run_is_idle(self, forge):
        """Forge run() should just idle (on-demand agent)."""
        forge._running = False
        await forge.run()
