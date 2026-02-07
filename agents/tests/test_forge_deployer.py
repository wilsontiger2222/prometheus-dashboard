"""Tests for the Forge deployer module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.forge.deployer import Deployer, DeployResult
from agents.forge.git_ops import GitResult


class TestDeployResult:
    def test_deploy_result_creation(self):
        result = DeployResult(
            success=True,
            repo="prometheus-dashboard",
            steps=["pull", "test", "restart"],
            output="All steps passed",
        )
        assert result.success is True
        assert result.repo == "prometheus-dashboard"

    def test_deploy_result_to_dict(self):
        result = DeployResult(
            success=False, repo="repo", steps=["pull"], output="failed"
        )
        d = result.to_dict()
        assert d["success"] is False
        assert d["repo"] == "repo"


class TestDeployer:
    @pytest.mark.asyncio
    async def test_deploy_pull_success(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Already up to date.", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(True, "OK"))

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner)
        result = await deployer.deploy(repo="test-repo", run_tests=False, restart_services=[])

        assert result.success is True
        mock_git.pull.assert_called_once()

    @pytest.mark.asyncio
    async def test_deploy_pull_failure_stops_pipeline(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=False, output="merge conflict", command="git pull"
        ))

        deployer = Deployer(git_ops=mock_git)
        result = await deployer.deploy(repo="test-repo")

        assert result.success is False
        assert "pull" in result.steps

    @pytest.mark.asyncio
    async def test_deploy_with_tests(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Updated", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(True, "5 passed"))

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner)
        result = await deployer.deploy(repo="test-repo", run_tests=True)

        assert result.success is True
        mock_runner.assert_called_once()
        assert "test" in result.steps

    @pytest.mark.asyncio
    async def test_deploy_test_failure_stops_pipeline(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Updated", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(False, "2 failed"))

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner)
        result = await deployer.deploy(repo="test-repo", run_tests=True)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_deploy_with_restart(self):
        mock_git = AsyncMock()
        mock_git.pull = AsyncMock(return_value=GitResult(
            success=True, output="Updated", command="git pull"
        ))
        mock_runner = AsyncMock(return_value=(True, "OK"))
        mock_restarter = AsyncMock(return_value=True)

        deployer = Deployer(git_ops=mock_git, test_runner=mock_runner, restarter=mock_restarter)
        result = await deployer.deploy(
            repo="test-repo", run_tests=False, restart_services=["sentinel"]
        )

        assert result.success is True
        mock_restarter.assert_called_once_with("sentinel")
        assert "restart" in result.steps
