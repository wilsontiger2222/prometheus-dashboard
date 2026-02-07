"""Tests for the Forge git operations module."""

import pytest
from unittest.mock import AsyncMock, patch
from agents.forge.git_ops import GitOps, GitResult


class TestGitResult:
    def test_git_result_success(self):
        result = GitResult(success=True, output="Already up to date.", command="git pull")
        assert result.success is True
        assert result.output == "Already up to date."

    def test_git_result_to_dict(self):
        result = GitResult(success=False, output="error: failed", command="git push")
        d = result.to_dict()
        assert d["success"] is False
        assert d["output"] == "error: failed"
        assert d["command"] == "git push"


class TestGitOps:
    @pytest.mark.asyncio
    async def test_run_command_success(self):
        ops = GitOps(workspace="/tmp/test")
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"output text", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ops._run_command("git", "status")

        assert result.success is True
        assert result.output == "output text"

    @pytest.mark.asyncio
    async def test_run_command_failure(self):
        ops = GitOps(workspace="/tmp/test")
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a repo"))
        mock_proc.returncode = 128

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await ops._run_command("git", "status")

        assert result.success is False
        assert "fatal" in result.output

    @pytest.mark.asyncio
    async def test_status(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="On branch main\nnothing to commit", command="git status"
        ))
        result = await ops.status()
        assert result.success is True
        assert "main" in result.output

    @pytest.mark.asyncio
    async def test_pull(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="Already up to date.", command="git pull"
        ))
        result = await ops.pull()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_commit(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="[main abc1234] test commit", command="git commit"
        ))
        result = await ops.commit("test commit")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_push(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="To github.com:user/repo.git\nmain -> main", command="git push"
        ))
        result = await ops.push()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_clone(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="Cloning into 'repo'...", command="git clone"
        ))
        result = await ops.clone("https://github.com/user/repo.git")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_log(self):
        ops = GitOps(workspace="/tmp/test")
        ops._run_command = AsyncMock(return_value=GitResult(
            success=True, output="abc1234 Initial commit\ndef5678 Add feature", command="git log"
        ))
        result = await ops.log(count=2)
        assert result.success is True
        assert "abc1234" in result.output
