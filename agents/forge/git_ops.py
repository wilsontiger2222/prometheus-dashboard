"""Git operations wrapper for the Forge agent.

Provides async wrappers around git commands using asyncio.subprocess.
All operations run in a configurable workspace directory.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GitResult:
    success: bool
    output: str
    command: str

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output": self.output,
            "command": self.command,
        }


class GitOps:
    """Async git operations in a workspace directory."""

    def __init__(self, workspace: str):
        self._workspace = Path(workspace)

    async def _run_command(self, *args: str) -> GitResult:
        """Run a shell command and return the result."""
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode().strip() or stderr.decode().strip()
        return GitResult(
            success=proc.returncode == 0,
            output=output,
            command=" ".join(args),
        )

    async def status(self) -> GitResult:
        """Run git status."""
        return await self._run_command("git", "status")

    async def pull(self, branch: str = "") -> GitResult:
        """Run git pull."""
        args = ["git", "pull"]
        if branch:
            args.extend(["origin", branch])
        return await self._run_command(*args)

    async def commit(self, message: str) -> GitResult:
        """Run git commit with the given message."""
        return await self._run_command("git", "commit", "-m", message)

    async def push(self, branch: str = "") -> GitResult:
        """Run git push."""
        args = ["git", "push"]
        if branch:
            args.extend(["origin", branch])
        return await self._run_command(*args)

    async def clone(self, repo_url: str, dest: str = "") -> GitResult:
        """Clone a repository."""
        args = ["git", "clone", repo_url]
        if dest:
            args.append(dest)
        return await self._run_command(*args)

    async def add(self, *paths: str) -> GitResult:
        """Stage files for commit."""
        return await self._run_command("git", "add", *paths)

    async def log(self, count: int = 10) -> GitResult:
        """Show recent commits."""
        return await self._run_command("git", "log", "--oneline", f"-{count}")
