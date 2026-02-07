"""Herald agent — social media & content creation."""

import asyncio
from agents.shared.base_agent import BaseAgent
from agents.herald.composer import Composer
from agents.herald.queue import ReviewQueue


class HeraldAgent(BaseAgent):
    """Hybrid content creation and publishing agent."""

    def __init__(self, **kwargs):
        super().__init__(name="herald", **kwargs)

        self._platforms = self.config.get("platforms", ["twitter"])
        self._require_approval = self.config.get("require_approval", True)
        self._schedule_interval = self.config.get("schedule_interval_seconds", 3600)

        self._composer = Composer(ai_client=self.ai, agent_name="herald")
        self._queue = ReviewQueue()

    async def run(self):
        """Herald listens for triggers from other agents."""
        await self.bus.subscribe("sentinel/alerts", self._on_sentinel_alert)
        await self.bus.subscribe("forge/deploys", self._on_forge_deploy)

        while self._running:
            await asyncio.sleep(self._schedule_interval)

    async def on_dispatch(self, message: dict):
        payload = message.get("payload", {})
        task = payload.get("task", "")
        args = payload.get("args", "")

        if task == "compose":
            await self._handle_compose(args, payload)
        elif task == "approve":
            self._queue.approve(args)
            await self.bus.publish(
                "herald/status",
                {"action": "approved", "draft_id": args},
                sender="herald",
            )
        elif task == "reject":
            self._queue.reject(args)
            await self.bus.publish(
                "herald/status",
                {"action": "rejected", "draft_id": args},
                sender="herald",
            )
        elif task == "status":
            pending = self._queue.get_pending()
            approved = self._queue.get_approved()
            await self.bus.publish(
                "herald/status",
                {
                    "pending": len(pending),
                    "approved": len(approved),
                    "total": len(self._queue.get_all()),
                },
                sender="herald",
            )
        elif task == "list":
            drafts = [d.to_dict() for d in self._queue.get_all()]
            await self.bus.publish(
                "herald/status",
                {"drafts": drafts},
                sender="herald",
            )

    async def _handle_compose(self, topic: str, payload: dict):
        """Compose a new draft and add to review queue."""
        platform = payload.get("platform", self._platforms[0])
        context = payload.get("context", "")
        draft = await self._composer.compose(
            topic=topic,
            platform=platform,
            trigger="manual",
            context=context,
        )
        self._queue.add(draft)
        self.logger.info(f"Draft created: {draft.id} — '{topic}'")

        if self._require_approval:
            await self.alert_telegram(
                f"New draft [{draft.id}]: {draft.content[:100]}"
            )

        await self.bus.publish(
            "herald/posts",
            {"action": "draft_created", "draft": draft.to_dict()},
            sender="herald",
        )

    async def _handle_trigger(self, trigger: str, topic: str, context: str = ""):
        """Handle a trigger from another agent — compose a draft."""
        platform = self._platforms[0] if self._platforms else "twitter"
        draft = await self._composer.compose(
            topic=topic,
            platform=platform,
            trigger=trigger,
            context=context,
        )
        self._queue.add(draft)
        self.logger.info(f"Trigger draft [{trigger}]: {draft.id}")

        if self._require_approval:
            await self.alert_telegram(
                f"Auto-draft [{draft.id}] from {trigger}: {draft.content[:100]}"
            )

    async def _on_sentinel_alert(self, channel: str, message: dict):
        """Handle trading alerts — draft a post about wins."""
        payload = message.get("payload", {})
        if payload.get("type") == "take_profit":
            symbol = payload.get("symbol", "")
            pnl = payload.get("pnl", 0)
            await self._handle_trigger(
                trigger="sentinel",
                topic=f"{symbol} trade win",
                context=f"Closed {symbol} position with PnL: {pnl}",
            )

    async def _on_forge_deploy(self, channel: str, message: dict):
        """Handle deploy events — draft a post about releases."""
        payload = message.get("payload", {})
        if payload.get("success"):
            repo = payload.get("repo", "")
            await self._handle_trigger(
                trigger="forge",
                topic=f"{repo} deployment",
                context=f"Successfully deployed {repo}",
            )
