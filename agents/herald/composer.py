"""Content composer for the Herald agent.

Uses AI to generate platform-appropriate content from topics and context.
Produces Draft objects that go through the review queue before publishing.
"""

import uuid
import time
from dataclasses import dataclass, field


@dataclass
class Draft:
    content: str
    platform: str
    trigger: str
    topic: str
    status: str = "pending"
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: float = field(default_factory=time.time)

    def approve(self):
        self.status = "approved"

    def reject(self):
        self.status = "rejected"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "platform": self.platform,
            "trigger": self.trigger,
            "topic": self.topic,
            "status": self.status,
            "created_at": self.created_at,
        }


class Composer:
    """Generate content drafts using AI."""

    COMPOSE_PROMPT = (
        "You are a social media content creator. "
        "Write a {platform} post about: {topic}. "
        "{context_line}"
        "Keep it concise, engaging, and appropriate for {platform}. "
        "Do not include hashtags unless asked. "
        "Return ONLY the post text, nothing else."
    )

    def __init__(self, ai_client, agent_name: str = "herald"):
        self._ai = ai_client
        self._agent_name = agent_name

    async def compose(
        self,
        topic: str,
        platform: str,
        trigger: str,
        context: str = "",
    ) -> Draft:
        """Generate a content draft for a given topic and platform."""
        context_line = f"Context: {context}\n" if context else ""
        prompt = self.COMPOSE_PROMPT.format(
            platform=platform,
            topic=topic,
            context_line=context_line,
        )

        content = await self._ai.call(self._agent_name, prompt)

        return Draft(
            content=content,
            platform=platform,
            trigger=trigger,
            topic=topic,
        )
